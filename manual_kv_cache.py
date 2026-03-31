from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple
from contextlib import nullcontext

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import time

LegacyKV = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
KVCache = Any
PrefillAttentionMode = Literal["default", "math", "flash"]


try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:
    SDPBackend = None
    sdpa_kernel = None


@dataclass
class ManualGenConfig:
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_k: int = 50
    do_sample: bool = True
    eos_token_id: Optional[int] = None


def _normalize_cache(past_key_values) -> Optional[KVCache]:
    if past_key_values is None:
        return None
    return past_key_values


def _cache_seq_len(cache: Optional[KVCache]) -> int:
    if cache is None:
        return 0
    if isinstance(cache, tuple):
        return cache[0][0].shape[2]
    if hasattr(cache, "get_seq_length"):
        return cache.get_seq_length()
    raise TypeError(f"Unsupported cache type for seq length: {type(cache)}")


def _merge_cache(old_cache: Optional[KVCache], new_cache: Optional[KVCache]) -> Optional[KVCache]:
    if old_cache is None:
        return new_cache
    if new_cache is None:
        return old_cache

    if not isinstance(old_cache, tuple) or not isinstance(new_cache, tuple):
        return new_cache

    old_len = _cache_seq_len(old_cache)
    new_len = _cache_seq_len(new_cache)

    if new_len == old_len + 1:
        return new_cache

    if new_len == 1:
        merged_layers = []
        for (old_k, old_v), (new_k, new_v) in zip(old_cache, new_cache):
            merged_layers.append((torch.cat([old_k, new_k], dim=2), torch.cat([old_v, new_v], dim=2)))
        return tuple(merged_layers)

    if new_len <= old_len:
        return old_cache

    return new_cache


def _sample_next_token(logits: torch.Tensor, cfg: ManualGenConfig) -> torch.Tensor:
    if not cfg.do_sample:
        return torch.argmax(logits, dim=-1)

    if cfg.temperature <= 0:
        return torch.argmax(logits, dim=-1)

    probs = torch.softmax(logits / cfg.temperature, dim=-1)

    if cfg.top_k > 0:
        topk_vals, topk_idx = torch.topk(probs, k=min(cfg.top_k, probs.shape[-1]), dim=-1)
        topk_vals = topk_vals / torch.sum(topk_vals, dim=-1, keepdim=True)
        sampled = torch.multinomial(topk_vals, num_samples=1)
        return torch.gather(topk_idx, dim=-1, index=sampled).squeeze(-1)

    return torch.multinomial(probs, num_samples=1).squeeze(-1)


class ManualKVCacheGenerator:
    def __init__(self, model, tokenizer, prefill_attention_mode: PrefillAttentionMode = "default"):
        self.model = model
        self.tokenizer = tokenizer
        self.prefill_attention_mode: PrefillAttentionMode = prefill_attention_mode

    def set_prefill_attention_mode(self, mode: PrefillAttentionMode) -> None:
        self.prefill_attention_mode = mode

    def _prefill_attention_context(self):
        # Force a deterministic SDPA backend for prefill timing experiments.
        if not torch.cuda.is_available():
            return nullcontext()

        if sdpa_kernel is not None and SDPBackend is not None:
            if self.prefill_attention_mode == "flash":
                return sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION])

            if self.prefill_attention_mode == "math":
                return sdpa_kernel(backends=[SDPBackend.MATH])

            return nullcontext()

        if not hasattr(torch.backends.cuda, "sdp_kernel"):
            return nullcontext()

        if self.prefill_attention_mode == "flash":
            return torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)

        if self.prefill_attention_mode == "math":
            return torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)

        return nullcontext()

    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        with self._prefill_attention_context():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
        logits = outputs.logits[:, -1, :]
        cache = _normalize_cache(outputs.past_key_values)
        return logits, cache

    @torch.inference_mode()
    def decode_step(
        self,
        next_input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cache: Optional[KVCache],
    ):
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids = position_ids[:, -1:].contiguous()

        outputs = self.model(
            input_ids=next_input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )

        logits = outputs.logits[:, -1, :]
        new_cache = _normalize_cache(outputs.past_key_values)
        merged_cache = _merge_cache(cache, new_cache)
        return logits, merged_cache

    @torch.inference_mode()
    def generate(self, prompt: str, cfg: ManualGenConfig) -> str:
        batch = self.tokenizer(prompt, return_tensors="pt")
        input_ids = batch["input_ids"].to(self.model.device)
        attention_mask = batch["attention_mask"].to(self.model.device)

        logits, cache = self.prefill(input_ids=input_ids, attention_mask=attention_mask)

        next_token = _sample_next_token(logits, cfg).unsqueeze(-1)
        generated = [next_token]

        eos_token_id = cfg.eos_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id

        for _ in range(cfg.max_new_tokens - 1):
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)],
                dim=-1,
            )

            logits, cache = self.decode_step(
                next_input_ids=next_token,
                attention_mask=attention_mask,
                cache=cache,
            )

            next_token = _sample_next_token(logits, cfg).unsqueeze(-1)
            generated.append(next_token)

            if eos_token_id is not None and torch.all(next_token.squeeze(-1) == eos_token_id):
                break

        new_tokens = torch.cat(generated, dim=1)
        full_ids = torch.cat([input_ids, new_tokens], dim=1)
        return self.tokenizer.decode(full_ids[0], skip_special_tokens=True)


def build_generator(
    model_path: str,
    dtype: torch.dtype = torch.float16,
    prefill_attention_mode: PrefillAttentionMode = "default",
    attn_implementation: Optional[str] = "sdpa",
) -> ManualKVCacheGenerator:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": "auto",
    }
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation

    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    except TypeError:
        model_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    model.eval()
    return ManualKVCacheGenerator(
        model=model,
        tokenizer=tokenizer,
        prefill_attention_mode=prefill_attention_mode,
    )


if __name__ == "__main__":
    model_path = "/home/scm/mistral_models/7B-Instruct-v0.3"
    prompt = "[INST] 请详细介绍什么是GQA。[/INST]"

    generator = build_generator(model_path=model_path, dtype=torch.float16)
    config = ManualGenConfig(
        max_new_tokens=80,
        temperature=0.7,
        top_k=40,
        do_sample=True,
    )

    text = generator.generate(prompt=prompt, cfg=config)
    # 把text写入文件
    with open("generated_output.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print(text)