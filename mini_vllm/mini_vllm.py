import asyncio
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from fastapi import FastAPI
from pydantic import BaseModel

# ======================
# 1. 加载模型
# ======================
model_path = "/home/scm/mistral_models/7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map=None  # 自动分配到双卡
)

model.eval()

# 使用第一块可用 CUDA 设备作为输入放置设备，避免与 device_map="auto" 冲突
if torch.cuda.is_available():
    hf_device_map = getattr(model, "hf_device_map", {})
    device = next(
        (
            d
            for d in hf_device_map.values()
            if isinstance(d, str) and d.startswith("cuda")
        ),
        "cuda:1",
    )
else:
    device = "cpu"

model.to(device)

# ======================
# 2. Request 定义
# ======================
class Request:
    def __init__(self, prompt, max_new_tokens=20):
        self.prompt = prompt
        # 存储为 1D token 序列，便于 pad_sequence 正确对齐不同长度请求
        self.input_ids = tokenizer(prompt, return_tensors="pt").input_ids.squeeze(0).to(device)
        self.generated = []
        self.max_new_tokens = max_new_tokens
        self.finished = False

        self.stage = "PREFILL"  # or DECODE, 实现Prefill + Decode分阶段处理

        # [((1, num_heads, seq_len, head_dim), (1, num_heads, seq_len, head_dim)), ...] 每层的 KV cache
        self.KV_cache = None  # 存储 KV cache，支持 Prefill + Decode, list of (k, v) tuples per layer

    def step(self, next_token):
        self.generated.append(next_token.item())
        if len(self.generated) >= self.max_new_tokens or next_token.item() == tokenizer.eos_token_id:
            self.finished = True

    def get_output(self):
        return tokenizer.decode(self.generated)


# ======================
# 3. 全局请求队列
# ======================
request_queue = asyncio.Queue()
waiting_queue = []


# ======================
# 4. Dynamic Batching Worker
# ======================
BATCH_SIZE = 4
TIMEOUT = 0.01  # 10ms


async def scheduler():
    global waiting_queue

    active_requests = []

    while True:
        # ======================
        # Step 1: 收集新请求
        # ======================
        try:
            while True:
                req = request_queue.get_nowait()
                waiting_queue.append(req)
        except asyncio.QueueEmpty:
            pass

        # admission control
        while waiting_queue and len(active_requests) < BATCH_SIZE:
            active_requests.append(waiting_queue.pop(0))

        if not active_requests:
            await asyncio.sleep(0.001)
            continue

        # ======================
        # Step 2: 拆分 PREFILL / DECODE
        # ======================
        prefill_reqs = [r for r in active_requests if r.stage == "PREFILL"]
        decode_reqs  = [r for r in active_requests if r.stage == "DECODE"]

        new_active = []

        # ======================
        # Step 3: PREFILL（单独 forward）
        # ======================
        if prefill_reqs:
            input_ids_list = [r.input_ids for r in prefill_reqs]
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids_list,
                batch_first=True,
                padding_value=tokenizer.pad_token_id
            ).to(device)

            attention_mask = (input_ids != tokenizer.pad_token_id).long()

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True
                )

            past = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            next_tokens = torch.argmax(logits, dim=-1)

            # 写回 KV + 更新状态
            for i, req in enumerate(prefill_reqs):
                req.step(next_tokens[i])
                req.input_ids = torch.cat(
                    [req.input_ids, next_tokens[i].view(1)], dim=0
                )

                # 提取该 request 的 KV（无 padding）
                actual_len = req.input_ids.size(0)

                req_kv = []
                for layer in past:
                    k, v = layer[0], layer[1]
                    k_i = k[i:i+1, :, :actual_len, :].clone()
                    v_i = v[i:i+1, :, :actual_len, :].clone()
                    req_kv.append((k_i, v_i))

                req.KV_cache = req_kv
                req.stage = "DECODE"

                if not req.finished:
                    new_active.append(req)

        # ======================
        # Step 4: DECODE（单独 forward）
        # ======================
        if decode_reqs:
            input_ids = torch.stack(
                [r.input_ids[-1:] for r in decode_reqs]
            ).to(device)  # [B,1]

            # ===== 构建 batched KV =====
            num_layers = len(decode_reqs[0].KV_cache)
            current_kv_lens = [r.KV_cache[0][0].size(2) for r in decode_reqs]
            max_kv_len = max(current_kv_lens)

            batched_past = DynamicCache()

            for layer_idx in range(num_layers):
                keys_to_cat = []
                values_to_cat = []

                for i, req in enumerate(decode_reqs):
                    k, v = req.KV_cache[layer_idx]

                    pad_len = max_kv_len - k.size(2)
                    if pad_len > 0:
                        pad_shape = (1, k.size(1), pad_len, k.size(3))
                        k = torch.cat([
                            k,
                            torch.zeros(pad_shape, device=device, dtype=k.dtype)
                        ], dim=2)
                        v = torch.cat([
                            v,
                            torch.zeros(pad_shape, device=device, dtype=v.dtype)
                        ], dim=2)

                    keys_to_cat.append(k)
                    values_to_cat.append(v)

                batched_past.update(
                    key_states=torch.cat(keys_to_cat, dim=0),
                    value_states=torch.cat(values_to_cat, dim=0),
                    layer_idx=layer_idx
                )

            # ===== 正确 attention mask =====
            total_len = max_kv_len + 1
            masks = []
            for i in range(len(decode_reqs)):
                valid_len = current_kv_lens[i] + 1
                m = torch.cat([
                    torch.ones(valid_len, device=device),
                    torch.zeros(total_len - valid_len, device=device)
                ])
                masks.append(m)

            attention_mask = torch.stack(masks).long()

            # ===== forward =====
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    past_key_values=batched_past
                )

            new_past = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            next_tokens = torch.argmax(logits, dim=-1)

            # ===== 写回 KV =====
            for i, req in enumerate(decode_reqs):
                req.step(next_tokens[i])
                req.input_ids = torch.cat(
                    [req.input_ids, next_tokens[i].view(1)], dim=0
                )

                old_len = current_kv_lens[i]
                new_len = old_len + 1

                req_kv = []
                for layer in new_past:
                    k, v = layer[0], layer[1]
                    k_i = k[i:i+1, :, :new_len, :]
                    v_i = v[i:i+1, :, :new_len, :]
                    req_kv.append((k_i, v_i))

                req.KV_cache = req_kv

                if not req.finished:
                    new_active.append(req)

        # ======================
        # Step 5: 更新 active
        # ======================
        active_requests = new_active

        if not active_requests:
            await asyncio.sleep(0.001)

# ======================
# 5. FastAPI 接口
# ======================
app = FastAPI()


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 20


@app.post("/generate")
async def generate(req: GenerateRequest):
    r = Request(req.prompt, req.max_new_tokens)
    await request_queue.put(r)

    # 简单等待完成（v0版本）
    while not r.finished:
        await asyncio.sleep(0.01)

    return {"output": r.get_output()}


# ======================
# 6. 启动 worker
# ======================
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(scheduler())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)