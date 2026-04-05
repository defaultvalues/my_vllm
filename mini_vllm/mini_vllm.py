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
# 1. KVCache 管理
# ======================

class KVCache:
    def __init__(self, num_blocks, num_layers, num_heads, head_dim, block_size, device, dtype):
        self.num_blocks = num_blocks  # KV cache 的总块数，决定了最大并发请求数和最大序列长度, 一个block存的KV cache一定来自同一个请求
        self.block_size = block_size  # 每块的 token 数量，决定了每块能存储多少 KV

        self.k_cache = torch.zeros(
            num_blocks, num_layers, num_heads, block_size, head_dim,
            device=device, dtype=dtype
        )
        self.v_cache = torch.zeros_like(self.k_cache)

        self.block_fill = torch.zeros(num_blocks, dtype=torch.int32, device=device)  # 最大填充长度为 block_size

        self.free_blocks = list(range(num_blocks))  # 表示空闲块的索引列表

    def alloc_block(self):
        assert self.free_blocks, "KV cache OOM"
        idx = self.free_blocks.pop()
        self.block_fill[idx] = 0
        return idx

    def free_block(self, idx):
        self.free_blocks.append(idx)


def append_kv(req, new_k, new_v, kv_cache):
    # new_k: [num_layers, num_heads, 1, head_dim]
    """
    req: Request 对象，包含 block_table 和 seq_len
    new_k, new_v: 当前 step 生成的 KV, shape [num_layers, num_heads, 1, head_dim]
    kv_cache: KVCache 对象，管理所有请求的 KV 存储和分配
    """

    if not req.block_table or \
       kv_cache.block_fill[req.block_table[-1]] == kv_cache.block_size:  # 当前 block 已满，分配新块

        new_block = kv_cache.alloc_block()  # 从 KVCache 获取一个新的 block 索引
        req.block_table.append(new_block)  # 将新 block 加入 request 的 block_table

    block_id = req.block_table[-1]  # 获取当前使用的 block 索引
    pos = kv_cache.block_fill[block_id]  # 当前 block 已填充的 token 数量

    kv_cache.k_cache[block_id, :, :, pos:pos+1, :] = new_k  # 写入新的 KV 到全局 KV cache 的正确位置， shape: [num_layers, num_heads, 1(seq_len), head_dim]
    kv_cache.v_cache[block_id, :, :, pos:pos+1, :] = new_v

    kv_cache.block_fill[block_id] += 1  # 更新当前 block 的填充长度
    req.seq_len += 1  # 更新 request 的 KV cache 长度


def build_past_from_blocks(requests, kv_cache, model): 
    """
    从 block-based KV 重建 HuggingFace 的 past_key_values (dense KV)

    这是一个“过渡函数”：
        - 现在为了兼容 HF 必须做 concat + padding
        - 未来接 FlashInfer 会完全删除

    输入:
        requests: List[Request]
        kv_cache: 全局 KV pool
        model: 用于获取 num_layers

    返回:
        past: DynamicCache (HF 使用)
        max_len: 当前 batch 的最大序列长度
    """

    num_layers = model.config.num_hidden_layers

    # ===== Step 1: 找 batch 内最大长度（用于 padding）=====
    max_len = max(r.seq_len for r in requests)  
    

    # 用于存每一层的 KV（最后会拼成 batch）=====
    batch_k = [[] for _ in range(num_layers)]
    batch_v = [[] for _ in range(num_layers)]

    # ===== Step 2: 遍历每个 request =====
    for r in requests:

        # 用于拼接该 request 的所有 KV
        k_list = []
        v_list = []

        # ===== Step 2.1: 遍历 block_table =====
        for bid in r.block_table:

            fill = kv_cache.block_fill[bid] # 当前 block 已填充的 token 数量

            # 从全局 KV pool 取出该 block 的有效部分
            # shape: [num_layers, num_heads, fill, head_dim]
            k_block = kv_cache.k_cache[bid, :, :, :fill, :]
            v_block = kv_cache.v_cache[bid, :, :, :fill, :]

            k_list.append(k_block)
            v_list.append(v_block)

        # ===== Step 2.2: 拼接所有 block（时间维）=====
        # dim=2 是 token 维， 按sequence length拼接，得到 shape: [num_layers, num_heads, seq_len, head_dim]
        
        k_cat = torch.cat(k_list, dim=2)
        v_cat = torch.cat(v_list, dim=2)

        # ===== Step 2.3: padding（HF 必须）=====
        pad_len = max_len - k_cat.size(2)

        if pad_len > 0:
            # padding tensor
            pad_k = torch.zeros(
                num_layers,
                k_cat.size(1),   # num_heads
                pad_len,
                k_cat.size(3),   # head_dim
                device=k_cat.device,
                dtype=k_cat.dtype
            )

            pad_v = torch.zeros_like(pad_k)

            k_cat = torch.cat([k_cat, pad_k], dim=2)
            v_cat = torch.cat([v_cat, pad_v], dim=2)

        # ===== Step 2.4: 按 layer 拆开 =====
        for l in range(num_layers):
            # 每个元素 shape:
            # [1, num_heads, max_len, head_dim]，第一维度是 batch 维，后面会拼成 batch
            batch_k[l].append(k_cat[l:l+1])
            batch_v[l].append(v_cat[l:l+1])

    # ===== Step 3: 构建 HF 的 past_key_values =====
    past = DynamicCache()

    for l in range(num_layers):
        past.update(
            key_states=torch.cat(batch_k[l], dim=0),   # [batch_size, num_heads, max_len, head_dim]
            value_states=torch.cat(batch_v[l], dim=0),
            layer_idx=l
        )

    return past, max_len

def write_prefill_kv_batch(requests, past_key_values, kv_cache):
    """
    支持 batch 的 prefill KV 写入
    """

    past = past_key_values
    num_layers = len(past)

    for i, req in enumerate(requests):

        # 每个 request 自己的有效长度
        valid_len = len(req.input_ids) 

        for t in range(valid_len):

            k_list = []
            v_list = []

            for past_layer in past:
                k, v = past_layer[0], past_layer[1]  # shape: [batch_size, num_heads, seq_len, head_dim]

                k_t = k[i, :, t:t+1, :]  # shape: [num_heads, 1, head_dim]
                v_t = v[i, :, t:t+1, :]

                k_list.append(k_t.unsqueeze(0))
                v_list.append(v_t.unsqueeze(0))

            new_k = torch.cat(k_list, dim=0)  # shape: [num_layers, num_heads, 1, head_dim]
            new_v = torch.cat(v_list, dim=0)

            append_kv(req, new_k, new_v, kv_cache)  # 逐token写入 KV cache

def extract_new_kv(requests, past_key_values, kv_cache):
    """
    对于decoding阶段的请求, 从 HF 的 past_key_values 中提取当前 step 生成的 KV, 并写入全局 KV cache
     - requests: 当前 batch 的请求列表
     - past_key_values: HF 模型输出的 past_key_values, 包含当前 step 生成的 KV
     - kv_cache: 全局 KVCache 对象，用于存储所有请求的 KV
    """

    past = past_key_values

    for i, req in enumerate(requests):
        
        k_list = []
        v_list = []

        for past_layer in past:
            k, v = past_layer[0], past_layer[1]  # shape: [batch_size, num_heads, seq_len, head_dim]

            k_i = k[i:i+1, :, -1:, :]  # 取当前 step 生成的 KV，shape: [1, num_heads, 1, head_dim]
            v_i = v[i:i+1, :, -1:, :]

            k_list.append(k_i)
            v_list.append(v_i)

        new_k = torch.cat(k_list, dim=0)  # shape: [num_layers, num_heads, 1, head_dim]
        new_v = torch.cat(v_list, dim=0)

        append_kv(req, new_k, new_v, kv_cache)  # 写入全局 KV cache

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

        self.block_table = []  # List(int), 存储该请求使用的 block 索引，支持 Prefill + Decode
        self.seq_len = 0  # 当前已生成的总长度（输入 + 输出）

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
    global waiting_queue, kv_cache

    active_requests = []

    kv_cache = KVCache(
        num_blocks=16,  # 假设最多支持16个并发请求（每个请求最多使用一个 block，实际可以更灵活）
        num_layers=model.config.num_hidden_layers,
        num_heads=model.config.num_key_value_heads,
        head_dim=model.config.hidden_size // model.config.num_attention_heads,
        block_size=32,  # 每块最多存32个 token 的 KV，实际使用中可以根据请求长度动态调整
        device=device,
        dtype=model.dtype
    )

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

            logits = outputs.logits[:, -1, :]
            next_tokens = torch.argmax(logits, dim=-1)

            write_prefill_kv_batch(prefill_reqs, outputs.past_key_values, kv_cache)

            # 写回 KV + 更新状态
            for i, req in enumerate(prefill_reqs):
                req.step(next_tokens[i])
                req.input_ids = torch.cat(
                    [req.input_ids, next_tokens[i].view(1)], dim=0
                )
                
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
            
            current_kv_lens = [r.seq_len for r in decode_reqs]
            
            batched_past, max_kv_len = build_past_from_blocks(decode_reqs, kv_cache, model)  # 构建 HF 的 past_key_values，返回当前 batch 的最大 KV 长度（用于构建 attention mask）


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

            extract_new_kv(decode_reqs, outputs.past_key_values, kv_cache)
            
            logits = outputs.logits[:, -1, :]
            next_tokens = torch.argmax(logits, dim=-1)

            # ===== 写回 KV =====
            for i, req in enumerate(decode_reqs):
                req.step(next_tokens[i])
                req.input_ids = torch.cat(
                    [req.input_ids, next_tokens[i].view(1)], dim=0
                )

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