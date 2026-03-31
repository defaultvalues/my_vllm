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
        "cuda:0",
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

prefill_queue = []
decode_queue = []



# ======================
# 4. Dynamic Batching Worker
# ======================
BATCH_SIZE = 4
TIMEOUT = 0.01  # 10ms


async def scheduler():
    global waiting_queue, prefill_queue, decode_queue

    while True:
        # Step 1: 收集新请求（dynamic batching）
        try:
            while True:
                req = request_queue.get_nowait()
                waiting_queue.append(req)
        except asyncio.QueueEmpty:
            pass

        # 构建prefill队列
        while waiting_queue and len(prefill_queue) < BATCH_SIZE:
            prefill_queue.append(waiting_queue.pop(0))
        
        if prefill_queue:
            input_ids_list = [req.input_ids for req in prefill_queue]
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id or 0
            )
            attention_mask = (input_ids != (tokenizer.pad_token_id or 0)).long()
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, 
                                attention_mask=attention_mask,
                                use_cache=True)  # 预填充阶段启用 KV cache
                
                past = outputs.past_key_values  # 获取 KV cache, DynamicCache
                logits = outputs.logits[:, -1, :]  # 只取最后一个 token 的 logits，供 decode 阶段使用

                # 这里可以提取 KV cache，支持后续 decode 阶段
                for i, req in enumerate(prefill_queue):
                    actual_len = req.input_ids.size(0)  # 实际的序列长度
                    req_kv_list = []
                    
                    # 遍历每一层（指模型的每一层 注意力 模块），提取对应请求的 KV
                    for past_layer in past:
                        # 提取第 i 个请求的 Key 和 Value
                        k_layer, v_layer = past_layer[0], past_layer[1]  # k_layer, v_layer: (B, num_heads, seq_len, head_dim)
                        k_i = k_layer[i:i+1, :, :actual_len, :].clone()  # 取第 i 个请求的 KV
                        v_i = v_layer[i:i+1, :, :actual_len, :].clone()
                        req_kv_list.append((k_i, v_i))
                    
                    # 将这个请求的专属 KV 存回 Request 对象
                    req.KV_cache = req_kv_list
                
                next_tokens = torch.argmax(logits, dim=-1)
            
            new_decode_queue = []
            for i, req in enumerate(prefill_queue):
                req.step(next_tokens[i])

                req.stage = "DECODE"  # 切换到 decode 阶段

                if not req.finished:
                    # 更新输入（append token）
                    req.input_ids = torch.cat([req.input_ids, next_tokens[i].view(1)], dim=0)
                    new_decode_queue.append(req)
            
            decode_queue.extend(new_decode_queue)
            prefill_queue = []  # 清空 prefill 队列


        # ======================
        # 处理 decode 阶段的请求
        # ======================
        if decode_queue:
            # 1. 准备 input_ids [Batch, 1]
            input_ids = torch.stack([req.input_ids[-1:] for req in decode_queue]).to(device)

            # --- 准备 Batch 化的 KV Cache 和 Attention Mask ---

            # 1. 创建空的 DynamicCache 对象
            batched_past = DynamicCache()

            # 2. 拼接每一层的 Batch KV
            num_layers = len(decode_queue[0].KV_cache)
            current_kv_lens = [req.KV_cache[0][0].size(2) for req in decode_queue]
            max_kv_len = max(current_kv_lens)

            for layer_idx in range(num_layers):
                keys_to_cat = []
                values_to_cat = []
                
                for i, req in enumerate(decode_queue):
                    k, v = req.KV_cache[layer_idx] # 每个 req 的 KV 是 [1, H, S, D]
                    
                    # 对齐长度（右填充）
                    pad_len = max_kv_len - k.size(2)
                    if pad_len > 0:
                        pad_shape = (1, k.size(1), pad_len, k.size(3))
                        k = torch.cat([k, torch.zeros(pad_shape, device=device, dtype=k.dtype)], dim=2)
                        v = torch.cat([v, torch.zeros(pad_shape, device=device, dtype=v.dtype)], dim=2)
                    
                    keys_to_cat.append(k)
                    values_to_cat.append(v)
                
                # --- 正确的填充方式 ---
                # 将拼接好的 Batch Tensor 直接放入对象的列表中
                # 这样 batched_past.get_seq_length(layer_idx) 就能返回 max_kv_len
                batched_past.update(key_states=torch.cat(keys_to_cat, dim=0),
                                    value_states=torch.cat(values_to_cat, dim=0),
                                    layer_idx=layer_idx)

            # 3. 准备正确的 Attention Mask [Batch, max_kv_len + 1]
            # 注意：因为 input_ids 此时是 [Batch, 1]，
            # 传入模型的 mask 长度必须等于历史 KV 长度 + 1
            total_mask_len = max_kv_len + 1
            batch_masks = []
            for i, req in enumerate(decode_queue):
                # 有效位是 1，Padding 位是 0
                # 这里的有效位 = 该请求的历史长度 + 当前这 1 个 token
                valid_len = current_kv_lens[i] + 1
                m = torch.cat([
                    torch.ones(valid_len, device=device),
                    torch.zeros(total_mask_len - valid_len, device=device)
                ])
                batch_masks.append(m)
            attention_mask = torch.stack(batch_masks).long()

            # 4. Forward
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    use_cache=True, 
                    past_key_values=batched_past  # 传入 Batch 化的 KV Cache
                )
                
            new_past = outputs.past_key_values # 包含了更新后的 KV
            next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1)

            # 5. 更新状态并写回 KV Cache
            for i, req in enumerate(decode_queue):
                req.step(next_tokens[i])
                
                # 更新 input_ids
                req.input_ids = torch.cat([req.input_ids, next_tokens[i].view(1)], dim=0)

                # 重要：从 new_past 中切分出属于该请求的 KV（去掉 Padding）
                # new_past 每层是 [Batch, head, seq_len+1, dim]
                actual_new_len = current_kv_lens[i] + 1
                req_new_kv = []
                for past_layer in new_past:
                    k_layer, v_layer = past_layer[0], past_layer[1]  # k_layer, v_layer: (Batch, num_heads, seq_len+1, head_dim)
                    # 提取该请求对应的行，并只取有效长度的部分
                    k_i = k_layer[i:i+1, :, :actual_new_len, :].clone()
                    v_i = v_layer[i:i+1, :, :actual_new_len, :].clone()
                    req_new_kv.append((k_i, v_i))
                req.KV_cache = req_new_kv

            # 过滤完成的
            decode_queue = [req for req in decode_queue if not req.finished]

            # Step 5: 避免空转
        if not decode_queue and not prefill_queue:
            await asyncio.sleep(0.001)  # 1ms

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