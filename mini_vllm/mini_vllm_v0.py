import asyncio
from contextlib import asynccontextmanager
import torch
import flashinfer

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb

from fastapi import FastAPI
from pydantic import BaseModel

import types
import uvicorn

# ======================
# 1. KVCache 管理
# ======================

class KVCache:
    def __init__(self, num_blocks, num_layers, num_heads, head_dim, block_size, device, dtype):
        """
        K_cache 和 V_cache 的维度设计为 [num_layers, num_blocks, num_heads, block_size, head_dim]，方便按 block 存储和访问 KV
        """

        self.num_blocks = num_blocks  # KV cache 的总块数，决定了最大并发请求数和最大序列长度, 一个block存的KV cache一定来自同一个请求
        self.block_size = block_size  # 每块的 token 数量，决定了每块能存储多少 KV

        self.kv_cache = torch.zeros(
            (num_layers, num_blocks, 2, block_size, num_heads, head_dim),  # 注意这里多了一个维度来区分 K 和 V
            dtype=dtype,
            device=device
        )

        self.free_blocks = list(range(num_blocks))  # 表示空闲块的索引列表

    def alloc_block(self):
        assert self.free_blocks, "KV cache OOM"
        idx = self.free_blocks.pop()
        return idx

    def free_block(self, idx):
        self.free_blocks.append(idx)

    def get_layer_cache(self, layer_idx):
        """
        返回某一层 KV cache
        shape: [num_blocks, 2, block_size, num_kv_heads, head_dim]
        """
        return self.kv_cache[layer_idx]
    

class InferenceMetadata:
    """
    统一管理KV cache相关的推理状态和元信息, 方便未来扩展更多功能(如分布式、FlashInfer 直接调用等)
    """
    def __init__(self):
        self.is_decode = False
        
        # paged KV cache
        self.paged_kv_indptr = None  # Tensor: [Batch + 1]，用来表示每个请求占用的block，在paged_kv_indices中的起止位置
        self.paged_kv_indices = None  # Tensor: [Total_Blocks_In_Batch], 表示所有已经被占用的内存块在全局 KV cache 中的索引
        self.paged_kv_last_page_len = None # Tensor: [Batch], 表示每个请求最后一个块实际使用的长度（因为可能不满 block_size），只有最后一个block可能不满，其他块都满
        self.qo_indptr = None
        
        self.workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda:1")
        
        # ===== wrappers =====
        self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer, "NHD"
        )

        self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer, "NHD"
        )


def flashinfer_attention_forward(self, hidden_states, position_embeddings, attention_mask=None, pask_key_values=None, **kwargs):
    global global_kv_cache, inference_metadata
    bsz, q_len, _ = hidden_states.size()
    device = hidden_states.device # 确保设备一致

    num_head = self.config.num_attention_heads
    head_dim = self.config.hidden_size // num_head
    num_kv_head = self.config.num_key_value_heads
    
    # 1. 计算 QKV
    query_states = self.q_proj(hidden_states).view(bsz, q_len, num_head, head_dim)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, num_kv_head, head_dim)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, num_kv_head, head_dim)

    # 2. RoPE
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(
        query_states.transpose(1, 2), key_states.transpose(1, 2), cos, sin
    )
    
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)

    # 这里的结果都是padding的结果，padding只是为了能够输入HF模型，但是真正要送入 FlashInfer 计算的 KV 是不包含 padding 的有效部分，所以我们需要根据 metadata 中的索引信息，取出每个请求对应的有效部分，拼接成一个连续的 Tensor，送入 FlashInfer。这个过程相当于在 forward 内部做了一次动态的 batch 切分和拼接，确保 FlashInfer 只处理有效的 KV 数据，同时也保持了每个请求 KV 在全局 KV cache 中的正确位置。

    # 取出结果中有效的部分，构造为符合 FlashInfer 输入的格式
    q_list, k_list, v_list = [], [], []
    qo_indptr = inference_metadata.qo_indptr  # [Batch + 1], 每个请求的 query 起止位置
    
    for i, req in enumerate(inference_metadata.requests):
        if inference_metadata.is_decode:
            valid_len = 1
        else:
            valid_len = len(req.input_ids)
        q_list.append(query_states[i, :valid_len])  # 注意：这里的 end 是相对于该请求的起始位置的偏移
        k_list.append(key_states[i, :valid_len])
        v_list.append(value_states[i, :valid_len])

    # 拼接成 [Total_Tokens, H, D] 的格式
    q_valid = torch.cat(q_list, dim=0)
    k_valid = torch.cat(k_list, dim=0)
    v_valid = torch.cat(v_list, dim=0)
    

    # 3. 将新的 KV 写入 Paged Cache
    flashinfer.append_paged_kv_cache(
        k_valid,
        v_valid,
        inference_metadata.batch_indices,     
        inference_metadata.positions,         
        global_kv_cache.kv_cache[self.layer_idx],  
        inference_metadata.paged_kv_indices,
        inference_metadata.paged_kv_indptr,
        inference_metadata.paged_kv_last_page_len,
        kv_layout="NHD"
    )

    assert q_valid.shape[0] == inference_metadata.append_indptr[-1]
    assert len(inference_metadata.batch_indices) == q_valid.shape[0]
    assert len(inference_metadata.positions) == q_valid.shape[0]

    # 4. 计算 Attention
    if not inference_metadata.is_decode:
        out_valid = inference_metadata.prefill_wrapper.run(
            q_valid, 
            global_kv_cache.kv_cache[self.layer_idx],
        )
    else:
        out_valid = inference_metadata.decode_wrapper.run(
            q_valid,
            global_kv_cache.kv_cache[self.layer_idx]
        )
    
    # === 关键修复：将结果填充回原始形状 ===
    out_valid_reshaped = out_valid.reshape(-1, self.config.hidden_size)
    output = torch.zeros((bsz, q_len, self.config.hidden_size), device=device, dtype=out_valid.dtype)
    
    offset = 0
    for i, req in enumerate(inference_metadata.requests):
        valid_len = 1 if inference_metadata.is_decode else len(req.input_ids)

        output[i, :valid_len] = out_valid_reshaped[offset: offset + valid_len]
        offset += valid_len
    
    # 恢复形状为 [B, S, Hidden_Size]
    return self.o_proj(output.reshape(bsz, q_len, -1)), None

def prepare_metadata(requests, kv_cache, metadata, is_decode):
    device = kv_cache.kv_cache.device
    metadata.is_decode = is_decode
    metadata.requests = requests  # 存储当前 batch 的请求列表，方便在 attention forward 中访问
    
    # 构建 FlashInfer 需要的各种索引 Tensor
    qo_indptr = [0]
    append_indptr = [0]

    batch_indices = []
    positions = []

    for i, req in enumerate(requests):
        if is_decode:
            cur_tokens = 1
            pos = [req.seq_len - 1]  # 计算在整个生成的序列中的绝对位置
        else:
            cur_tokens = len(req.input_ids)
            pos = list(range(cur_tokens))

        qo_indptr.append(qo_indptr[-1] + cur_tokens)
        append_indptr.append(append_indptr[-1] + cur_tokens)

        batch_indices.extend([i] * cur_tokens)
        positions.extend(pos)

    metadata.qo_indptr = torch.tensor(qo_indptr, dtype=torch.int32, device=device)
    metadata.append_indptr = torch.tensor(append_indptr, dtype=torch.int32, device=device)

    metadata.batch_indices = torch.tensor(batch_indices, dtype=torch.int32, device=device)
    metadata.positions = torch.tensor(positions, dtype=torch.int32, device=device)

    
    # Paged KV cache 相关的索引
    page_indices = []
    page_indptr = [0]
    last_page_len = []

    # pad 补齐
    for req in requests:
        page_indices.extend(req.block_table)
        page_indptr.append(page_indptr[-1] + len(req.block_table))
        
        # 关键：计算 append 之前的起始位置
        # 注意：在 forward 执行 append 之前，KV cache 里的长度
        # cur_seq_len = req.seq_len - (1 if is_decode else len(req.input_ids))
        # last_page_len.append((cur_seq_len - 1) % kv_cache.block_size + 1)  # 这个到底表示没填充的部分，还是这次要填充的部分？ 
        last_page_len.append((req.seq_len - 1) % kv_cache.block_size + 1)  # 这个表示这次要填充的部分，因为我们在 forward 之前就更新了 seq_len，表示即将生成的新 token 会占用 KV cache 中的位置，所以这里计算的就是这次要填充的部分长度
        # old_seq_len = req.seq_len  # 尚未包含本次要写入的token
        # last_page_len.append(0 if old_seq_len == 0 else (old_seq_len - 1) % kv_cache.block_size + 1)

    metadata.paged_kv_indices = torch.tensor(page_indices, dtype=torch.int32, device=device)
    metadata.paged_kv_indptr = torch.tensor(page_indptr, dtype=torch.int32, device=device)
    metadata.paged_kv_last_page_len = torch.tensor(last_page_len, dtype=torch.int32, device=device)


    if is_decode:
        metadata.decode_wrapper.plan(
            metadata.paged_kv_indptr, metadata.paged_kv_indices, metadata.paged_kv_last_page_len,
            model.config.num_attention_heads, model.config.num_key_value_heads,
            model.config.hidden_size // model.config.num_attention_heads,
            kv_cache.block_size, pos_encoding_mode="NONE", q_data_type=torch.float16
        )
    else:
        metadata.prefill_wrapper.plan(
            metadata.qo_indptr, metadata.paged_kv_indptr, metadata.paged_kv_indices,
            metadata.paged_kv_last_page_len, model.config.num_attention_heads,
            model.config.num_key_value_heads, model.config.hidden_size // model.config.num_attention_heads,
            kv_cache.block_size
        )

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

global_kv_cache = None  # 全局 KV cache 对象，供 scheduler 和 attention forward 访问
inference_metadata = InferenceMetadata()  # 管理 KV cache 相关的推理状态和元信息

# ======================
# 4. Dynamic Batching Worker
# ======================
BATCH_SIZE = 1
TIMEOUT = 0.01  # 10ms


async def scheduler():
    global waiting_queue, global_kv_cache, inference_metadata

    active_requests = []

    global_kv_cache = KVCache(
        num_blocks=32,  # 假设最多支持16个并发请求（每个请求最多使用一个 block，实际可以更灵活）
        num_layers=model.config.num_hidden_layers,
        num_heads=model.config.num_key_value_heads,
        head_dim=model.config.hidden_size // model.config.num_attention_heads,
        block_size=16,  # 每块最多存32个 token 的 KV，实际使用中可以根据请求长度动态调整
        device=device,
        dtype=model.dtype
    )

    while True:
        # ======================
        # Step 1: 收集新请求
        # ======================
        try:
            while True:
                req = await asyncio.wait_for(request_queue.get(), timeout=0.001)
                waiting_queue.append(req)
        except asyncio.TimeoutError:
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


            # 提前分配好 KV cache 的 block，并准备好 metadata 供 attention forward 使用
            for req in prefill_reqs:
                num_tokens = len(req.input_ids)
                
                num_blocks_needed = (num_tokens + global_kv_cache.block_size - 1) // global_kv_cache.block_size
                for _ in range(num_blocks_needed):
                    new_block_id = global_kv_cache.alloc_block()
                    req.block_table.append(new_block_id)

                req.seq_len = num_tokens  # 更新 seq_len，表示已经填充了这么多 KV，只是声明占用，实际写入会在 attention forward 中完成

            # 设置position_id供HF模型使用
            max_len = input_ids.shape[1]
            position_ids = []
            
            for req in prefill_reqs:
                # 假设左填充（Left Padding），则位置需要偏移
                # 假设右填充（Right Padding），位置直接从 0 到 len-1
                p_ids = torch.arange(len(req.input_ids), dtype=torch.long, device=device)
                # 如果需要补齐到 max_len (针对 HF model forward 的 batch 要求)
                pad_len = max_len - len(req.input_ids)
                # 右填充：
                p_ids = torch.cat([p_ids, torch.zeros(pad_len, dtype=torch.long, device=device)])
                position_ids.append(p_ids)
            
            position_ids = torch.stack(position_ids)
            
            # 设置metadata，管理 KV cache 的索引和状态，供 attention forward 使用
            prepare_metadata(prefill_reqs, global_kv_cache, inference_metadata, is_decode=False)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, position_ids=position_ids, use_cache=False)

            logits = outputs.logits[:, -1, :]
            next_tokens = torch.argmax(logits, dim=-1)


            # 更新request状态，准备下一步输入
            for i, req in enumerate(prefill_reqs):
                req.step(next_tokens[i])
                req.input_ids = torch.cat(
                    [req.input_ids, next_tokens[i].view(1)], dim=0
                )
                
                req.stage = "DECODE"

                if not req.finished:
                    new_active.append(req)
                else:
                    # 释放 KV block
                    for block_id in req.block_table:
                        global_kv_cache.free_block(block_id)
                    req.block_table = []  # 清空 block_table，表示不再占用 KV cache

        # ======================
        # Step 4: DECODE（单独 forward）
        # ======================
        if decode_reqs:
            input_ids = torch.stack([r.input_ids[-1:] for r in decode_reqs]).to(device)  # [B,1]

            # 1. 每个请求根据当前 seq_len 判断是否需要分配新块（如果当前块已满），并更新 seq_len（表示即将生成的新 token 会占用 KV cache 中的位置）
            for req in decode_reqs:
                # 
                if req.seq_len % global_kv_cache.block_size == 0:  # 每当生成的 token 数达到 block_size 的倍数时，分配一个新块
                    new_block_id = global_kv_cache.alloc_block()
                    req.block_table.append(new_block_id)
                
                # 更新逻辑长度（注意：这是在模型 forward 之前更新，因为我们要写进去）
                req.seq_len += 1 
            
            position_ids = torch.tensor(
                [[req.seq_len - 1] for req in decode_reqs], 
                dtype=torch.long, 
                device=device
            ) # 形状 [B, 1]
            
            prepare_metadata(decode_reqs, global_kv_cache, inference_metadata, is_decode=True)


            with torch.no_grad():
                # 进入模型，此时内部的 Attention 已经被 Patch 成了 FlashInfer.run()
                # 它会自动完成：写入新 KV -> 计算 PagedAttention -> 输出
                outputs = model(input_ids=input_ids, position_ids=position_ids, use_cache=False)
                    
            logits = outputs.logits[:, -1, :]
            next_tokens = torch.argmax(logits, dim=-1)

            # 更新request状态，准备下一步输入
            for i, req in enumerate(decode_reqs):
                req.step(next_tokens[i])
                req.input_ids = torch.cat(
                    [req.input_ids, next_tokens[i].view(1)], dim=0
                )

                if not req.finished:
                    new_active.append(req)
                else:
                    # 释放 KV block
                    for block_id in req.block_table:
                        global_kv_cache.free_block(block_id)
                    req.block_table = []  # 清空 block_table，表示不再占用 KV cache

        # ======================
        # Step 5: 更新 active
        # ======================
        active_requests = new_active

        if not active_requests:
            await asyncio.sleep(0.001)

# ======================
# 5. FastAPI 接口
# ======================
@asynccontextmanager
async def lifespan(_: FastAPI):
    scheduler_task = asyncio.create_task(scheduler())
    try:
        yield
    finally:
        scheduler_task.cancel()
        try:
            await scheduler_task
        except asyncio.CancelledError:
            pass


app = FastAPI(lifespan=lifespan)


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
if __name__ == "__main__":
    
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

    device = "cuda:1"

    model.to(device)

    # 替换HF模型中每层的 self_attn.forward 为 flashinfer_attention_forward
    for i, layer in enumerate(model.model.layers):  
        layer.self_attn.layer_idx = i  # 给每层 attention 绑定一个 layer_idx 属性，方便在 forward 中访问对应的 KV cache block
        layer.self_attn.forward = types.MethodType(flashinfer_attention_forward, layer.self_attn)

    uvicorn.run(app, host="0.0.0.0", port=8001)