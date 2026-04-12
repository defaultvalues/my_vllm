"""
这一版本希望实现真正的continous batching。
"""

import asyncio
import torch
import flashinfer

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb
from transformers.models.mistral.modeling_mistral import MistralRotaryEmbedding as rotary_emb

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
        
        self.workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
        
        # ===== wrappers =====
        self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer, "NHD"
        )

        self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer, "NHD"
        )


def flashinfer_attention_forward(self, hidden_states, position_embeddings, attention_mask=None, **kwargs):
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
    # q_list, k_list, v_list = [], [], []
    # qo_indptr = inference_metadata.qo_indptr  # [Batch + 1], 每个请求的 query 起止位置
    
    # for i, req in enumerate(inference_metadata.requests):
    #     if inference_metadata.is_decode:
    #         valid_len = 1
    #     else:
    #         valid_len = len(req.input_ids)
    #     q_list.append(query_states[i, :valid_len])  # 注意：这里的 end 是相对于该请求的起始位置的偏移
    #     k_list.append(key_states[i, :valid_len])
    #     v_list.append(value_states[i, :valid_len])

    # 拼接成 [Total_Tokens, H, D] 的格式
    q_valid = query_states.squeeze(0)  # [Total_Tokens, H, D]
    k_valid = key_states.squeeze(0)  # [Total_Tokens, H, D]
    v_valid = value_states.squeeze(0)  # [Total_Tokens, H, D]
    

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


    out_valid = inference_metadata.prefill_wrapper.run(
        q_valid, 
        global_kv_cache.kv_cache[self.layer_idx],
    )

    
    # === 关键修复：将结果填充回原始形状 ===
    out_valid_reshaped = out_valid.reshape(-1, self.config.hidden_size).unsqueeze(0)  # [Total_Tokens, Hidden_Size]
    # output = torch.zeros((bsz, q_len, self.config.hidden_size), device=device, dtype=out_valid.dtype)
    
    # offset = 0
    # for i, req in enumerate(inference_metadata.requests):
    #     valid_len = 1 if inference_metadata.is_decode else len(req.input_ids)

    #     output[i, :valid_len] = out_valid_reshaped[offset: offset + valid_len]
    #     offset += valid_len
    
    # 恢复形状为 [B, S, Hidden_Size]
    return self.o_proj(out_valid_reshaped), None

def prepare_metadata(requests, kv_cache, metadata: InferenceMetadata):
    device = kv_cache.kv_cache.device
    metadata.requests = requests  # 存储当前 batch 的请求列表，方便在 attention forward 中访问
    
    # 构建 FlashInfer 需要的各种索引 Tensor
    qo_indptr = [0]
    append_indptr = [0]

    batch_indices = []
    positions = []

    for i, req in enumerate(requests):
        if req.stage == "DECODE":
            cur_tokens = 1
            pos = [req.seq_len - 1]
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
        
        # 注意：在 forward 执行 append 之前，KV cache 里的长度
        # cur_seq_len = req.seq_len - (1 if is_decode else len(req.input_ids))
        last_page_len.append((req.seq_len - 1) % kv_cache.block_size + 1)  # 这里表示flash infer读取的时候最后一个块要读到哪里；由于我们是先append kv cache再计算 attention，所以这里的 seq_len 已经包含了当前这次要计算的 token 了，因此直接用 seq_len 就可以了

    metadata.paged_kv_indices = torch.tensor(page_indices, dtype=torch.int32, device=device)
    metadata.paged_kv_indptr = torch.tensor(page_indptr, dtype=torch.int32, device=device)
    metadata.paged_kv_last_page_len = torch.tensor(last_page_len, dtype=torch.int32, device=device)

    # TODO: qo_head 和 kv_head 分别指什么
    metadata.prefill_wrapper.plan(
        qo_indptr=metadata.qo_indptr, 
        paged_kv_indptr=metadata.paged_kv_indptr, 
        paged_kv_indices=metadata.paged_kv_indices,
        paged_kv_last_page_len=metadata.paged_kv_last_page_len, 
        num_qo_heads=model.config.num_attention_heads,
        num_kv_heads=model.config.num_key_value_heads, 
        head_dim_qk=model.config.hidden_size // model.config.num_attention_heads,
        page_size=kv_cache.block_size,
        causal=True
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

        self.cursor = 0  # 当前 prefill 进度

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
BATCH_SIZE = 4
TIMEOUT = 0.01  # 10ms
# CHUNCK_SIZE = 16  # 每次处理的 token 数量，过大可能增加延迟，过小可能降低吞吐量，实际使用中可以根据请求长度动态调整


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
        # TODO: Admission Control
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
        # Step 2: 准备模型的输入（调度哪些请求在这次进行推理）
        # 对于prefill阶段的任务，考虑它放多少进来prefill
        # ======================
    
        input_ids_list = []
        position_ids_list = []

        for req in active_requests:
            if req.stage == "PREFILL":
                input_ids_list.append(req.input_ids)
                position_ids_list.append(torch.arange(len(req.input_ids), dtype=torch.long, device=device))
                # TODO: 这里可以后续调整为chunk prefill
                num_new_tokens = len(req.input_ids) - req.cursor
                print(num_new_tokens)

                # 根据新的输入长度，计算需要分配多少 KV block，并更新请求的 block table
                num_blocks_needed = (num_new_tokens + global_kv_cache.block_size - 1) // global_kv_cache.block_size
                for _ in range(num_blocks_needed):
                    new_block_id = global_kv_cache.alloc_block()
                    req.block_table.append(new_block_id)

            else:
                input_ids_list.append(req.input_ids[-1:])  # decode阶段每次只送入最后一个 token
                position_ids_list.append(torch.tensor([len(req.input_ids) - 1], dtype=torch.long, device=device))
                num_new_tokens = 1  # decode阶段每次只生成一个 token

                # 如果现在的block table已经满了，说明之前分配的 KV block 已经用完了，需要再分配一个新的 block 来存储新的 KV
                if req.seq_len % global_kv_cache.block_size == 0:
                    new_block_id = global_kv_cache.alloc_block()
                    req.block_table.append(new_block_id)
            
            
            req.seq_len += num_new_tokens  # 更新 seq_len，表示已经填充了这么多 KV，只是声明占用，实际写入会在 attention forward 中完成
        
        input_ids = torch.cat(input_ids_list).to(device)
        position_ids = torch.cat(position_ids_list).to(device)
        
        # 设置metadata，管理 KV cache 的索引和状态，供 attention forward 使用
        prepare_metadata(active_requests, global_kv_cache, inference_metadata)

        with torch.no_grad():
            outputs = model(input_ids=input_ids.unsqueeze(0), 
                            position_ids=position_ids.unsqueeze(0),
                            use_cache=False)

        all_logits = outputs.logits.squeeze(0)


        # ======================
        # Step 3: 从模型输出中取出每个请求对应的 logits，并采样得到下一个 token
        # ======================
        new_active = []

        for i, req in enumerate(active_requests):
            last_token_idx = inference_metadata.qo_indptr[i + 1] - 1  # 每个请求最后一个 token 的位置
            logits = all_logits[last_token_idx]  # 取出对应位置的 logits
            next_token = torch.argmax(logits, dim=-1)  # 简单 greedy 采样，后续可以替换为更复杂的采样策略
            req.step(next_token)

            req.input_ids = torch.cat(
                [req.input_ids, next_token.view(1)], dim=0
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
        # Step 5: 更新 active
        # ======================
        active_requests = new_active

        # if not active_requests:
        #     await asyncio.sleep(0.001)

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

    device = "cuda:0"

    model.to(device)

    # 替换HF模型中每层的 self_attn.forward 为 flashinfer_attention_forward
    for i, layer in enumerate(model.model.layers):  
        layer.self_attn.layer_idx = i  # 给每层 attention 绑定一个 layer_idx 属性，方便在 forward 中访问对应的 KV cache block
        layer.self_attn.forward = types.MethodType(flashinfer_attention_forward, layer.self_attn)

    uvicorn.run(app, host="0.0.0.0", port=8001)