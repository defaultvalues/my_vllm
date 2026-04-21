# mini-vLLM

一个从零实现的轻量级 LLM 推理引擎，复现了 vLLM 的核心机制，包括 Paged KV Cache、Continuous Batching、Chunked Prefill 以及抢占式调度。以 Mistral-7B 为后端模型，通过 FlashInfer 加速 Attention 计算，并对外暴露兼容 OpenAI 风格的 HTTP 接口。

## 整体架构

```
HTTP 请求
    │
    ▼
FastAPI Server (/generate)
    │  将请求放入 asyncio.Queue
    ▼
Scheduler (async loop)
    ├── Admission Control（准入控制）
    ├── 调度决策（Prefill / Decode / 抢占）
    └── 模型推理（HF model + FlashInfer Attention）
          │
          ▼
      KVCache（Paged Block Pool）
```

## 核心模块

### 1. KV Cache 管理（`KVCache`）

采用 **Paged KV Cache** 设计，将显存划分为固定大小的 Block（默认 `block_size=16`），每个 Block 存储若干 token 的 KV 向量。

- 全局 Cache 张量形状：`[num_layers, num_blocks, 2, block_size, num_kv_heads, head_dim]`
- 每个请求维护一张 `block_table`，记录该请求占用的 Block 索引列表
- 支持 **Block 预占（reserve）** 机制：在请求被调度前提前预留显存，避免推理中途 OOM

### 2. Attention 替换（`flashinfer_attention_forward`）

用 **monkey patch** 的方式替换 HuggingFace Mistral 模型每一层的 `self_attn.forward`，将标准的 `nn.MultiheadAttention` 替换为基于 FlashInfer 的 Paged Attention：

1. 计算 QKV 并施加 RoPE
2. 调用 `flashinfer.append_paged_kv_cache` 将新 KV 写入全局 Block Pool
3. 调用 `flashinfer.BatchPrefillWithPagedKVCacheWrapper.run` 完成 Attention 计算

### 3. 调度器（`scheduler`）

异步 `while True` 循环，每轮执行以下步骤：

#### Step 1 — 准入控制（Admission Control）
从 `waiting_queue` 中取出请求，只要剩余 Block 数量足以覆盖该请求的 Prefill 长度，就将其移入 `active_requests` 并预占对应 Block。

#### Step 2 — 调度决策
在 `token_budget`（默认 128 tokens/step）的约束下：

- **Decode 请求优先**：确保已经在生成阶段的请求每轮都能获得 1 个 token 的预算
- **Chunked Prefill**：Prefill 请求按 `CHUNK_SIZE`（默认 16）分块送入，避免单个长请求独占算力，同时也控制 KV Block 分配的粒度
- **抢占（Preemption）**：若 Decode 请求无法获得新 Block，调度器会选择"代价最小"的受害者（优先抢占 Prefill 请求，其次 Decode 请求）释放其 Block，保证高优先级请求继续执行

#### Step 3 — 推理与采样
将当前 batch 的所有 token 拼接为单个序列送入模型（`input_ids` 形状 `[1, total_tokens]`），通过 `InferenceMetadata` 中的 `qo_indptr` 还原每个请求的边界，按 greedy 策略采样下一个 token。

### 4. 推理元信息（`InferenceMetadata`）

封装 FlashInfer 所需的所有索引张量，在每轮 `scheduler` 调用前通过 `prepare_metadata` 重新构建：

| 字段 | 含义 |
|---|---|
| `paged_kv_indices` | 本 batch 所有请求占用的 Block 全局索引 |
| `paged_kv_indptr` | 每个请求在 `paged_kv_indices` 中的起止偏移 |
| `paged_kv_last_page_len` | 每个请求最后一个 Block 的实际有效长度 |
| `qo_indptr` | 每个请求的 Query token 在拼接序列中的起止偏移 |
| `batch_indices` / `positions` | 每个 token 对应的请求编号及位置编码索引 |

### 5. HTTP 接口

基于 FastAPI，在 `8001` 端口提供：

```
POST /generate
{
  "prompt": "...",
  "max_new_tokens": 20
}
```

请求放入异步队列后轮询完成，返回生成文本。


## 快速上手

```bash
# 安装依赖
pip install torch transformers fastapi uvicorn flashinfer

# 启动服务（需要修改 model_path 为本地路径）
python mini_vllm/mini_vllm.py

# 发送请求
curl -X POST http://localhost:8001/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, who are you?", "max_new_tokens": 50}'
```
