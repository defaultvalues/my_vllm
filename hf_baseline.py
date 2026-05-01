"""
HF Baseline 实现，使用 Hugging Face 的 AutoModelForCausalLM 加载 Mistral 模型，并通过 FastAPI 提供一个简单的文本生成接口。
这里实现了基础的 Static Batching（静态批处理）机制，利用 model.generate 来体现不加优化的原始 HF 的性能表现。
"""
import asyncio
import json
import torch
import flashinfer  # 保留你原有的 import 结构

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import import_utils as hf_import_utils
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

import uvicorn

# ======================
# 2. Request 定义
# ======================
class Request:
    def __init__(self, prompt, max_new_tokens=20):
        self.prompt = prompt
        # 存储为 1D token 序列
        self.input_ids = tokenizer(prompt, return_tensors="pt").input_ids.squeeze(0).to(device)
        self.generated =[]
        self.max_new_tokens = max_new_tokens
        self.finished = False
        self.past_key_values = None  
        self.stage = "PREFILL"  
        self.block_table =[]  
        self.seq_len = 0  
        self.cursor = 0  
        self.stream_queue = asyncio.Queue()  # 流式事件队列

    def step(self, next_token):
        self.generated.append(next_token.item())
        if len(self.generated) >= self.max_new_tokens or next_token.item() == tokenizer.eos_token_id:
            self.finished = True

    def get_output(self):
        # 增加 skip_special_tokens=True 确保输出干净
        return tokenizer.decode(self.generated, skip_special_tokens=True)


# ======================
# 3. 全局请求队列
# ======================
request_queue = asyncio.Queue()
waiting_queue =[]

# ======================
# 4. Static/Dynamic Batching Worker
# ======================
BATCH_SIZE = 16
TIMEOUT = 0.01  # 10ms

async def scheduler_hf():
    global waiting_queue

    active_requests =[]

    while True:
        # ======================
        # Step 1: 收集请求
        # ======================
        try:
            while True:
                req = await asyncio.wait_for(request_queue.get(), timeout=0.001)
                waiting_queue.append(req)
        except asyncio.TimeoutError:
            pass

        # 尝试塞满 BATCH_SIZE
        while waiting_queue and len(active_requests) < BATCH_SIZE:
            active_requests.append(waiting_queue.pop(0))

        if not active_requests:
            await asyncio.sleep(0.001)
            continue

        # ======================
        # Step 2: 使用 generate 方法生成结果
        # ======================
        # 关键：HuggingFace 的批处理生成 (Batch Generation) 必须使用左侧 Padding
        tokenizer.padding_side = "left"

        prompts =[req.prompt for req in active_requests]
        # 在静态 Batch 中，HF baseline 的生成长度取决于当前批次里设定的最大值
        max_new_tokens = max(req.max_new_tokens for req in active_requests)

        # 进行 padding 并转换为 Tensor
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

        def run_generate():
            with torch.no_grad():
                return model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False  # 保持贪婪解码与你的 mini-vLLM 一致
                )

        # 挂载到独立线程执行，防止 generate 长时间的计算阻塞主事件循环（导致无法并发接收请求）
        outputs = await asyncio.to_thread(run_generate)

        # ======================
        # Step 3: 将结果回填并标记完成
        # ======================
        input_len = inputs.input_ids.shape[1]
        for i, req in enumerate(active_requests):
            # 截取新生成的部分（丢弃前面的 prompt tokens）
            new_tokens = outputs[i][input_len:]
            
            # 移除 HF 补齐带来的 pad token
            valid_tokens =[tok for tok in new_tokens.tolist() if tok != tokenizer.pad_token_id]

            for token_id in valid_tokens:
                req.stream_queue.put_nowait(
                    {
                        "type": "token",
                        "token_id": int(token_id),
                        "text": tokenizer.decode([int(token_id)], skip_special_tokens=False),
                    }
                )
            
            req.generated = valid_tokens
            req.finished = True  # Baseline 中，一个 batch 是共同结束的
            req.stream_queue.put_nowait(
                {
                    "type": "done",
                    "output": req.get_output(),
                }
            )

        # ======================
        # Step 4: 清理 finished
        # ======================
        active_requests =[req for req in active_requests if not req.finished]


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

    # 简单等待完成
    while not r.finished:
        await asyncio.sleep(0.01)

    return {"output": r.get_output()}


@app.post("/generate_stream")
async def generate_stream(req: GenerateRequest):
    r = Request(req.prompt, req.max_new_tokens)
    await request_queue.put(r)

    async def event_generator():
        while True:
            event = await r.stream_queue.get()
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            if event.get("type") == "done":
                break

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)


# ======================
# 6. 启动 worker
# ======================
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(scheduler_hf())


if __name__ == "__main__":
    
    # ======================
    # 1. 加载模型
    # ======================
    model_path = "/home/scm/mistral_models/7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 强制开启左侧 Padding
    tokenizer.padding_side = "left"

    # transformers 5.x 某些版本在检测 flash_attn 时会访问缺失映射，做一次兼容兜底。
    hf_import_utils.PACKAGE_DISTRIBUTION_MAPPING.setdefault("flash_attn", ["flash_attn"])

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=None  
    )
    model.eval()
    model.to(device)

    # 将端口改为 8002，避免和你的优化版 (8001) 端口冲突
    uvicorn.run(app, host="0.0.0.0", port=8001)