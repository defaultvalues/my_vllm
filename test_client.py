import asyncio
import httpx
import time

async def send_request(client, prompt, max_tokens):
    start_time = time.time()
    data = {
        "prompt": prompt,
        "max_new_tokens": max_tokens
    }
    try:
        # 设置较长的超时时间，因为 7B 模型推理需要时间
        response = await client.post("http://127.0.0.1:8001/generate", json=data, timeout=120.0)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n[Success] Prompt: {prompt}")
            print(f"Output: {result['output']}")
            print(f"Time taken: {end_time - start_time:.2f}s")
        else:
            print(f"[Error] {response.status_code}: {response.text}")
    except Exception as e:
        print(f"[Exception] {e}")

async def main():
    # 测试多个不同长度的并发请求
    prompts = [
        "[INST] Explain what is a black hole in one sentence. [/INST]",
        "[INST] How to make a cup of coffee? [/INST]",
        "[INST] Write a 3-word poem. [/INST]",
        "[INST] What is the capital of France? [/INST]",
    ]
    
    async with httpx.AsyncClient() as client:
        print(f"Sending {len(prompts)} concurrent requests to Mini-vLLM...")
        # 同时发送所有请求
        tasks = [send_request(client, p, 20) for p in prompts]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())