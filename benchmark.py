import asyncio
import aiohttp
import time
import statistics
import json
from transformers import AutoTokenizer

# ======================
# 配置参数
# ======================
API_URL = "http://localhost:8001/generate"
MODEL_PATH = "/home/scm/mistral_models/7B-Instruct-v0.3" # 仅用于本地计算 token 数
CONCURRENCY = 8       # 并发数（同时发送请求的数量）
TOTAL_REQUESTS = 32   # 总请求数
PROMPT = "Explain the importance of open source software in one paragraph."
MAX_NEW_TOKENS = 50

# 加载 tokenizer 用于精确计算 token 数量
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

class BenchmarkResult:
    def __init__(self):
        self.latencies = []
        self.total_tokens = 0
        self.start_time = 0
        self.end_time = 0

async def send_request(session, result_obj):
    payload = {
        "prompt": PROMPT,
        "max_new_tokens": MAX_NEW_TOKENS
    }
    
    start = time.perf_counter()
    try:
        async with session.post(API_URL, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                output_text = data["output"]
                latency = time.perf_counter() - start
                
                # 计算生成的 token 数量
                tokens = tokenizer.encode(output_text)
                num_tokens = len(tokens)
                
                result_obj.latencies.append(latency)
                result_obj.total_tokens += num_tokens
                return True
            else:
                print(f"Error: {response.status}")
                return False
    except Exception as e:
        print(f"Request failed: {e}")
        return False

async def runner(result_obj):
    # 使用信号量控制并发数
    semaphore = asyncio.Semaphore(CONCURRENCY)
    
    async with aiohttp.ClientSession() as session:
        tasks = []

        async def wrapped_request():
            async with semaphore:
                await send_request(session, result_obj)

        result_obj.start_time = time.perf_counter()
        for _ in range(TOTAL_REQUESTS):
            tasks.append(wrapped_request())
        
        await asyncio.gather(*tasks)
        result_obj.end_time = time.perf_counter()

def print_stats(res):
    total_time = res.end_time - res.start_time
    avg_latency = statistics.mean(res.latencies)
    p50 = statistics.median(res.latencies)
    p95 = statistics.quantiles(res.latencies, n=20)[18] # 95th percentile
    
    req_per_sec = TOTAL_REQUESTS / total_time
    tokens_per_sec = res.total_tokens / total_time

    print("\n" + "="*40)
    print("Benchmark Results")
    print("="*40)
    print(f"Total Requests:      {TOTAL_REQUESTS}")
    print(f"Concurrency:         {CONCURRENCY}")
    print(f"Total Time:          {total_time:.2f} s")
    print(f"Total Tokens:        {res.total_tokens}")
    print("-" * 40)
    print(f"Throughput (Req/s):  {req_per_sec:.2f} req/s")
    print(f"Throughput (Tok/s):  {tokens_per_sec:.2f} tokens/s")
    print("-" * 40)
    print(f"Avg Latency:         {avg_latency:.2f} s")
    print(f"P50 Latency:         {p50:.2f} s")
    print(f"P95 Latency:         {p95:.2f} s")
    print("="*40)

if __name__ == "__main__":
    results = BenchmarkResult()
    asyncio.run(runner(results))
    print_stats(results)