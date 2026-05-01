import asyncio
import time
import json

import aiohttp
import numpy as np
from transformers import AutoTokenizer

# 如果测试你的 Mini vLLM 改成 8001，测试 Baseline 改成 8002
URL = "http://localhost:8001/generate_stream"
CONCURRENCY = 16
TOTAL_REQUESTS = 100
REQUEST_TIMEOUT = 180
MAX_NEW_TOKENS_MIN = 32
MAX_NEW_TOKENS_MAX = 1024

# 使用本地 tokenizer 精确计算输出 token 数
MODEL_PATH = "/home/scm/mistral_models/7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# 准备一组不同长度和复杂度的测试 Prompt
PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "Write a Python script to scrape a website.",
    "What are the main differences between Python and C++?",
    "Give me a recipe for chocolate chip cookies.",
    "Summarize the plot of the movie Inception.",
    "How does a transformer neural network work?",
    "Write a poem about the sea.",
    "What is the history of the Roman Empire?",
    "Translate the following sentence to French: Hello, how are you?",
    "Provide a step-by-step guide to installing Ubuntu.",
]


def p(values, q):
    if not values:
        return 0.0
    return float(np.percentile(values, q))


async def fetch(session, request_id, prompt):
    max_new_tokens = int(np.random.randint(MAX_NEW_TOKENS_MIN, MAX_NEW_TOKENS_MAX + 1))
    payload = {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
    }
    start_time = time.time()

    record = {
        "request_id": request_id,
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "ok": False,
        "status": None,
        "error": None,
        "latency_s": None,
        "output_tokens": 0,
        "token_rate_tps": 0.0,
    }

    try:
        async with session.post(URL, json=payload, timeout=REQUEST_TIMEOUT) as response:
            latency = time.time() - start_time
            record["status"] = response.status
            record["latency_s"] = latency
            record["ttft_s"] = None
            record["tpot_s"] = None
            record["stream_tokens"] = 0

            if response.status != 200:
                body = await response.text()
                record["error"] = f"HTTP {response.status}: {body[:200]}"
                return record

            first_token_time = None
            token_timestamps = []
            output_text = ""

            while True:
                raw_line = await response.content.readline()
                if not raw_line:
                    break

                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line or not line.startswith("data:"):
                    continue

                data_str = line[len("data:") :].strip()
                if not data_str:
                    continue

                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type")
                if event_type == "token":
                    t_now = time.time()
                    if first_token_time is None:
                        first_token_time = t_now
                    token_timestamps.append(t_now)
                elif event_type == "done":
                    output_text = event.get("output", "")
                    break

            latency = time.time() - start_time
            record["latency_s"] = latency
            output_tokens = len(tokenizer(output_text).input_ids)

            record["ok"] = True
            record["output_tokens"] = int(output_tokens)
            record["token_rate_tps"] = float(output_tokens / latency) if latency > 0 else 0.0

            if first_token_time is not None:
                record["ttft_s"] = float(first_token_time - start_time)

            stream_tokens = len(token_timestamps)
            record["stream_tokens"] = int(stream_tokens)
            if stream_tokens >= 2:
                record["tpot_s"] = float((token_timestamps[-1] - token_timestamps[0]) / (stream_tokens - 1))

            return record
    except Exception as exc:
        record["latency_s"] = time.time() - start_time
        record["ttft_s"] = None
        record["tpot_s"] = None
        record["stream_tokens"] = 0
        record["error"] = str(exc)
        return record


async def worker(session, queue, results):
    while True:
        try:
            request_id, prompt = queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        record = await fetch(session, request_id, prompt)
        results.append(record)
        queue.task_done()


def print_request_level_stats(results, total_time_s):
    total = len(results)
    success_records = [r for r in results if r["ok"]]
    fail_records = [r for r in results if not r["ok"]]
    success = len(success_records)
    fail = len(fail_records)

    latencies = [r["latency_s"] for r in success_records if r["latency_s"] is not None]

    print("\n" + "=" * 46)
    print("Request-Level Metrics")
    print("=" * 46)
    print(f"Total Requests:        {total}")
    print(f"Success / Failed:      {success} / {fail}")
    print(f"Success Rate:          {100.0 * success / total if total else 0.0:.2f}%")
    print(f"Wall Time:             {total_time_s:.2f} s")
    print(f"QPS (all requests):    {total / total_time_s if total_time_s > 0 else 0.0:.2f} req/s")
    print(f"QPS (success only):    {success / total_time_s if total_time_s > 0 else 0.0:.2f} req/s")
    print("-" * 46)
    print(f"Avg Latency:           {float(np.mean(latencies)) if latencies else 0.0:.2f} s")
    print(f"P50 / P90 / P99:       {p(latencies, 50):.2f} / {p(latencies, 90):.2f} / {p(latencies, 99):.2f} s")

    if fail_records:
        print("-" * 46)
        print("Sample Errors (up to 5):")
        for rec in fail_records[:5]:
            print(f"  - req#{rec['request_id']}: {rec['error']}")


def print_token_level_stats(results, total_time_s):
    success_records = [r for r in results if r["ok"]]

    output_tokens = [r["output_tokens"] for r in success_records]
    token_rates = [r["token_rate_tps"] for r in success_records if r["token_rate_tps"] > 0]
    lat_per_100 = [
        (r["latency_s"] / r["output_tokens"] * 100.0)
        for r in success_records
        if r["output_tokens"] > 0 and r["latency_s"] is not None
    ]

    total_output_tokens = int(sum(output_tokens))
    throughput_tps = float(total_output_tokens / total_time_s) if total_time_s > 0 else 0.0

    print("\n" + "=" * 46)
    print("Token-Level Metrics")
    print("=" * 46)
    print(f"Total Output Tokens:   {total_output_tokens}")
    print(f"Global Throughput:     {throughput_tps:.2f} tokens/s")
    print("-" * 46)
    print(f"Output Tokens P50/P90/P99: {p(output_tokens, 50):.1f} / {p(output_tokens, 90):.1f} / {p(output_tokens, 99):.1f}")
    print(f"Per-Req Token Rate P50/P90/P99: {p(token_rates, 50):.2f} / {p(token_rates, 90):.2f} / {p(token_rates, 99):.2f} tokens/s")
    print(f"Latency per 100 tokens P50/P90/P99: {p(lat_per_100, 50):.2f} / {p(lat_per_100, 90):.2f} / {p(lat_per_100, 99):.2f} s")


def print_stream_level_stats(results):
    success_records = [r for r in results if r["ok"]]
    ttft_values = [r["ttft_s"] for r in success_records if r.get("ttft_s") is not None]
    tpot_values = [r["tpot_s"] for r in success_records if r.get("tpot_s") is not None]
    stream_tokens = [r.get("stream_tokens", 0) for r in success_records]

    print("\n" + "=" * 46)
    print("Streaming Metrics")
    print("=" * 46)
    print(f"TTFT P50/P90/P99:      {p(ttft_values, 50):.3f} / {p(ttft_values, 90):.3f} / {p(ttft_values, 99):.3f} s")
    print(f"TPOT P50/P90/P99:      {p(tpot_values, 50):.4f} / {p(tpot_values, 90):.4f} / {p(tpot_values, 99):.4f} s/token")
    print(f"Stream Tokens P50/P90/P99: {p(stream_tokens, 50):.1f} / {p(stream_tokens, 90):.1f} / {p(stream_tokens, 99):.1f}")


async def main():
    print(f"Starting benchmark against {URL}")
    print(f"Concurrency={CONCURRENCY}, TotalRequests={TOTAL_REQUESTS}")

    queue = asyncio.Queue()
    for i in range(TOTAL_REQUESTS):
        queue.put_nowait((i, PROMPTS[i % len(PROMPTS)]))

    results = []
    start = time.time()

    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.create_task(worker(session, queue, results))
            for _ in range(CONCURRENCY)
        ]
        await asyncio.gather(*tasks)

    total_time_s = time.time() - start

    results.sort(key=lambda x: x["request_id"])
    print_request_level_stats(results, total_time_s)
    print_token_level_stats(results, total_time_s)
    print_stream_level_stats(results)
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())