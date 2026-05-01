import asyncio
import json
import time

import httpx
from transformers import AutoTokenizer


BASE_URL = "http://127.0.0.1:8001"
USE_STREAM = True  # True: 调 /generate_stream 看流式输出，False: 调 /generate
MAX_NEW_TOKENS = 96
TIMEOUT_S = 180.0
MODEL_PATH = "/home/scm/mistral_models/7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


PROMPTS = [
    "[INST] 请用两句话介绍一下你自己。[/INST]",
    "[INST] What is the capital of France? Answer in one sentence. [/INST]",
    "[INST] Explain black holes to a 10-year-old in 3 bullet points. [/INST]",
]


def check_output_quality(prompt: str, output: str, latency_s: float) -> None:
    stripped = output.strip()
    print("  Quality Check:")
    print(f"  - Non-empty: {'yes' if len(stripped) > 0 else 'no'}")
    print(f"  - Output chars: {len(output)}")
    print(f"  - Latency: {latency_s:.2f}s")

    if len(stripped) == 0:
        print("  - Warning: empty output, likely request or decode path has an issue.")
    elif len(stripped) < 10:
        print("  - Warning: output is very short, may have hit early stop/eos.")

    if "capital of France" in prompt and "Paris" not in output and "paris" not in output:
        print("  - Hint: expected keyword 'Paris' not found for this prompt.")


async def send_non_stream(client: httpx.AsyncClient, prompt: str, max_tokens: int) -> None:
    endpoint = f"{BASE_URL}/generate"
    payload = {"prompt": prompt, "max_new_tokens": max_tokens}

    start = time.time()
    response = await client.post(endpoint, json=payload, timeout=TIMEOUT_S)
    latency = time.time() - start

    print("\n" + "=" * 70)
    print("Mode: non-stream")
    print(f"Prompt: {prompt}")
    print(f"HTTP: {response.status_code}")

    if response.status_code != 200:
        print(f"Error body: {response.text}")
        return

    result = response.json()
    output = result.get("output", "")
    print("Output:")
    print(output)
    check_output_quality(prompt, output, latency)


async def send_stream(client: httpx.AsyncClient, prompt: str, max_tokens: int) -> None:
    endpoint = f"{BASE_URL}/generate_stream"
    payload = {"prompt": prompt, "max_new_tokens": max_tokens}

    print("\n" + "=" * 70)
    print("Mode: stream")
    print(f"Prompt: {prompt}")

    start = time.time()
    output_parts = []
    streamed_token_ids = []
    rendered_text = ""
    first_token_at = None
    token_count = 0

    async with client.stream("POST", endpoint, json=payload, timeout=TIMEOUT_S) as response:
        print(f"HTTP: {response.status_code}")
        if response.status_code != 200:
            err = await response.aread()
            print(f"Error body: {err.decode('utf-8', errors='ignore')}")
            return

        async for line in response.aiter_lines():
            if not line or not line.startswith("data:"):
                continue

            try:
                event = json.loads(line[len("data:") :].strip())
            except json.JSONDecodeError:
                continue

            if event.get("type") == "token":
                token_count += 1
                if first_token_at is None:
                    first_token_at = time.time()

                token_id = event.get("token_id")
                if isinstance(token_id, int):
                    streamed_token_ids.append(token_id)
                    # 用累计 token 重解码，再只打印新增片段；比逐 token decode 更接近真实空格表现。
                    decoded_now = tokenizer.decode(streamed_token_ids, skip_special_tokens=False)
                    if decoded_now.startswith(rendered_text):
                        delta = decoded_now[len(rendered_text):]
                    else:
                        delta = decoded_now
                    rendered_text = decoded_now
                    print(delta, end="", flush=True)
                else:
                    # 兜底：当 token_id 缺失时，退回服务端 text 字段
                    text = event.get("text", "")
                    output_parts.append(text)
                    print(text, end="", flush=True)
            elif event.get("type") == "done":
                # 以服务端最终 output 为准，避免逐 token decode 造成的拼接差异。
                final_output = event.get("output", "")
                if final_output:
                    output_parts = [final_output]
                break

    total_latency = time.time() - start
    ttft = (first_token_at - start) if first_token_at is not None else None
    output = "".join(output_parts)

    print("\n")
    print(f"TTFT: {ttft:.3f}s" if ttft is not None else "TTFT: N/A")
    print(f"Stream tokens received: {token_count}")
    check_output_quality(prompt, output, total_latency)


async def main() -> None:
    async with httpx.AsyncClient() as client:
        print(f"Target: {BASE_URL}")
        print(f"Use stream: {USE_STREAM}")
        print(f"Requests: {len(PROMPTS)} (sequential for easy inspection)")

        for idx, prompt in enumerate(PROMPTS, start=1):
            print(f"\n>>> Request {idx}/{len(PROMPTS)}")
            try:
                if USE_STREAM:
                    await send_stream(client, prompt, MAX_NEW_TOKENS)
                else:
                    await send_non_stream(client, prompt, MAX_NEW_TOKENS)
            except Exception as exc:
                print(f"[Exception] {exc}")


if __name__ == "__main__":
    asyncio.run(main())