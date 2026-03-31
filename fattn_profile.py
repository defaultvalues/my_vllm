import torch
import time
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.profiler import profile, record_function, ProfilerActivity
import gc

#Path
# model_path = "/home/scm/mistral_models/7B-Instruct-v0.3"
model_path = "/home/scm/mistral_models/TransMLA/outputs/mistral_7b"

# 可配置窗口大小：设置为正整数启用滑动窗口；设为空字符串则不覆盖模型默认值
# window_size_env = os.getenv("FATTN_WINDOW_SIZE", "4096").strip()
# WINDOW_SIZE = int(window_size_env) if window_size_env else None

# --- 1. 加载模型 (关键修改点) ---
print("正在加载模型并开启 Flash Attention 2...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True
        # 【核心优化】开启 Flash Attention 2
        # 这会自动处理 Mistral 的 GQA 和 Sliding Window Attention
        # attn_implementation="flash_attention_2" 
    )
except ImportError:
    print("Error: 请先安装 flash-attn 库: pip install flash-attn --no-build-isolation")
    exit()

# if WINDOW_SIZE is not None:
#     if hasattr(model.config, "sliding_window"):
#         model.config.sliding_window = WINDOW_SIZE
#         print(f"已设置 sliding_window={WINDOW_SIZE}")
#     else:
#         print("警告: 当前模型配置不包含 sliding_window，无法设置窗口注意力。")
# else:
#     print("未覆盖 sliding_window，使用模型默认配置。")

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

# --- 2. 准备输入 ---
# 保持和 Baseline 一样的长度，以便控制变量对比
prompt = "[INST] 写一句关于人工智能的问候语。[/INST]" * 10 
tokenize_kwargs = {"return_tensors": "pt"}
# if WINDOW_SIZE is not None:
#     # 与 sliding window 对齐，避免输入长度远超窗口导致额外无效计算
#     tokenize_kwargs.update({"truncation": True, "max_length": WINDOW_SIZE})

inputs = tokenizer(prompt, **tokenize_kwargs).to(model.device)
input_token_len = inputs.input_ids.shape[1]
print(f"输入长度: {input_token_len} tokens")

# --- 3. Warmup (预热) ---
# FA2 的 kernel 第一次运行会有 autotune 开销，必须预热
print("正在预热 (Warmup)...")
model.generate(**inputs, max_new_tokens=5)

# --- 4. 彻底清理环境 ---
# 确保之前的显存碎片不影响本次测量
torch.cuda.empty_cache()
gc.collect()
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()

print("开始 Profiling (Optimized)...")

# --- 5. Profiling ---
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=False, # 关闭调用栈以减少开销，FA2 的 kernel 名称已经足够识别
) as prof:
    
    with torch.no_grad():
        with record_function("mistral_fa2_inference"): # 改个名字方便识别
            # 生成参数保持与 Baseline 一致
            model.generate(
                **inputs, 
                max_new_tokens=10, 
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True, # 保持一致
                temperature=0.7
            )

# --- 6. 结果输出 ---
# 打印统计表
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# 导出 Trace
output_file = "mistral_fa2_trace.json"
prof.export_chrome_trace(output_file)
print(f"Trace 已导出到 {output_file}，请在 Perfetto/Chrome 中打开并与 Baseline 对比。")

# --- 额外：打印显存峰值对比 ---
peak_memory = torch.cuda.max_memory_allocated() / 1024**3
print(f"Peak Memory (Optimized): {peak_memory:.2f} GB")