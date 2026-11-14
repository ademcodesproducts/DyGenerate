# run_vllm.py
#
# What it is for:
#   Runs one model over one or more item files with vLLM and saves every raw output.
#   Same script on Kaggle (T4 x2) and on the RTX 4060 (inside WSL2).
#
# What to include:
#
# 1. Arguments: model short name (looked up in config.py), item files, prompt version (native or canonical),
#    number of samples, and an optional item limit for smoke tests.
#
# 2. Model loading from the registry: dtype half on T4, tensor-parallel size 2 for 7B on Kaggle and 1 on the 4060,
#    max model length, memory utilisation around 0.9.
#
# 3. Prompt formatting: chat models get their own chat template; base models get plain text (prompts.py decides which).
#
# 4. Sampling: temperature, top-p, max new tokens and seed from config.py.
#
# 5. Resume support: skip items already present in the output file, and write results after every batch,
#    because Kaggle sessions end after 12 hours.
#
# 6. Output: responses/<model>/<item file>_<prompt version>.jsonl, one line per sample with:
#    item ID, model, prompt version, wrapper version, sample index, raw text, finish reason,
#    number of generated tokens, and the vLLM and model versions.
#
# 7. A run log: start and end time, tokens per second, how many outputs hit the token limit,
#    and the first two outputs printed in full so garbage output under fp16 is caught early.
