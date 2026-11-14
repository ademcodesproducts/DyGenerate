# RTX 4060 setup through WSL2 (Countdown, 3B models)

What this note is for: vLLM does not run on native Windows, so the 4060 runs it inside WSL2. This note records the working setup.

What to include:

- WSL2 with Ubuntu installed, and confirmation that the GPU is visible inside it (the Windows NVIDIA driver provides CUDA to WSL2; do not install a separate Linux driver).
- Python environment inside WSL2 and the pinned vllm version from requirements.txt.
- Where the project folder is accessed from (working inside the Linux file system is much faster than the mounted Windows drive).
- Model settings that fit 8 GB: a 3B model in fp16 (about 6 GB), tensor-parallel size 1, max model length around 4096, memory utilisation around 0.9.
- What to do if it does not fit: shorter max model length, fewer parallel sequences, or move the 3B models to Kaggle.
- Measured numbers from the smoke test: tokens per second and average output length per model.
