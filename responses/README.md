# responses/

What this folder is for: every raw model output, exactly as generated, so scoring can be redone at any time without rerunning models.

What goes here:

- One subfolder per model, each with one JSON Lines file per item file and prompt version, written by run_vllm.py.
- The run logs (timing, tokens per second, truncation counts).
- Never edit or delete outputs to clean them up; parsing and scoring handle bad outputs.
- These files get large: keep them out of git and back them up (Kaggle dataset versions or cloud storage).
