# items/

What this folder is for: the generated item files, the fixed set every model is evaluated on.

What goes here:

- One JSON Lines file per task, generator, purpose and size, named <task>_<generator>_<purpose>_<size>.jsonl, written by make_items.py.
- Next to each file, its summary (items per cell, rejections and why, generator version).
- Never edit an item file by hand. If something is wrong, fix the adapter and regenerate.
- Train items (Arul's) and calibration items live here too, but in their own files with their own seed ranges.
