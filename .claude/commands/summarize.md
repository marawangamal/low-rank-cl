---
description: Print a methods × datasets table of final top-1 accuracies from logs/
---

Run `python3 summarize.py` from the repo root and show the user the resulting table.

The script walks `logs/<method>/<dataset>/t*/seed0.log`, extracts the last value of each `(curve) top1 Acc: [...]` line, and prints a methods × datasets table.

Flags (pass through if the user specifies):
- `--logs <dir>` — log root (default `logs`)
- `--seed <name>` — seed file stem (default `seed0`)

If a method listed in `configs/` has no row in the output, it means no `seed0.log` exists under `logs/<method>/` — mention this to the user rather than silently omitting it.
