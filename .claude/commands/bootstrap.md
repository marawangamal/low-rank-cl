---
description: Set up the repo with uv and run a CIFAR-100 EWC-LoRA smoke experiment
---

Bootstrap this repository on the current machine (typically a fresh server), then verify the install by running the CIFAR-100 EWC-LoRA experiment.

## Install

If `pyproject.toml` and `uv.lock` already exist in the repo, just run:

```bash
uv sync
```

Otherwise, create them:

```bash
uv init --no-readme --python 3.9
uv add torch==2.0.1 torchvision==0.15.2
uv add timm==0.6.12 numpy==1.25.2 scikit-learn==1.2.0
uv add six ipdb scipy pyyaml tqdm
```

Then verify torch sees the GPU:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If `cuda.is_available()` is `False` on a CUDA machine, PyPI's `torch==2.0.1` wheel is cu117 — add a cu118 index to `pyproject.toml` and re-sync:

```toml
[[tool.uv.index]]
name = "pytorch-cu118"
url = "https://download.pytorch.org/whl/cu118"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu118" }
torchvision = { index = "pytorch-cu118" }
```

## Run

CIFAR-100 downloads automatically on first run. Kick off the EWC-LoRA experiment on GPU 0:

```bash
uv run python main.py --device 0 --config configs/ewclora/cifar100.json
```

Logs stream to stdout and to `logs/ewclora/cifar100/t10/seed0.log`. Per-task top-1 accuracy curves appear after each task completes.
