# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

PyTorch implementation of LoRA-based continual learning methods on ViT backbones, accompanying the paper "Revisiting Weight Regularization for Low-Rank Continual Learning" (arXiv:2602.17559). Implements Baseline, InfLoRA, SD-LoRA, CL-LoRA, and the paper's contribution EWC-LoRA.

## Environment

```bash
conda env create -f environment.yaml   # python 3.9, pytorch 2.0.1, cuda 11.8, timm 0.6.12
conda activate low-rank-cl
```

Datasets: CIFAR-100 auto-downloads. DomainNet/ImageNet-R/ImageNet-A must be placed at `./data` (symlink supported) or path overridden in the config JSON's `data_path`.

## Running experiments

```bash
python main.py --device <gpu_id> --config configs/<method>/<dataset>.json
# e.g.
python main.py --device 0 --config configs/ewclora/cifar100.json
python main.py --device 0 --config configs/ewclora/cifar100.json --save_ckp  # enable per-task checkpoints
python main.py --device 0 --config configs/ewclora/cifar100.json --debug     # logs go to logs/<method>/<dataset>/t<sessions>/debug/
```

Logs land in `logs/<method>/<dataset>/t<sessions>/seed<seed>.log`. Checkpoints are OFF by default (disk savings) — use `--save_ckp` to enable.

There is no test suite, linter, or build step. Validate changes by running a short debug experiment (shrink `epochs` and/or `sessions` in the chosen config) and watching the per-task accuracy curves in the log.

## Architecture

The entry point `main.py` loads a JSON config, merges it with argparse flags (CLI overrides config keys), and calls `trainer.start(args)`. `trainer.train` runs the canonical CL loop:

```
for task in tasks:
    model.before_task(data_manager)        # grows classifier, advances cur_task
    model.incremental_train(data_manager)  # builds train loader, calls _train / _train_function
    model.incremental_test(data_manager)   # reports task-agnostic + task-aware top-1, plus task-id acc
    model.after_task()                     # bookkeeping (e.g., Fisher update, LoRA accumulation)
```

Four cooperating layers — **when adding a new method, you typically touch all four**:

1. **`methods/<name>.py`** — subclass of `methods.base.BaseLearner`. Implements `_train_function`, `freeze_network`, and (if needed) overrides `_train`, `after_task`, and the classifier construction path. `BaseLearner` handles the optimizer/scheduler selection (`sgd`/`adam`/`adamw`, `constant`/`cosine`/`steplr`), the test loop, task-aware vs task-agnostic evaluation, and the abstract training contract.
2. **`models/net_<name>.py`** — the `Net` wrapper: ViT encoder + a `classifier_pool` (one `nn.Linear` per task). Exposes `forward`, `interface` (used by `_test`), and `update_fc`.
3. **`models/vit_<name>.py`** — a per-method fork of a timm ViT (each method tweaks `Attention` → `Attention_LoRA` differently; e.g. EWC-LoRA uses `lora_new_A/B_{k,v}` plus an accumulated `lora_A/B_{k,v}`, and tags new params with `_is_new_a`/`_is_new_b` so methods can filter them). These forks are intentionally independent; do not attempt to unify them without a specific reason.
4. **`configs/<method>/<dataset>.json`** — every hyperparameter lives here (`method`, `dataset`, `load` pretrained ViT, `rank`, `init_cls`/`increment`/`sessions`, optimizer/scheduler settings, and method-specific keys like EWC's `gamma`/`lambda`). Method selection is string-dispatched in `utils/factory.py::get_model` — register new methods there.

`dataloaders/data_manager.DataManager` owns class ordering/shuffling, the per-task class split, and `get_dataset(indices, source, mode)`; `dataloaders/data.py` has per-dataset classes (`iCIFAR100`, `iIMAGENET_R`, `iIMAGENET_A`, `iDomainNet`, `iCUB`, `iCIFAR10`). DomainNet uses the pickled splits in `dataloaders/splits/`.

### EWC-LoRA specifics

After each task, `methods/ewclora.py::FisherComputer` builds an empirical Fisher over the new LoRA delta `ΔW = B @ A` by registering hooks on `delta_w_{k,v}_new_grad` attributes inside `Attention_LoRA`. Importance accumulates as `omega = gamma * omega_prev + fisher_new`. The training loss adds `lambda/2 * sum(omega * ΔW^2)` once `count_updates > 0`. `network.accumulate_and_reset_lora()` folds the trained new LoRA into the accumulated LoRA and resets the new one for the next task.

## Conventions

- Parameter freezing is method-specific and done in `freeze_network`. Train-time unfrozen sets typically include `classifier_pool.<cur_task>` plus method-specific LoRA params.
- `BaseLearner.build_optimizer` accepts either a flat iterable or a list of param-group dicts (used by EWC-LoRA to give the classifier its own `fc_lrate`).
- `check_params_consistency(network, optimizer)` runs after every optimizer build — if it prints a warning, the freezing logic and the optimizer param list are out of sync.
- Config JSON keys are accessed directly as `args[...]`; missing optional keys use `args.get(...)`. Add new hyperparameters to the config first, not as CLI flags.

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/): `<type>: <short description>`. Do **not** use parenthesized scopes in this repo (use `fix: ...`, not `fix(test): ...`).

Types used in this repo:
- `feat:` — new feature/capability
- `fix:` — bug fix
- `refactor:` — restructure without behavior change
- `perf:` — performance improvement
- `test:` — add/update tests
- `docs:` — documentation only
- `chore:` — tooling, ignores, cleanup, deps
- `revert:` — revert a previous commit

Keep the subject ≤ 72 chars, imperative mood ("add X", not "added X"), lowercase after the colon. Add a body only when the *why* isn't obvious from the diff. No Claude co-author trailers.
