# Architecture: how CL methods are wired in `low-rank-cl`

Four layers, one string (`method`) threading through them. No "axes" composition — each method owns its own full ViT fork. Simpler dispatch, but more copy-paste per new method.

```
┌──────────────────────────────────────────────────────────────────────┐
│  LAYER 1 — CLI + JSON config                                         │
│  main.py:12-14                                                       │
│    args = parse_args()       # --config --device --debug --save_ckp  │
│    args.update(load_json(args.config))                               │
│                                                                      │
│  Everything substantive (method, dataset, rank, epochs, lrate,       │
│  method-specific knobs like EWC's gamma/lambda) lives in             │
│  configs/<method>/<dataset>.json. Only 4 real CLI flags.             │
└───────────────────┬──────────────────────────────────────────────────┘
                    │
                    ▼  utils/factory.py::get_model       (string dispatch)
┌──────────────────────────────────────────────────────────────────────┐
│  LAYER 2 — method selection                                          │
│    if method == 'baseline':  return Baseline(args)                   │
│    elif method == 'inflora': return InfLoRA(args)                    │
│    elif method == 'sdlora':  return SDLoRA(args)                     │
│    elif method == 'cllora':  return CLLoRA(args)                     │
│    elif method == 'ewclora': return EWCLoRA(args)                    │
│                                                                      │
│  Single switch. Register a new method by adding an elif here.        │
└───────────────────┬──────────────────────────────────────────────────┘
                    │
                    ▼  trainer.py::train                 (canonical CL loop)
┌──────────────────────────────────────────────────────────────────────┐
│  LAYER 3 — the task loop (same for every method)                     │
│    for task in range(data_manager.task_num):                         │
│        model.before_task(data_manager)     # grow classifier         │
│        model.incremental_train(data_manager)                         │
│        accy, accy_with_task, accy_task =                             │
│            model.incremental_test(data_manager)                      │
│        model.after_task()                  # method bookkeeping      │
│                                                                      │
│  Lives in methods/base.py::BaseLearner. Each method overrides a      │
│  subset of {_train, _train_function, freeze_network, after_task}.    │
└───────────────────┬──────────────────────────────────────────────────┘
                    │
                    ▼  each method's own ViT fork
┌──────────────────────────────────────────────────────────────────────┐
│  LAYER 4 — method-specific ViT surgery                               │
│                                                                      │
│  Each method forks the timm ViT into models/vit_<method>.py and      │
│  replaces `Attention` with its own `Attention_LoRA` variant:         │
│                                                                      │
│    baseline  → lora_{A,B}_{k,v} + lora_new_{A,B}_{k,v}, accumulate   │
│    inflora   → per-task lora_A_k[task], cur_matrix/DualGPM hooks     │
│    sdlora    → scaling/direction decomposed LoRA                     │
│    cllora    → its own variant                                       │
│    ewclora   → baseline + delta_w_{k,v}_new_grad hooks for Fisher    │
│                                                                      │
│  These forks are INDEPENDENT. Attention_LoRA.__init__,               │
│  .init_param, .accumulate_and_reset_lora, .forward all differ.       │
│  No shared base class.                                               │
└──────────────────────────────────────────────────────────────────────┘
```

## The four touchpoints per method

| layer | file | role |
|-------|------|------|
| 1 | `configs/<method>/<dataset>.json` | hyperparameters + method-specific knobs |
| 2 | `utils/factory.py` | one elif mapping string → class |
| 3 | `methods/<method>.py` | `BaseLearner` subclass: `_train_function`, `freeze_network`, `after_task` |
| 3 | `models/net_<method>.py` | `Net` wrapper: ViT encoder + `classifier_pool` |
| 4 | `models/vit_<method>.py` | forked timm ViT with a method-specific `Attention_LoRA` |

## Where everything lives (file index)

```
main.py                         Layer 1: parse CLI, load JSON, call trainer.start
trainer.py                      Layer 3: the canonical for-task loop

utils/factory.py                Layer 2: method string → class
utils/toolkit.py                logging, seed/device, print_trainable_params,
                                check_params_consistency (freeze/optim sanity check)

methods/base.py                 BaseLearner — optimizer/scheduler builder,
                                _test (task-agnostic + task-aware + task-id acc),
                                _evaluate, build_{train,test}_loader,
                                abstract _train_function
methods/baseline.py             LoRA + accumulate, no regularization
methods/inflora.py              DualGPM subspace projection over LoRA
methods/sdlora.py               Scale/direction LoRA decomposition
methods/cllora.py               (check source for specifics)
methods/ewclora.py              EWCLoRA class + FisherComputer
                                empirical Fisher over ΔW=B@A via grad hooks

models/net.py, net_<method>.py  ViT encoder + nn.ModuleList classifier_pool
                                (one nn.Linear per task). interface() used by _test.
models/vit.py, vit_<method>.py  Per-method fork of the full timm ViT
                                (~1000+ lines each). The Attention_LoRA class
                                is where methods actually differ.
models/modules/                 (shared modules, if any)

dataloaders/data_manager.py     DataManager: class ordering, per-task split,
                                get_dataset(indices, source, mode)
dataloaders/data.py             iCIFAR100, iIMAGENET_R, iIMAGENET_A,
                                iDomainNet, iCUB, iCIFAR10
dataloaders/splits/             DomainNet pickled splits

configs/<method>/<dataset>.json every hyperparameter + method-specific keys
```

## EWC-LoRA as a concrete walkthrough

End of each task: measure **which LoRA ΔW entries mattered** for that task (Fisher over ΔW=B@A, not over the raw B/A pair). Next task: penalize changes to those entries. Then fold the new LoRA into the accumulated LoRA and reset.

```
┌─── TASK t TRAINING ────────────────────────────────────────────┐
│ every iter:                                                    │
│   loss = CE(y, ŷ)                                              │
│   if count_updates > 0:                                        │
│     for each attention layer:                                  │
│         ΔW = lora_new_B @ lora_new_A                           │
│         loss += λ/2 · Σ ω · ΔW²                                │
└────────────────────────────────────────────────────────────────┘
              ↓ task t finished (after_task)
┌─── END-OF-TASK-t ──────────────────────────────────────────────┐
│ FisherComputer.compute():                                      │
│   hook delta_w_{k,v}_new onto .grad via register_hook          │
│   run backward over train_loader                               │
│   F = E[(∂L/∂ΔW)²]                                             │
│ ω ← γ·ω_prev + F                       (online EWC)            │
│ network.accumulate_and_reset_lora():                           │
│   lora_{A,B}_{k,v} += lora_new_{A,B}_{k,v}                     │
│   re-init lora_new_{A,B}_{k,v}                                 │
└────────────────────────────────────────────────────────────────┘
```

Note: Fisher is on the **product** ΔW=B@A (registered via `save_grad` hooks in `Attention_LoRA.forward`), not on A and B separately. This is the paper's distinctive move.

## Adding a new method — concrete touchpoints

Say you're adding `mymethod`:

```
1. configs/mymethod/cifar100.json
   └─ "method": "mymethod", plus your hyperparameters

2. utils/factory.py
   └─ elif method == 'mymethod':
          from methods.mymethod import MyMethod
          return MyMethod(args)

3. methods/mymethod.py  (~100-250 lines typically)
   class MyMethod(BaseLearner):
       def __init__(self, args): super().__init__(args); self.network = Net(args)
       def _train_function(...):  # per-epoch loop, your loss
       def freeze_network(...):   # which LoRA params to unfreeze this task
       def after_task(...):       # Fisher / subspace update / whatever
       def _train(...):           # often override to customize optimizer param-groups

4. models/net_mymethod.py  (~130 lines, usually light edit of net.py)
   └─ import Attention_LoRA from models.vit_mymethod
   └─ forward signature may add kwargs your method needs (e.g., get_cur_feat)

5. models/vit_mymethod.py  (~1100 lines — copy of vit.py with edits)
   └─ Edit Attention_LoRA: state, init_param, accumulate_and_reset_lora, forward
   └─ Propagate any new forward kwargs up through Block → VisionTransformer
```

## Difficulty assessment

**Shape of the work:** ~250–400 new lines of actual logic, plus ~1100 lines of `vit_<method>.py` that is 95% copy and 5% edits. Touching 5 files.

**Easier than it looks because:**
- The loop in `trainer.py` is fixed and every hook you need is already a named method on `BaseLearner`.
- The task-aware / task-agnostic / task-id evaluation is free.
- Adding method-specific hyperparameters is literally "put a key in the JSON".
- `check_params_consistency` catches freeze/optim mismatches — the most common footgun.
- `ewclora` is the cleanest reference; `inflora` shows how to do subspace ops.

**Harder than it looks because:**
- There is no shared `Attention_LoRA` base — each method reinvents it. If your method needs a novel LoRA parameterization, the vit fork is unavoidable.
- The forward-signature of `Net` / `VisionTransformer` / `Block` / `Attention_LoRA` changes per method (`use_new`, `register_hook`, `get_cur_feat`, per-task task index, …). You'll propagate kwargs through 4 levels.
- Accumulating vs. per-task-slot LoRA is a structural choice baked into the vit fork (baseline/ewclora accumulate into one pair; inflora stores `lora_A_k[task]`). Switching modes is not a flag flip.
- No tests. The only validation is "run a short debug experiment and watch the accuracy curve."

**Bottom line:** a method that *reuses* the baseline LoRA structure (new state on the side + a regularization term + an end-of-task update, like EWC-LoRA) is ~1 day of work, mostly in `methods/<name>.py`. A method that needs a **new** LoRA parameterization (different A/B shapes, per-task slots, new forward math) is ~2–3 days because of the vit fork.

## One-sentence summary

**JSON config → factory switch → `BaseLearner` subclass → forked ViT**, with `method` as the only selector and each method owning an independent `Attention_LoRA` implementation. Adding a new method is a structured copy-paste across five files, not a composition of orthogonal axes.
