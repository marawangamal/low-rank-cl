import os
import copy
import json
import argparse
import itertools
from pathlib import Path

from trainer import start
from utils.toolkit import make_logdir
from summarize import final_acc


SWEEP_SKIP_KEYS = {"seed"}


def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)
    args.update(param)

    sweep_keys = [
        k
        for k, v in args.items()
        if k not in SWEEP_SKIP_KEYS and isinstance(v, list) and len(v) > 1
    ]

    if not sweep_keys:
        args["logdir"] = make_logdir(args)
        run_seeds(args)
        return

    best = run_tune(args, sweep_keys)
    print("\nLaunching full run with best HPs: {}".format(best))
    full_args = copy.deepcopy(args)
    full_args.update(best)
    for k in ("tune_tag", "train_subsample_per_class", "max_tasks"):
        full_args.pop(k, None)
    full_args["logdir"] = make_logdir(full_args)
    run_seeds(full_args)


def run_seeds(args):
    device = copy.deepcopy(args["device"]).split(",")
    seed_list = copy.deepcopy(args["seed"])
    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        log_path = os.path.join(args["logdir"], "seed{}.log".format(seed))
        if os.path.exists(log_path):
            print("Skipping seed {}: {} exists".format(seed, log_path))
            continue
        start(args)


def run_tune(args, sweep_keys):
    base_tag = args.get("tune_tag") or "_".join(sorted(sweep_keys))
    values = [args[k] for k in sweep_keys]

    trials = []
    for combo in itertools.product(*values):
        combo_dict = dict(zip(sweep_keys, combo))
        trial_args = copy.deepcopy(args)
        trial_args.update(combo_dict)
        trial_args["train_subsample_per_class"] = args["val_n_samples"]
        trial_args["max_tasks"] = args["val_n_tasks"]
        trial_args["save_ckp"] = False
        tag_parts = "_".join("{}={}".format(k, v) for k, v in combo_dict.items())
        trial_args["tune_tag"] = "{}/{}".format(base_tag, tag_parts)
        trial_args["logdir"] = make_logdir(trial_args)
        run_seeds(trial_args)
        trials.append((combo_dict, trial_args["logdir"]))

    ranked = []
    for combo_dict, logdir in trials:
        accs = [
            a
            for a in (final_acc(p) for p in sorted(Path(logdir).glob("seed*.log")))
            if a is not None
        ]
        mean = sum(accs) / len(accs) if accs else None
        ranked.append((mean, combo_dict))
    ranked.sort(key=lambda x: (x[0] is None, -(x[0] or 0)))

    print("\nTuning results:")
    for acc, combo in ranked:
        print("  {}  ->  {}".format(combo, acc))

    return ranked[0][1]


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    parser = argparse.ArgumentParser(
        description="Reproduce of Multiple Incremental Learning Algorithms.",
        allow_abbrev=False,
        add_help=False,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="./configs/baseline.json",
        help="Json file of settings.",
    )
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--debug", action="store_true", help="Debug mode.")
    parser.add_argument(
        "--save_ckp", action="store_true", help="Whether to save checkpoints."
    )
    parser.add_argument(
        "--val-n-samples",
        type=int,
        default=50,
        help="Per-class training samples during tuning.",
    )
    parser.add_argument(
        "--val-n-tasks", type=int, default=3, help="Number of tasks during tuning."
    )
    parser.add_argument(
        "--tune-tag",
        type=str,
        default=None,
        help="Label under tuning_logs/<method>/<dataset>/. Default: joined sweep-key names.",
    )

    return parser


if __name__ == "__main__":
    main()
