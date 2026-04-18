import argparse
import re
from pathlib import Path

CURVE_RE = re.compile(r"\(curve\) top1 Acc: \[([^\]]+)\]")


def final_acc(log_path: Path) -> float | None:
    last = None
    for line in log_path.read_text().splitlines():
        m = CURVE_RE.search(line)
        if m:
            last = m.group(1)
    if last is None:
        return None
    return float(last.rsplit(",", 1)[-1].strip())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--logs", default="logs")
    p.add_argument("--seed", default="seed0")
    args = p.parse_args()

    root = Path(args.logs)
    results: dict[str, dict[str, float]] = {}
    datasets: set[str] = set()

    for log in sorted(root.glob(f"*/*/t*/{args.seed}.log")):
        method, dataset, _tdir, _ = log.relative_to(root).parts
        acc = final_acc(log)
        if acc is None:
            continue
        results.setdefault(method, {})[dataset] = acc
        datasets.add(dataset)

    if not results:
        print(f"No results found under {root}/")
        return

    ds_cols = sorted(datasets)
    method_w = max(len("method"), *(len(m) for m in results))
    col_w = max(8, *(len(d) for d in ds_cols))

    header = f"{'method':<{method_w}}  " + "  ".join(f"{d:>{col_w}}" for d in ds_cols)
    print(header)
    print("-" * len(header))
    for method in sorted(results):
        row = f"{method:<{method_w}}  " + "  ".join(
            f"{results[method][d]:>{col_w}.2f}" if d in results[method] else f"{'-':>{col_w}}"
            for d in ds_cols
        )
        print(row)


if __name__ == "__main__":
    main()
