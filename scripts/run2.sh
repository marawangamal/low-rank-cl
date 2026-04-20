#!/usr/bin/env bash
set -euo pipefail

DEVICE=0

METHODS=(ewcdlora)
DATASETS=(cifar100 imagenet-r imagenet-a)

cd /home/mila/m/marawan.gamal/scratch/low-rank-cl
source .venv/bin/activate

for method in "${METHODS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        config="configs/${method}/${dataset}.json"
        if [[ ! -f "$config" ]]; then
            echo "skip: $config (missing)"
            continue
        fi
        echo "=== $method / $dataset ==="
        python main.py --device "$DEVICE" --config "$config"
    done
done
