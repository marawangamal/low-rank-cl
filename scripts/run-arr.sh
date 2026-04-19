#!/usr/bin/env bash
#SBATCH --job-name=low-rank-cl
#SBATCH --partition=main
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/slurm/%x-%A_%a.out

set -euo pipefail

METHODS=(l2lora-l10.0)
DATASETS=(imagenet-a imagenet-r)

idx=$SLURM_ARRAY_TASK_ID
m_idx=$(( idx / ${#DATASETS[@]} ))
d_idx=$(( idx % ${#DATASETS[@]} ))
method="${METHODS[$m_idx]}"
dataset="${DATASETS[$d_idx]}"

cd /home/mila/m/marawan.gamal/scratch/low-rank-cl
source .venv/bin/activate

mkdir -p logs/slurm

config="configs/${method}/${dataset}.json"
echo "=== $method / $dataset (array=$idx) ==="
python main.py --device 0 --config "$config"
