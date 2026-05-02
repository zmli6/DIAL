#!/bin/bash
#SBATCH --job-name=dial-signal
#SBATCH --output=logs/signal_%A_%a.out
#SBATCH --error=logs/signal_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --array=0-17%6
# ════════════════════════════════════════════════════════════════════
# Tab `signal-discovery` — ρ(token_entropy, U) on 6 envs × 3 seeds.
# Submit one backbone at a time:
#   BACKBONE=qwen3-4b sbatch scripts/slurm/run_signal_discovery.sbatch
# ════════════════════════════════════════════════════════════════════
set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/DIAL}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-dial}"
BACKBONE="${BACKBONE:-qwen3-4b}"
EPISODES="${EPISODES:-50}"

cd "$PROJECT_ROOT"
mkdir -p logs results
source "$CONDA_SH"
set +u; conda activate "$CONDA_ENV"; set -u
export DIAL_VLLM_ENDPOINT="${DIAL_VLLM_ENDPOINT:-http://localhost:$((9500 + SLURM_ARRAY_TASK_ID))/v1}"

ALL_ENVS=(hotpotqa apps webshop fever twexpress plancraft)
ALL_SEEDS=(42 123 456)

IDX=${SLURM_ARRAY_TASK_ID}
ENV=${ALL_ENVS[$((IDX / 3))]}
SEED=${ALL_SEEDS[$((IDX % 3))]}

python experiments/signal_discovery.py \
    --env "$ENV" \
    --seed "$SEED" \
    --episodes "$EPISODES" \
    --backbone "$BACKBONE" \
    --output-root "results/signal_discovery/$BACKBONE"
