#!/bin/bash
#SBATCH --job-name=dial-main
#SBATCH --output=logs/dial_main_%A_%a.out
#SBATCH --error=logs/dial_main_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --array=0-17%6
# ════════════════════════════════════════════════════════════════════
# DIAL main results (paper Fig `pareto`, Tab `full-results`).
# 6 envs × 3 seeds = 18 runs per backbone.
#
# Submit one backbone at a time:
#   BACKBONE=qwen3-4b   sbatch scripts/slurm/run_dial_main.sbatch
#   BACKBONE=phi-3.5    sbatch scripts/slurm/run_dial_main.sbatch
#   BACKBONE=llama-3.1  sbatch scripts/slurm/run_dial_main.sbatch
# ════════════════════════════════════════════════════════════════════
set -eo pipefail

# ── User-configurable env vars (override at submit time) ──
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/DIAL}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-dial}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
WEBSHOP_ROOT="${WEBSHOP_ROOT:-}"
BACKBONE="${BACKBONE:-qwen3-4b}"
EPISODES="${EPISODES:-100}"

cd "$PROJECT_ROOT"
mkdir -p logs results

source "$CONDA_SH"
set +u; conda activate "$CONDA_ENV"; set -u
export HF_HOME WEBSHOP_ROOT
export PYTHONPATH="${WEBSHOP_ROOT}:${PYTHONPATH:-}"
export DIAL_VLLM_ENDPOINT="${DIAL_VLLM_ENDPOINT:-http://localhost:$((9300 + SLURM_ARRAY_TASK_ID))/v1}"

ALL_ENVS=(hotpotqa apps webshop fever twexpress plancraft)
ALL_SEEDS=(42 123 456)

IDX=${SLURM_ARRAY_TASK_ID}
ENV_IDX=$((IDX / 3))
SEED_IDX=$((IDX % 3))
ENV=${ALL_ENVS[$ENV_IDX]}
SEED=${ALL_SEEDS[$SEED_IDX]}

echo "═══ DIAL run | env=$ENV seed=$SEED backbone=$BACKBONE episodes=$EPISODES ═══"

python experiments/run_dial.py \
    --config configs/methods/dial.yaml \
    --env "$ENV" \
    --method dial \
    --seed "$SEED" \
    --episodes "$EPISODES" \
    --output-root "results/main/$BACKBONE"
