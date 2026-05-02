#!/bin/bash
#SBATCH --job-name=dial-capacity
#SBATCH --output=logs/capacity_%A_%a.out
#SBATCH --error=logs/capacity_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --array=0-8%3
# ════════════════════════════════════════════════════════════════════
# Tab `capacity` — Logistic / MLP / Hidden-state probe on HotpotQA × 3 seeds.
# 3 gates × 3 seeds = 9 runs.
# ════════════════════════════════════════════════════════════════════
set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/DIAL}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-dial}"
BACKBONE="${BACKBONE:-qwen3-4b}"

cd "$PROJECT_ROOT"
mkdir -p logs results
source "$CONDA_SH"
set +u; conda activate "$CONDA_ENV"; set -u
export DIAL_VLLM_ENDPOINT="${DIAL_VLLM_ENDPOINT:-http://localhost:$((9600 + SLURM_ARRAY_TASK_ID))/v1}"

GATES=(logistic mlp probe)
SEEDS=(42 123 456)

IDX=${SLURM_ARRAY_TASK_ID}
GATE=${GATES[$((IDX / 3))]}
SEED=${SEEDS[$((IDX % 3))]}

# Step 1: collect hidden states + features (only needed for probe)
if [ "$GATE" = "probe" ]; then
    python experiments/capacity_collect.py --env hotpotqa --seed "$SEED" --backbone "$BACKBONE" \
        --output-root "results/capacity/$BACKBONE/hidden_states"
    python experiments/capacity_train.py --env hotpotqa --seed "$SEED" \
        --hidden-states-dir "results/capacity/$BACKBONE/hidden_states" \
        --output-root "results/capacity/$BACKBONE/probes"
fi

# Step 2: evaluate the gate
python experiments/capacity_eval.py \
    --env hotpotqa \
    --gate "$GATE" \
    --seed "$SEED" \
    --backbone "$BACKBONE" \
    --output-root "results/capacity/$BACKBONE"
