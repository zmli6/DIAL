#!/bin/bash
#SBATCH --job-name=dial-controlled
#SBATCH --output=logs/controlled_%A_%a.out
#SBATCH --error=logs/controlled_%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --array=0-7%4
# ════════════════════════════════════════════════════════════════════
# Tab `controlled` (Section 5.3 P2) — InfoPoor / InfoRich on HotpotQA.
# 2 variants × (1 base + 3 DIAL seeds) = 8 runs.
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
export DIAL_VLLM_ENDPOINT="${DIAL_VLLM_ENDPOINT:-http://localhost:$((9700 + SLURM_ARRAY_TASK_ID))/v1}"

VARIANTS=(infopoor inforich)
TASKS=(base dial_42 dial_123 dial_456)

IDX=${SLURM_ARRAY_TASK_ID}
VARIANT=${VARIANTS[$((IDX / 4))]}
TASK=${TASKS[$((IDX % 4))]}

CONFIG="configs/controlled/hotpotqa_${VARIANT}.yaml"

if [ "$TASK" = "base" ]; then
    # base_only + always_trigger as bounds (200 episodes each)
    python experiments/run_dial.py --config "$CONFIG" --env hotpotqa \
        --method bound:base_only --seed 42 --episodes 200 \
        --output-root "results/controlled/$BACKBONE"
    python experiments/run_dial.py --config "$CONFIG" --env hotpotqa \
        --method bound:always_trigger --seed 42 --episodes 200 \
        --output-root "results/controlled/$BACKBONE"
else
    SEED="${TASK#dial_}"
    python experiments/run_dial.py --config "$CONFIG" --env hotpotqa \
        --method dial --seed "$SEED" --episodes 200 \
        --output-root "results/controlled/$BACKBONE"
fi
