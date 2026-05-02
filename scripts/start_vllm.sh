#!/bin/bash
# Launch a vLLM OpenAI-compatible server. Required whenever DIAL is run
# in real (non-stub) mode, since the LLM proposer talks to it. The server
# URL is read from `$DIAL_VLLM_ENDPOINT` by `dial.inference.proposer`;
# default port 9300.
#
# Usage:
#   bash scripts/start_vllm.sh                    # default Qwen3-4B
#   MODEL=microsoft/Phi-3.5-mini-instruct \
#     PORT=9301 bash scripts/start_vllm.sh
set -eo pipefail

MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
PORT="${PORT:-9300}"
DTYPE="${DTYPE:-bfloat16}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

echo "Starting vLLM server: model=$MODEL port=$PORT dtype=$DTYPE"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --dtype "$DTYPE" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --trust-remote-code
