#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Allow overriding number of processes per node via environment variable; default to 8
NPROC_PER_NODE="${SFT_NPROC:-8}"

torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" \
    -m verl.trainer.fsdp_sft_trainer \
    --config-path "${SCRIPT_DIR}" \
    --config-name run_qwen2.5-math-0.5b-sft \
    "$@"

