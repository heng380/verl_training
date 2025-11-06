#!/bin/bash

# Load configuration from the same directory as this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 -m verl.trainer.main_ppo \
  --config-path="$SCRIPT_DIR" \
  --config-name=run_qwen2.5-math-0.5b