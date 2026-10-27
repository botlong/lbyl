#!/bin/bash
# LBYL M0 full run — needs a GPU attached. Launch with:
#   setsid bash run_full.sh > m0_run.log 2>&1 </dev/null & disown
set -e
cd "$(dirname "$0")"
export PLAYWRIGHT_BROWSERS_PATH=/root/rivermind-data/pw-browsers
export HF_HOME=/root/rivermind-data/hf
M=/root/rivermind-data/models
echo "=== GPU check ==="; python3 -c "import torch;print('cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo "=== M0: 3B then 7B (anti-scaling) ==="
python3 run_m0.py --models "$M/Qwen2.5-VL-3B-Instruct" "$M/Qwen2.5-VL-7B-Instruct"
echo "ALL_M0_DONE"
