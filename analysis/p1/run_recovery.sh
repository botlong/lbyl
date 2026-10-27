#!/bin/bash
set -euo pipefail
cd /root/lbyl-neucom/m0
R=results/neucom_p1/rerun_m1_mixed
mkdir -p "$R"
restore() { systemctl start hy-mt.service; systemctl is-active hy-mt.service > "$R/translation_unit_status.txt"; }
trap restore EXIT
systemctl stop hy-mt.service
export GPHI_MODEL=/root/lbyl-neucom/models/Qwen2.5-VL-3B-Instruct OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
date -Is > "$R/start.txt"
LBYL_DETECTOR=mixed LBYL_MODES=abstain,reground LBYL_LEVELS=0,1,2 LBYL_N=12 LBYL_TEMPLATES=4 LBYL_OUT="$R/m1_sweep_mixed.json" ../.venv/bin/python run_m1.py /root/lbyl-neucom/models/Qwen2.5-VL-3B-Instruct /root/lbyl-neucom/models/Qwen2.5-VL-7B-Instruct > "$R/m1_sweep_mixed.log" 2>&1
date -Is > "$R/done.txt"
