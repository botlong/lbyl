#!/bin/bash
# P1 rerun, stage B: frozen Qwen2.5-VL-3B features for every re-rendered set (GPU).
set -e
cd "$(dirname "$0")"
PY=../.venv/bin/python
export GPHI_MODEL=/root/lbyl-neucom/models/Qwen2.5-VL-3B-Instruct
for d in gphi_data_55 gphi_data gphi_data_realistic_v2 gphi_data_crossmodal gphi_data_class2 \
         gphi_data_susbench gphi_data_susbench_rawaria gphi_data_susbench_dest; do
  [ -f "$d/gphi_data.jsonl" ] || { echo "skip $d (not rendered)"; continue; }
  echo "== $d"; GPHI_DATA_DIR=$d $PY extract_feats.py 2>&1 | grep -a -E "probe|saved|EXTRACT|Error|error" || true
done
GPHI_DATA_DIR=gphi_data_realistic_v2 TXT_MODE=hidden FEATS_OUT=feats_hidden.pt $PY extract_feats.py 2>&1 | grep -a -E "saved|EXTRACT|rror" || true
echo EXTRACT_ALL_DONE
