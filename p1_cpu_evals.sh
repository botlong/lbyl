#!/bin/bash
# P1 rerun, stage C: detector-level evaluations on the re-extracted features (CPU heads).
cd "$(dirname "$0")"
PY=../.venv/bin/python
R=results/neucom_p1; mkdir -p $R
export OMP_NUM_THREADS=8
run() { echo "### $1"; shift; "$@" 2>&1 | grep -a -v "Warning\|warn(" ; }
# RQ2 (environment set): LOTO + level split, channel ablations, 8-seed CI
GPHI_DATA_DIR=gphi_data_55 run base $PY train_gphi.py > $R/train_gphi_base.log
GPHI_DATA_DIR=gphi_data_55 GPHI_NO_NUMERIC=1 run nonum $PY train_gphi.py > $R/train_gphi_nonum.log
GPHI_DATA_DIR=gphi_data_55 GPHI_NO_NUMERIC=1 GPHI_ZERO_VIS=1 run zerovis $PY train_gphi.py > $R/train_gphi_zerovis.log
GPHI_DATA_DIR=gphi_data_55 GPHI_NO_NUMERIC=1 GPHI_ZERO_TXT=1 run zerotxt $PY train_gphi.py > $R/train_gphi_zerotxt.log
GPHI_DATA_DIR=gphi_data_55 GPHI_NO_NUMERIC=1 CI_RUNS=8 run ci8 $PY ci_gphi.py > $R/ci_gphi_8seed.log
# RQ3: channel-balanced v2 (group splits, 8 seeds), grid (pair-disjoint), hidden-state text, 364 trios
CM_DIR=gphi_data_realistic_v2 SPLIT=v2 run v2 $PY crossmodal_eval.py > $R/cm_v2_qwen.log
CM_DIR=gphi_data_realistic_v2 SPLIT=v2 FEATS_FILE=feats_hidden.pt run v2h $PY crossmodal_eval.py > $R/cm_v2_qwen_hidden.log
CM_DIR=gphi_data_crossmodal SPLIT=pair CM_HOLDOUT=subscribe,download_ad,allow_notif run grid $PY crossmodal_eval.py > $R/cm_grid_pair.log
CM_DIR=gphi_data_realistic_v2 SPLIT=sweep run sweep $PY crossmodal_eval.py > $R/cm_v2_sweep364.log
# Class-2 zero-shot, SusBench transfer
run class2 $PY class2_eval.py > $R/class2_eval.log
RESULTS=$R/susbench_transfer.txt run sus $PY susbench_eval.py > $R/susbench_eval.log
echo CPU_EVALS_DONE
