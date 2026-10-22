#!/bin/bash
# Task 1 (B⑥) — DUDE-lite two-regime baseline vs g_phi.
# Regime A = contradiction class (realistic_cm, g_phi's turf); Regime B = salient/consistent
# (susbench visual/linguistic, a pure-visual detector's turf). Claim: complementary
# (g_phi wins A, DUDE-lite wins B). DUDE-lite = screenshot-only zero-shot proxy (no DOM).
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
cd /root/autodl-tmp/lbyl/m0
export GPHI_MODEL=/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct
export OMP_NUM_THREADS=8
LOG=/root/autodl-tmp/lbyl/m0/dude_lite.log
rm -f /root/autodl-tmp/lbyl/m0/dude_lite.done
{
echo "############ Regime A: CONTRADICTION (realistic_cm) ############"
echo "==== DUDE-lite (screenshot-only) + generic visual, balanced 400 ===="
MODE=dude,visual GPHI_DATA_DIR=gphi_data_realistic_cm LIMIT=400 SEED=0 python prompted_baseline.py
echo "==== g_phi learned head (crossmodal_eval: RANDOM + HOLDOUT) ===="
CM_DIR=gphi_data_realistic_cm python crossmodal_eval.py
echo ""
echo "############ Regime B: SALIENT/CONSISTENT (susbench) ############"
echo "==== DUDE-lite (screenshot-only) + generic visual, all 169 ===="
MODE=dude,visual GPHI_DATA_DIR=gphi_data_susbench python prompted_baseline.py
echo "==== g_phi zero-shot on susbench per dp_class (susbench_eval) ===="
python susbench_eval.py
echo "ALL_DUDE_LITE_DONE"
} > "$LOG" 2>&1
touch /root/autodl-tmp/lbyl/m0/dude_lite.done
