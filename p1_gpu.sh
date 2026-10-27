#!/bin/bash
# P1 rerun, stage D: everything that needs the frozen VLM at run time (live detector, real agents).
# Sequential so runs never share the GPU. Each step logs to results/neucom_p1/.
cd "$(dirname "$0")"
PY=../.venv/bin/python
R=results/neucom_p1; mkdir -p $R
M=/root/lbyl-neucom/models
export GPHI_MODEL=$M/Qwen2.5-VL-3B-Instruct PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
STEPS=${STEPS:-truthfree,e2e,e3,prompted,m1,rl,adaptive}
has() { [[ ",$STEPS," == *",$1,"* ]]; }
filt() { grep -a -v "Loading weights\|Warning\|warn(" ; }
has truthfree && $PY test_truthfree.py 2>&1 | filt > $R/truthfree.log
has e2e && E2E_SEEDS=0,1,2,3,4,5,6,7 E2E_LOG=$R/e2e_8seed.json $PY run_e2e_learned.py 2>&1 | filt > $R/e2e_8seed.log
has e3 && $PY e3_stabilize.py 2>&1 | filt > $R/e3.log
if has prompted; then
  GPHI_DATA_DIR=gphi_data_realistic_v2 SUBSET=new-instance MODE=neutral,dude $PY prompted_baseline.py 2>&1 | filt > $R/prompted_v2_newinstance.log
  GPHI_DATA_DIR=gphi_data_55 MODE=neutral LIMIT=400 $PY prompted_baseline.py 2>&1 | filt > $R/prompted_env55.log
  GPHI_DATA_DIR=gphi_data MODE=neutral $PY prompted_baseline.py 2>&1 | filt > $R/prompted_deploy.log
  GPHI_DATA_DIR=gphi_data_susbench MODE=dude $PY prompted_baseline.py 2>&1 | filt > $R/prompted_susbench.log
fi
if has m1; then   # the 144-task real-agent sweep (4 templates x 3 levels x 12), heuristic then mixed learned gate
  for D in heuristic mixed; do
    MODES=none,abstain,reground; [ $D = mixed ] && MODES=abstain,reground
    LBYL_DETECTOR=$D LBYL_MODES=$MODES LBYL_LEVELS=0,1,2 LBYL_N=12 LBYL_TEMPLATES=4 \
      LBYL_OUT=$R/m1_sweep_$D.json $PY run_m1.py $M/Qwen2.5-VL-3B-Instruct $M/Qwen2.5-VL-7B-Instruct 2>&1 | filt > $R/m1_sweep_$D.log
  done
fi
if has rl; then
  H=disguised_decline,fake_close_ad,inverted_consent_toggle
  TR=$($PY -c "import darkpatterns as d; print(','.join(sorted({t(0,0)[1]['template'] for t in d.TEMPLATES} - set('$H'.split(',')))))")
  Q="AGENT=qwen QWEN_PATH=$M/Qwen2.5-VL-3B-Instruct FEATS=live THREADS=8"
  env $Q DETECTOR=heuristic K=8 EPOCHS=6 LAM=0.05 HOLDOUT=$H OUT=$R/pi_theta_heur.pt $PY train_pi.py 2>&1 | filt > $R/rl_train_heur.log
  env $Q DETECTOR=heuristic HOLDOUT=$TR LIMIT=60 OUT=$R/pi_theta_heur.pt $PY eval_pi.py 2>&1 | filt > $R/rl_eval_indist_heur.log
  env $Q DETECTOR=heuristic HOLDOUT=$H OUT=$R/pi_theta_heur.pt $PY eval_pi.py 2>&1 | filt > $R/rl_eval_heldout_heur.log
  env $Q DETECTOR=heuristic HOLDOUT=$TR LIMIT=60 $PY eval_fixed_remove.py 2>&1 | filt > $R/rl_fixed_remove_heur.log
  env $Q DETECTOR=learned HOLDOUT=$H K=5 EPOCHS=4 LAM=0.05 LIMIT=100 CI_PI_SEEDS=5 RESULTS_DIR=$R $PY ci_pi.py 2>&1 | filt > $R/rl_ci_learned.log
fi
has adaptive && ADAPTIVE_OUT=$R/adaptive QWEN_PATH=$M/Qwen2.5-VL-3B-Instruct $PY run_adaptive_gpu.py run 2>&1 | filt > $R/adaptive.log
echo GPU_STAGE_DONE > $R/gpu_stage.done
