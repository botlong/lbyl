#!/bin/bash
# Second-round review runs: GPU RL evaluations (B-1/B-6) and CPU head scores (B-2/B-4/B-5) in parallel.
cd "$(dirname "$0")/../.."
PY=../.venv/bin/python
R=results/neucom_p2; mkdir -p $R/scores
M=/root/lbyl-neucom/models
export GPHI_MODEL=$M/Qwen2.5-VL-3B-Instruct PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
filt() { grep -a -v "Loading weights\|Warning\|warn(" ; }

H=disguised_decline,fake_close_ad,inverted_consent_toggle
TR=$($PY -c "import darkpatterns as d; print(','.join(sorted({t(0,0)[1]['template'] for t in d.TEMPLATES} - set('$H'.split(',')))))")
$PY -c "import darkpatterns; darkpatterns.generate(20, clear=False)" > $R/pages.log 2>&1
Q="AGENT=qwen QWEN_PATH=$M/Qwen2.5-VL-3B-Instruct DETECTOR=heuristic OMP_NUM_THREADS=2"
env $Q SPLIT=indist HOLDOUT=$TR LIMIT=60 RULES=nodef,relabel $PY analysis/p2/p2_rl.py 2>&1 | filt > $R/rl_indist_a.log &
env $Q SPLIT=indist HOLDOUT=$TR LIMIT=60 RULES=highlight,noop $PY analysis/p2/p2_rl.py 2>&1 | filt > $R/rl_indist_b.log &
env $Q SPLIT=heldout HOLDOUT=$H RULES=nodef,remove,relabel,highlight,noop $PY analysis/p2/p2_rl.py 2>&1 | filt > $R/rl_heldout.log &

{
  for s in 0 1 2 3 4 5 6 7; do echo "loto feats.pt full $s"; done
  for s in 0 1 2 3 4 5 6 7; do echo "newinst feats.pt full $s"; done
  for f in feats.pt feats_internvl.pt; do for c in full name pixel; do for s in 0 1 2 3 4 5 6 7; do
    echo "level $f $c $s"; done; done; done
} | xargs -P 9 -L 1 bash -c 'TASK=$0 FEATS_FILE=$1 CONFIG=$2 SEED=$3 THREADS=2 '"$PY"' analysis/p2/p2_scores.py' \
  > $R/scores.log 2>&1
echo CPU_DONE >> $R/scores.log
wait
touch $R/p2.done
