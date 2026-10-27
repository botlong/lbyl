#!/bin/bash
# P1 rerun, stage A: re-render every element-level dataset with the computed accessible name (CPU).
set -e
cd "$(dirname "$0")"
PY=../.venv/bin/python
mkdir -p _pre_p1
for d in gphi_data gphi_data_55 gphi_data_class2 gphi_data_susbench gphi_data_susbench_dest adaptive_gphi_data gphi_data_crossmodal; do
  [ -d "$d" ] && [ ! -e "_pre_p1/$d" ] && mv "$d" "_pre_p1/$d" || true
done
$PY -c "import make_gphi_dataset as m; m.main(20, levels=(0,1,2), out_dir='gphi_data_55', manifest_name='manifest_55.json')" | tail -3
$PY -c "import make_gphi_dataset as m; m.main(20, levels=(0,))" | tail -3          # deployment set (280 tasks)
$PY make_crossmodal.py 80 realistic_v2 | tail -3
$PY make_crossmodal.py 30 crossmodal | tail -3
$PY class2_pages.py 12 | tail -4
$PY adaptive_attack.py build 12 | tail -2
$PY run_adaptive_gpu.py crop | tail -2
$PY -c "
import json,collections
for d in ['gphi_data_55','gphi_data','gphi_data_realistic_v2','gphi_data_crossmodal','gphi_data_class2','adaptive_gphi_data']:
    rows=[json.loads(l) for l in open(d+'/gphi_data.jsonl')]
    src=collections.Counter(r.get('ax_from') or 'none' for r in rows)
    same=sum((r.get('ax_name') or '')==(r.get('aria') or '') for r in rows)
    print(f'{d}: {len(rows)} rows, label1={sum(r[\"label\"] for r in rows)}, name==raw aria {same}, sources {dict(src)}')
"
echo RENDER_DONE
