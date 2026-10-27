"""B-4: per-template outcomes of the heuristic gate in the scripted-agent E2E (run_e2e_learned's
heuristic arm, which the E2E log printed only in aggregate). Pages are not regenerated."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); M0 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, M0); os.chdir(M0)
import env as envmod
from detectors import HeuristicDetector
from run_e2e_learned import eval_detector

tasks = envmod.load_manifest()
e = envmod.GUIEnv(headless=True)
try:
    res = eval_detector("HEURISTIC (per-template dump)", HeuristicDetector(), e, tasks)
finally:
    e.close()
os.makedirs("results/neucom_p2", exist_ok=True)
json.dump(res, open("results/neucom_p2/e2e_heuristic_by_tpl.json", "w"), indent=1)
print("P2_HEUR_DONE", flush=True)
