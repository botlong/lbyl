# Look Before You Leap (LBYL)

<p align="center" width="100%">
<img src="assets/lbyl_overview.png" alt="LBYL overview" style="width: 95%; min-width: 300px; display: block; margin: auto;">
</p>

This repository accompanies our paper "[Look Before You Leap? A Shortcut-Audited Evaluation of Screenshot–Structure Consistency Checks for GUI Agents](#authors-and-citation)" (submitted to *Neurocomputing*). It contains the offline deceptive-interface environment, the mismatch head `g_phi`, the correction policy `pi_theta`, every evaluation protocol in the paper, and the result logs behind each reported number, for both the earlier and the corrected pipeline.

**What LBYL is.** A GUI agent can observe a control in two ways: through its pixels in the screenshot, and through its structure in the accessibility tree (the browser-computed accessible name and role). Some deceptive interfaces make the two disagree, for example a button that *looks* like "Reject non-essential" but is *declared* as "Accept all cookies and tracking". LBYL freezes a vision-language model, encodes both channels of the control a click would reach, and trains a small head that compares them through `[p, s, p-s, p*s]`. When the head flags the control, the page is corrected (for example, a hijacking overlay is removed), the agent chooses again, and the new target is checked again on the re-captured page.

**What the paper finds, in short.** The paper is an evaluation study, and most of its findings are limits:

- On a concept-balanced factorial set, where retrained single-channel heads stay at chance, the head is above chance, so it benefits from access to both channels. It generalizes poorly, though: accuracy is 0.63 on unseen wording and appearance, below a zero-shot prompted VLM (0.72, at a false-positive rate of 0.54).
- On unseen dark-pattern types its recall is 0.73 at FPR 0.17, against 0.83 at 0.21 for a keyword heuristic. The difference is not significant over the fourteen types.
- Its clearest strength is lying accessible names (Class 2): it flags 0.72–0.89 of them, against 0.21–0.57 for the heuristic.
- Run live as the gate of a scripted agent it is unstable across seeds. A trained correction policy only matches the best constant rule.
- In the part of SusBench we could re-create offline, we found no control whose accessible name contradicts its pixels.
- An earlier version of this pipeline reported much stronger numbers. Two implementation details (encoding the raw `aria-label` instead of the computed accessible name, and testing on inputs that also occurred in training) caused most of the difference; the paper's Section 6.8 and Table 9 quantify each one.

Please treat LBYL as an instrument for studying the perceptual-discrepancy class, not as a detector to deploy on real pages.

<p align="center" width="100%">
<img src="assets/crossmodal.png" alt="Channel-balanced realistic set" style="width: 85%; min-width: 300px; display: block; margin: auto;">
</p>
<p align="center"><em>Channel-balanced realistic set (chance 0.50). (a) The learned head by split, eight seeds; single-channel heads stay at 0.50. (b) New-instance split against a zero-shot prompted VLM and a crop-only proxy for DUDE's visual-only setting (not DUDE itself).</em></p>

## :hammer_and_wrench: Setting up

### Step 1. Python requirements
We tested on Python 3.12 with torch 2.8 (CUDA 12.8) and transformers 5.x.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

The batch scripts (`p1_*.sh`, `analysis/p2/p2_run.sh`) call `PY=../.venv/bin/python`; edit that line if your environment lives elsewhere.

### Step 2. Models
The main backbone and agent is Qwen2.5-VL-3B-Instruct. The threat study also uses Qwen2.5-VL-7B and -32B, and the backbone replication uses InternVL3-8B (the `-hf` checkpoint). Only the 3B model is needed for the core pipeline.

```bash
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir /path/to/models/Qwen2.5-VL-3B-Instruct
export GPHI_MODEL=/path/to/models/Qwen2.5-VL-3B-Instruct   # feature extractor for g_phi
export QWEN_PATH=/path/to/models/Qwen2.5-VL-3B-Instruct    # frozen agent
```

Default paths in the scripts point at our machine (`/root/lbyl-neucom/...`); each script's docstring lists the variables it reads.

**Hardware.** Page generation, the keyword heuristic, and training `g_phi` on extracted features run on CPU. Feature extraction and agent rollouts need a GPU; a 24 GB card is enough for the 3B and 7B models. The 32B threat run used a 96 GB card.

## :running: Reproducing the paper

All commands run from the repository root. The corrected pipeline runs in five stages; each writes its logs to `results/neucom_p1/` or `results/neucom_p2/`.

### Stage 0. Generate the environment and run the self-test (CPU)

```bash
python run_m0.py --gen 8      # 14 templates x 3 difficulty levels -> pages/
python run_m0.py --selftest   # every trap click -> "trap", every safe click -> "safe"
```

In `transparent_overlay`, clicking the *visible* safe button lands on the invisible overlay and is recorded as `trap`; that is the deception working as intended.

### Stage A. Render every element-level dataset (CPU)

```bash
bash p1_render.sh
```

This builds the environment set (`gphi_data_55`, 1840 controls), the deployment set (`gphi_data`, the candidate controls of the 280 end-to-end tasks), the channel-balanced realistic set (`gphi_data_realistic_v2`, 4480) and grid (`gphi_data_crossmodal`, 720), the Class-2 set (`gphi_data_class2`, 1008), and the adaptive-attack crops. Every control records its crop, its browser-computed accessible name and role (read through the DevTools protocol, `ax.py`), and its geometry. The SusBench re-creation needs a local checkout of SusBench:

```bash
SUSBENCH_DIR=/path/to/SusBench/dark-pattern-browser-extension python susbench_probe.py
```

### Stage B. Extract frozen features (GPU)

```bash
bash p1_extract.sh
# InternVL3-8B replication:
GPHI_MODEL=/path/to/InternVL3-8B-hf GPHI_DATA_DIR=gphi_data_55 FEATS_OUT=feats_internvl.pt python extract_feats_internvl.py
GPHI_MODEL=/path/to/InternVL3-8B-hf GPHI_DATA_DIR=gphi_data_realistic_v2 FEATS_OUT=feats_internvl.pt python extract_feats_internvl.py
```

### Stage C. Detector evaluations (CPU)

```bash
bash p1_cpu_evals.sh
```

This runs leave-one-template-out (LOTO) and the difficulty split with channel ablations and eight seeds (`train_gphi.py`, `ci_gphi.py`), the channel-balanced group splits and all 364 three-type hold-outs (`crossmodal_eval.py`), the Class-2 zero-shot test (`class2_eval.py`), and SusBench transfer (`susbench_eval.py`). The name/role factorial of the implementation audit is `analysis/p1/factor_audit.py`.

### Stage D. Everything with the VLM in the loop (GPU)

```bash
bash p1_gpu.sh            # STEPS=truthfree,e2e,e3,prompted,m1,rl,adaptive (default: all)
```

| Step | What it runs |
|---|---|
| `truthfree` | deletes all ground-truth markup from every page and checks that all 560 detector decisions stay the same (`test_truthfree.py`) |
| `e2e` | the learned head as live detector in the scripted agents' loop, eight head seeds (`run_e2e_learned.py`) |
| `e3` | the three remedies fixed in advance in `results/neucom_20261025/E3_PLAN.md`: ensemble, prompted fallback, mixed training (`e3_stabilize.py`) |
| `prompted` | the zero-shot prompted VLM and the crop-only proxy (`prompted_baseline.py`) |
| `m1` | the 144-task real-agent sweep with Qwen2.5-VL-3B/7B, heuristic and learned gate (`run_m1.py`) |
| `rl` | trains and evaluates `pi_theta`, the constant "always remove" rule, and the five-seed learned-gate runs (`train_pi.py`, `eval_pi.py`, `eval_fixed_remove.py`, `ci_pi.py`) |
| `adaptive` | the detector-aware adaptive attack (`run_adaptive_gpu.py`) |

### Stage E. Second-round checks and statistics

```bash
bash analysis/p2/p2_run.sh          # RL baselines and all four constant rules; per-row head scores
python3 analysis/p2/p2_analyze.py   # AUROC, matched-FPR recall, template bootstrap, sign tests
```

### Tips
- **Generation is deterministic.** All generated sets are rebuilt identically from the generators; only SusBench needs an external repository.
- **Regenerate pages before feature-dependent runs.** `pages/manifest.json` must match the crops the features were extracted from.
- **Mind the threads on CPU.** Set `OMP_NUM_THREADS` (and `THREADS` where a script reads it) to avoid oversubscription.
- **Playwright can occasionally fail to take a screenshot.** This is transient; re-run the affected evaluation.

<p align="center" width="100%">
<img src="assets/adaptive.png" alt="Adaptive attack" style="width: 85%; min-width: 300px; display: block; margin: auto;">
</p>
<p align="center"><em>Detector-aware adaptive attack. Evasion succeeds on the label axis only once the accessibility label stops disclosing the true action; opacity is not an evasion axis.</em></p>

## :bar_chart: Where each number comes from

Section, table, and figure numbers refer to the journal manuscript. All paths are under `results/`.

| Paper result | Archived output |
|---|---|
| Sec. 6.1, Fig. 2: threat by model size and level (archived runs; no detector) | `m1_sweep_results.json`, `m1_results_32b.json`, `R5_32b_summary.txt`; re-run on the current stack: `neucom_p1/m1_sweep_heuristic.log` |
| Sec. 6.2, Table 3: LOTO, difficulty split, ablations, eight seeds | `neucom_p1/train_gphi_{base,nonum,zerovis,zerotxt,nonum_internvl}.log`, `neucom_p1/ci_gphi_8seed{,_internvl}.log` |
| Sec. 6.2: AUROC, matched-FPR recall, template tests; difficulty split over eight seeds | `neucom_p2/p2_analysis.txt`, `neucom_p2/scores/` |
| Sec. 6.3, Fig. 3, Table 4: channel-balanced sets, 364 hold-outs, hidden states, InternVL | `neucom_p1/cm_v2_{qwen,qwen_hidden,internvl,sweep364}.log`, `neucom_p1/cm_grid_pair.log` |
| Sec. 6.3: prompted VLM and crop-only proxy | `neucom_p1/prompted_{v2_newinstance,env55,deploy,susbench}.log` |
| Sec. 6.3, Table 5: Class-2, zero-shot | `neucom_p1/class2_eval.log` |
| Sec. 6.4: real agents with heuristic and learned gate | `neucom_p1/m1_sweep_{heuristic,mixed}.{log,json}` |
| Sec. 6.4, Table 6: live learned detector, eight seeds | `neucom_p1/e2e_8seed.{log,json}` |
| Sec. 6.4, Table 6: pre-specified remedies | plan: `neucom_20261025/E3_PLAN.md`; logs: `neucom_p1/e3.log`; fallback re-run on unrounded probabilities: `neucom_p2/e3c_unrounded_*.log` |
| Sec. 6.5, Table 7: SusBench transfer | `neucom_p1/susbench_{eval.log,transfer.txt}`, per-seed rates: `neucom_p2/susbench_perseed_*.json` |
| Sec. 6.6, Table 8: correction policy | `neucom_p1/rl_{train_heur,eval_indist_heur,eval_heldout_heur,fixed_remove_heur,ci_learned}.log`; no defense and the four constant rules: `neucom_p2/rl_*.{log,json}` |
| Sec. 6.7, Fig. 4: adaptive attack | `neucom_p1/adaptive/`, `neucom_p1/adaptive.log` |
| Sec. 6.8, Table 9: implementation audit | name/role factorial and overlap audit: `neucom_p1/analysis/`; ground-truth deletion test: `neucom_p1/truthfree.log`; earlier-pipeline logs: `repro_20260708/`, `neucom_20261025/`, and the top-level files of `results/` |

`neucom_p1/provenance/` records the input-archive hashes and the code hashes of the server run. The earlier-pipeline logs are kept unchanged so that every number in Table 9 can be traced; do not mix them with the corrected results.

## :wrench: Customization

### Adding a dark-pattern template
1. Write a generator in `darkpatterns.py` following an existing one, such as `t_disguised_decline`. It takes `(seed, level)` and returns a page in which every clickable element carries `data-lbyl="safe" | "trap" | "distractor"`. Detectors never read this attribute; `test_truthfree.py` checks that.
2. Add it to `TEMPLATES` at the bottom of `darkpatterns.py`.
3. Run `python run_m0.py --gen 8 && python run_m0.py --selftest`.
4. Re-run Stages A–C. LOTO automatically includes the new template as a held-out type.

### Plugging in a different VLM backbone
`g_phi` needs two vectors per control, a pixel feature and a structural feature from the same model. `extract_feats_internvl.py` shows how to adapt `extract_feats.py`; it writes the same `feats.pt` layout, and the head projects each channel to 256 dimensions whatever the input size. Point the downstream scripts at the new file with `FEATS_FILE`.

## Authors and Citation

**Authors:** Hua Long, Haolong Shi, Chongyang Dai (Chongqing University of Technology)

```
@article{long2026lbyl,
  title   = {Look Before You Leap? A Shortcut-Audited Evaluation of Screenshot--Structure Consistency Checks for GUI Agents},
  author  = {Long, Hua and Shi, Haolong and Dai, Chongyang},
  journal = {Submitted to Neurocomputing},
  year    = {2026}
}
```

## Acknowledgements

Our dark-pattern taxonomy builds on [DECEPTICON](https://arxiv.org/abs/2512.22894) and [SusBench](https://arxiv.org/abs/2510.11035). The real-world transfer probe executes SusBench's publicly released injection mapping, and we thank its authors for making it available. We use [Qwen2.5-VL](https://arxiv.org/abs/2502.13923) and [InternVL3](https://arxiv.org/abs/2504.10479) under their model licenses.

This project is released under the MIT License (see `LICENSE`).
