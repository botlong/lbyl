# LBYL: Look Before You Leap

Code, evaluation harness, and result logs for the paper

> **Look Before You Leap: Cross-Modal Screenshot-versus-DOM Consistency Checking for Defending GUI Agents against Deceptive Interfaces**
> Hua Long, Haolong Shi, Chongyang Dai. Chongqing University of Technology. Under review at *Computers & Security*.

LBYL defends vision-language GUI agents against deceptive interfaces (dark patterns) by checking whether
the two channels an agent can observe agree: the **pixels** of a control (screenshot crop) and its
**structure** (accessible name / role in the DOM). On a frozen VLM it trains a small mismatch head
`g_phi` over the comparison basis `[p, s, p-s, p*s]`. When the head flags a control, a corrective
re-observation step edits the page (e.g. removes a hijacking overlay) before the agent commits.

The paper reports both what this signal achieves and where it stops working. Its scope is the
*perceptual-discrepancy* class, where screenshot and DOM disagree. It does **not** transfer to real
SusBench dark patterns, which keep both channels consistent. Please read the paper's scope and
limitations before using LBYL as a general dark-pattern defense.

## Repository layout

| File(s) | Role |
|---|---|
| `darkpatterns.py` | Generates the offline environment: 14 dark-pattern templates x 3 difficulty levels (`pages/*.html` + `manifest.json`) |
| `env.py` | Playwright environment: screenshot (pixel channel), DOM observation (structural channel), real mouse clicks (topmost element wins, so overlays behave as in a browser) |
| `agent.py`, `fake_agents.py` | Qwen2.5-VL screenshot-only agent; GPU-free oracle agents for harness tests |
| `consistency.py`, `detectors.py` | Keyword heuristic detector; detector interface used inside the agent loop |
| `make_gphi_dataset.py`, `extract_feats.py`, `extract_feats_internvl.py` | Element crops + frozen-VLM features (Qwen2.5-VL / InternVL3) |
| `learned_gphi.py`, `train_gphi.py`, `ci_gphi.py` | Mismatch head `g_phi`: model, leave-one-template-out training, multi-seed CI |
| `crossmodal.py`, `realistic_crossmodal.py`, `make_crossmodal.py`, `crossmodal_eval.py` | Shortcut-proof look x action factorial sets (grid / realistic) and channel-ablation evaluation |
| `prompted_baseline.py`, `run_dude_lite.sh` | Prompted-VLM baseline and DUDE-protocol (screenshot-only) proxy |
| `defended_agent.py`, `run_m0.py`, `run_m1.py`, `test_defense.py`, `run_e2e_learned.py` | End-to-end defended agent, threat sweep, oracle and live-detector evaluations |
| `policy.py`, `train_pi.py`, `eval_pi.py`, `ci_pi.py`, `eval_fixed_remove.py` | Correction policy `pi_theta` (GRPO-style RL with a frozen agent), evaluation, 5-seed CI, constant-remove baseline |
| `susbench_probe.py`, `susbench_eval.py`, `susbench_logit_probe.py` | Offline re-creation of SusBench injections and zero-shot transfer evaluation |
| `adaptive_attack.py`, `run_adaptive_gpu.py`, `test_adaptive_logic.py` | White-box adaptive attacker (aria-morph and overlay-opacity families) |
| `results/`, `adaptive_out/` | Archived logs and JSON for every number in the paper (see the map below) |

## Setup

```bash
pip install -r requirements.txt
playwright install chromium
```

Tested with Python 3.12, torch 2.8 (CUDA 12.8), and transformers 5.x. The exact environment is
recorded in `results/repro_20260708/pip_freeze.txt`. Generating the environment, running the
heuristic, and training or evaluating `g_phi` on extracted features all run on CPU. VLM feature
extraction and agent rollouts need a GPU: a 24 GB card is enough for the 3B/7B models, and the 32B
agent was run in bf16 on a 96 GB card (about 65 GB of weights).

Model and data locations default to the paths on the original training machine (`/root/...`).
Override them with environment variables, for example `GPHI_MODEL`, `QWEN_PATH`, `GPHI_DATA_DIR`,
`SUSBENCH_DIR`, and `PLAYWRIGHT_BROWSERS_PATH`. Each script's docstring lists the variables it reads.

## Quick start

```bash
python run_m0.py --gen 8        # generate pages (8 instances per template)
python run_m0.py --selftest     # GPU-free harness check with oracle agents
python test_adaptive_logic.py   # GPU-free self-test of the adaptive-attack harness
```

## Paper result -> command map

Section, table, and figure numbers refer to the journal manuscript.

| Paper artifact | Script / config | Data | Archived output |
|---|---|---|---|
| RQ1 (Sec. 6.1, Fig. 2) 3B/7B by-level sweep | `run_m1.py` (4 templates × 3 levels × 12; modes none/abstain/reground) | `pages/` | `results/m1_sweep_results.json` |
| RQ1 (Sec. 6.1, Fig. 2) 32B full + matched pool | `run_m1.py` (14 × 3 × 20, none) + `compute_r5_bylevel.py` | `pages/` | `results/m1_results_32b.json`, `results/R5_32b_summary.txt` |
| RQ2 (Sec. 6.2) heuristic 80/80 · Table 3 heuristic rows | `consistency.py` scored over `gphi_data_55/gphi_data.jsonl` (CPU-recomputable) | `gphi_data_55/` | — |
| RQ2 (Sec. 6.2) oracle re-observation (80 L0 tasks) | `test_defense.py` after `darkpatterns.generate(20, templates=dp.TEMPLATES[:4])` (= 4 originals × 20 @ L0) | `pages/` | `results/repro_20260708/test_defense_oracle80.log` (safe completion 0→0.75, 0.25 refused) |
| RQ2 (Sec. 6.2) Table 3 learned rows + numeric/channel ablations | `train_gphi.py` (`GPHI_DATA_DIR=gphi_data_55`; `NO_NUMERIC=1`, zero-vis/zero-txt); 8-seed CI: `ci_gphi.py` | `gphi_data_55/feats.pt` | `results/repro_20260708/train_gphi_{base,nonum,zerovis,zerotxt}.log`, `ci_gphi_8seed.log` (deterministic re-run) |
| RQ3 (Sec. 6.3, Fig. 3) grid / realistic shortcut-proof | `make_crossmodal.py` / `realistic_crossmodal.py` → `extract_feats.py` → `crossmodal_eval.py`; InternVL: `extract_feats_internvl.py` | `gphi_data_crossmodal/` (rebuilt 2026-07-09, `make_crossmodal.py 30`), `gphi_data_realistic_cm/` | `results/R4_internvl_summary.txt`; `results/repro_20260708/crossmodal_grid_{embed,hidden}.log`, `chrome_holdout.log`, `ho_sweep_results.txt` (364-split sweep, mean 0.62±0.01) |
| RQ3 (Sec. 6.3, Table 4) geometry-leak audit | `crossmodal_eval.py` with CONFIGS extended to FULL(no-num)/NUM-ONLY over the archived feats (fixed 2×2 layout makes bbox label-predictive: x=111 all label-0 / x=453 all label-1) | `gphi_data_realistic_cm/feats{,_internvl}.pt` | `results/denum_check_20260712.log`, `results/ho_sweep_denum_20260712.txt` (off-box CPU re-check, torch 2.13; FULL(+num) anchors match `chrome_holdout.log`/`R4_internvl_summary.txt` bit-for-bit) |
| RQ3 (Sec. 6.3) prompted-VLM / DUDE-protocol proxy | `prompted_baseline.py` via `run_dude_lite.sh`; the paper's "cross-modal prompt" = `MODE=neutral` (de-biased crop+aria), not `MODE=crossmodal` | `gphi_data_realistic_cm/` (crops rebuilt, jsonl md5-identical) | `results/repro_20260708/dude_lite.log`, `prompted_neutral_{realistic,55}.log`, `prompted_susbench.log` (Regime B on re-rendered crops: dude acc 0.47) |
| RQ4 (Sec. 6.4, Table 5) live learned detector | `run_e2e_learned.py` | `gphi_data/feats.pt` + regenerated `pages/` | `results/R3_summary.txt` |
| RQ5 (Sec. 6.5, Table 6) SusBench transfer | `susbench_probe.py` (needs SusBench repo checkout) → `susbench_eval.py` | `gphi_data_susbench{,_dest}/` | `results/susbench_transfer.txt`; cold re-run 2026-07-08: `results/repro_20260708/susbench_eval.log` (tables bit-identical, adds the "feature provenance OK" line; logit diagnostics: `susbench_logit_probe.py`) |
| RQ6 (Sec. 6.6, Table 7) πθ headline + 5-seed CI | `train_pi.py` → `eval_pi.py`; `ci_pi.py` | `gphi_data/feats.pt` | `results/{full_train,eval_indist,eval}.log`, `results/ci_pi.json`, `results/ci_pi_n5.md`; clean n=5 machine record: `results/repro_20260708/ci_pi_rerun.json` (2026-07-09, held-out ASR 0.00±0.00 / always-safe completion 0.667±0.00 both reproduce exactly) |
| RQ6 (Sec. 6.6, Table 7) constant-remove baseline | `eval_fixed_remove.py` (same protocol/harness as `eval_pi.py`, action pinned to `remove`; no feats needed) | `pages/` | `results/eval_fixed_remove_20260712.log` (off-box 5070Ti; fixed-badge anchor matches `eval_indist.log` bit-for-bit, CONST-REMOVE = 0.683/0.217 ties π_θ → the +23 is the action choice, not RL) |
| RQ7 (Sec. 6.7, Fig. 4) adaptive frontier | `run_adaptive_gpu.py` (`adaptive_attack.py`) | `adaptive_gphi_data/` | `adaptive_out/frontier_adaptive_{learned,heuristic}.json` |
| Fig. 3 bars | values transcribed from the `crossmodal_eval` and `dude_lite` logs above | — | — |

Gotchas: (1) `pages/manifest.json` must match `gphi_data/feats.pt` crop keys —
`run_e2e_learned.py` regenerates pages only if the manifest is missing, so delete a
stale `pages/` first (`ci_pi.py` regenerates unconditionally). (2) All `crops/`
subdirectories are empty locally; env/cm crops regenerate deterministically from the
generators, SusBench crops need the external SusBench repo. (3) Default model/data
paths point at the training box (`/root/...`) — override via the documented env vars.
(4) `results/ci_pi.json` shows seed0 `failed` and n=4 summaries — that seed's eval
subprocess hit a transient Playwright screenshot crash; `results/ci_pi_n5.md` records
the successful seed0 retry (identical values → the paper's clean n=5). A cold re-run
of seed0 should make the json itself n=5. Also: the json field `fpr_safe` is a
misnomer — it is always-safe *safe-completion* (`ci_pi.py:52,108`), so 0.667 means
~33% over-intervention, not 67%. (5) `results/R5_32b_summary.txt`'s original
"normal scaling" conclusion is retracted in its appended 2026-07-08 erratum (pool
confound); quote the erratum / paper Sec. 6.1 numbers, not the original 3-point table.


## License

MIT, see `LICENSE`.
