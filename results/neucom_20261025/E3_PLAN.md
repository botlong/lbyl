# E3 plan (written and committed before any E3 run, 2026-10-25)

Trigger: E2 showed the deployed learned detector's end-to-end ASR is 0.318 +- 0.110 over eight head
seeds (seed 0 = 0.071, the number in the C&S draft), versus 0.143 for the heuristic.

Question: can the live learned detector be made stable without tuning on the deployment tasks?

Arms (all reported, whatever the outcome):
- **E3a (primary)**: average P(mismatch) of the eight deployed heads (seeds 0-7, realistic
  channel-balanced training, numeric features removed); flag at 0.5. No free parameters.
- **E3b**: leave-one-template-out end to end. For each template T, train on realistic rows of the
  other 13 types plus environment-page rows of the other 13 types; run live on T's 20 tasks only.
  Eight seeds; results pooled over templates per seed.
- **E3c**: when a head's P(mismatch) is in [0.3, 0.7], use the zero-shot prompted VLM verdict
  (prompted_baseline.py MODE=neutral, Qwen2.5-VL-3B, the prompt already reported in the paper).
  Eight seeds, plus once on top of the E3a ensemble.

Fixed in advance: tau = 0.5; band = [0.3, 0.7]; prompt mode = neutral; scripted agents and
deployment tasks as in E2. Nothing is tuned on the deployment tasks.

Metrics: susceptible-agent ASR and safe completion under reground; always-safe completion and
refusal under reground. Mean +- 95% CI half-width (normal approximation) where there are seeds.

Reporting rule: the paper reports E3a as the primary remedy and E3b/E3c as secondary, including any
arm that does not help. If no arm beats the heuristic's 0.143 ASR, the paper says so.

## Addendum (2026-10-25 14:20, after E3a ran and while E3b was running; no E3c result had been seen)
The deployment-set crops (`gphi_data/crops`) were never archived, only their features, so the first
prompted-VLM dump read 0 crops and the first E3c pass would have had no verdicts to fall back on.
That E3c pass is discarded. Fix: re-render the deployment crops into a separate directory
(`gphi_data_deploy_rerender`, same generator and seeds), check that their features match the
archived `gphi_data/feats.pt` (per-crop cosine similarity), dump the prompted verdicts on those
crops, and re-run E3c alone with the same pre-specified band and prompt. If features do not match,
E3c is reported as not run.
