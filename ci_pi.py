"""Task / B(vii) — multi-seed CI for the pi_theta re-observation policy (paper Section 5.9).

Mirrors ci_gphi.py, but for the GRPO policy instead of the MLP head. For each seed we run, as
ISOLATED subprocesses (crash-safe, no shared CUDA state, no import-time-CFG contamination):
    train_pi.py   GRPO with the frozen Qwen-3B in the loop, best-epoch checkpoint, seed-specific OUT
    eval_pi.py    held-out templates + the always-safe over-intervention (FPR) check
and aggregate mean +/- 95% CI of the seed-varying headline numbers:
    * train best-epoch safe_frac   (training-time safe-completion of the kept epoch)
    * held-out pi_theta ASR(trap)  (should stay ~0 = safety)
    * held-out pi_theta safe-completion (grounding-limited on unseen layouts -> honest boundary)
    * always-safe pi_theta safe-completion (the Section 5.9 in-dist over-intervention number)

train_pi/eval_pi are reused VERBATIM (both read SEED and OUT from the env, so per-seed isolation is
just a matter of the env we hand each subprocess). We regenerate the 280-task pi_theta manifest first
(darkpatterns.generate(20)) because a prior R5 run may have left an 840-task leveled manifest that does
NOT match gphi_data/feats.pt -> train_pi would skip every state. Results stream to results/ci_pi.json
incrementally so a late-seed crash still leaves the earlier seeds' numbers.

Run (GPU box, one session):
  QWEN_PATH=/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct FEATS=gphi_data/feats.pt \
    HOLDOUT=disguised_decline,fake_close_ad,inverted_consent_toggle AGENT=qwen DETECTOR=learned \
    K=5 EPOCHS=4 LAM=0.05 THREADS=8 CI_PI_SEEDS=5 python ci_pi.py
"""
import os, sys, re, json, subprocess, statistics
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import torch
import darkpatterns

PY = sys.executable
SEEDS = int(os.environ.get("CI_PI_SEEDS", "5"))
RESULTS = os.path.join(HERE, "results")
JSON_OUT = os.path.join(RESULTS, "ci_pi.json")


def ci(xs):
    """mean +/- 95% CI (normal approx). Returns (nan, nan) if no data, (m, 0) for a single point."""
    xs = [x for x in xs if x is not None]
    if not xs:
        return float("nan"), float("nan")
    m = statistics.mean(xs)
    if len(xs) < 2:
        return m, 0.0
    return m, 1.96 * statistics.stdev(xs) / (len(xs) ** 0.5)


def parse_eval(out):
    """Pull the two pi_theta result lines from eval_pi.py stdout (label + ASR + safe on one line)."""
    def grab(tag):
        m = re.search(tag + r".*?ASR\(trap\)=([0-9.]+)\s+safe-completion=([0-9.]+)", out)
        return (float(m.group(1)), float(m.group(2))) if m else (None, None)
    held_asr, held_safe = grab(r"pi_theta \(learned, greedy\)")
    _, fpr_safe = grab(r"pi_theta on always-safe agent")
    return held_asr, held_safe, fpr_safe


def main():
    os.makedirs(RESULTS, exist_ok=True)
    # reset to the 280-task pi_theta manifest (deterministic) so crop names match gphi_data/feats.pt
    darkpatterns.generate(20)
    print(f"[ci_pi] SEEDS={SEEDS} AGENT={os.environ.get('AGENT')} DETECTOR={os.environ.get('DETECTOR')} "
          f"FEATS={os.environ.get('FEATS')} K={os.environ.get('K')} EPOCHS={os.environ.get('EPOCHS')} "
          f"HOLDOUT={os.environ.get('HOLDOUT')}", flush=True)

    per = []
    for s in range(SEEDS):
        ckpt = os.path.join(RESULTS, f"pi_theta_seed{s}.pt")
        env = dict(os.environ, SEED=str(s), OUT=ckpt)
        print(f"\n########## seed {s}: TRAIN (best-epoch ckpt -> {os.path.basename(ckpt)}) ##########",
              flush=True)
        rc = subprocess.run([PY, "train_pi.py"], env=env, cwd=HERE).returncode   # live stdout -> log
        if rc != 0 or not os.path.exists(ckpt):
            print(f"!! seed {s}: train_pi rc={rc}, ckpt_exists={os.path.exists(ckpt)} -> skip seed",
                  flush=True)
            per.append(dict(seed=s, failed=True))
            json.dump(dict(seeds=SEEDS, per_seed=per), open(JSON_OUT, "w"), indent=2)
            continue
        try:
            bsf = torch.load(ckpt).get("best_safe_frac")
        except Exception as ex:
            print(f"!! seed {s}: torch.load({os.path.basename(ckpt)}) failed: {ex!r} -> skip seed", flush=True)
            per.append(dict(seed=s, failed=True, load_error=repr(ex)[:120]))
            json.dump(dict(seeds=SEEDS, per_seed=per), open(JSON_OUT, "w"), indent=2)
            continue
        print(f"########## seed {s}: EVAL ##########", flush=True)
        p = subprocess.run([PY, "eval_pi.py"], env=env, cwd=HERE,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        sys.stdout.write(p.stdout); sys.stdout.flush()
        held_asr, held_safe, fpr_safe = parse_eval(p.stdout)
        if p.returncode != 0 or None in (held_asr, held_safe, fpr_safe):
            # eval crashed or a metric line was missing -> numbers unreliable; mark failed (don't
            # let a None silently shrink the CI's n without being visible).
            print(f"!! seed {s}: eval rc={p.returncode} parse=({held_asr},{held_safe},{fpr_safe}) "
                  f"-> mark failed", flush=True)
            per.append(dict(seed=s, best_safe_frac=bsf, failed=True, eval_rc=p.returncode,
                            held_asr=held_asr, held_safe=held_safe, fpr_safe=fpr_safe))
            json.dump(dict(seeds=SEEDS, per_seed=per), open(JSON_OUT, "w"), indent=2)
            continue
        per.append(dict(seed=s, best_safe_frac=bsf, held_asr=held_asr,
                        held_safe=held_safe, fpr_safe=fpr_safe, eval_rc=p.returncode))
        json.dump(dict(seeds=SEEDS, per_seed=per), open(JSON_OUT, "w"), indent=2)   # incremental

    ok = [p for p in per if not p.get("failed")]
    print("\n=== pi_theta multi-seed CI (mean +/- 95% CI) ===", flush=True)
    summary = {}
    for key, lbl in [("best_safe_frac", "train best-epoch safe_frac"),
                     ("held_asr", "held-out pi_theta ASR(trap)"),
                     ("held_safe", "held-out pi_theta safe-completion"),
                     ("fpr_safe", "always-safe pi_theta safe-completion")]:
        vals = [p.get(key) for p in ok if p.get(key) is not None]
        m, h = ci(vals)
        jm = None if m != m else m; jh = None if h != h else h    # nan -> None (valid JSON)
        summary[key] = dict(mean=jm, ci95=jh, n=len(vals))
        print(f"  {lbl:38} {m:.3f} +/- {h:.3f}   (n={len(vals)})", flush=True)
    json.dump(dict(seeds=SEEDS, per_seed=per, summary=summary), open(JSON_OUT, "w"), indent=2)
    print(f"saved -> {JSON_OUT}\nCI_PI_DONE", flush=True)


if __name__ == "__main__":
    main()
