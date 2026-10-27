"""
Oracle / fake agents — validate the harness AND the defended loop WITHOUT a GPU.

All agents expose  act(env, task, warning=None) -> (x, y)  so they drop into the
DefendedAgent wrapper exactly like the real Qwen VLM agent.
"""

class OracleSafe:
    """Clicks the center of the element truly achieving the goal."""
    name = "oracle_safe"
    def act(self, env, task, warning=None):
        return env.center_of("safe") or (1, 1)

class OracleTrap:
    """Clicks the center of the trap element."""
    name = "oracle_trap"
    def act(self, env, task, warning=None):
        return env.center_of("trap") or (1, 1)

class SalientThenSafe:
    """Simulates a susceptible agent: drawn to the salient deceptive (trap) element,
    but RE-GROUNDS to the safe control once the defense warns it. Lets us test the
    abstain vs reground behaviours with no model."""
    name = "salient_then_safe"
    def act(self, env, task, warning=None):
        if warning is None:
            return env.center_of("trap") or (1, 1)        # naive: clicks the trap
        return env.center_of("safe") or (1, 1)            # warned: picks the safe control


def _center_of_elem(e):
    b = e["bbox"]
    return (b["x"] + b["w"] // 2, b["y"] + b["h"] // 2)


class StubbornAgent:
    """CPU smoke-test agent mimicking the REAL frozen model's key property (§5.4): drawn to
    the salient trap and IGNORES warning badges, but DOES respond to interventions that change
    what it actually perceives. Lets the GRPO loop (train_pi.py) learn 'avoid badge/noop,
    prefer remove/relabel/highlight' with NO GPU. After a warning (re-observing the corrected
    page) it decides by inspecting the corrected DOM, NOT by being told which op was applied:
      1. trap neutralized (gone / zero-area in the DOM)         -> clicks safe   (+)
      2. trap relabeled honest (visible text now == its action) -> sees it's bad, clicks safe (+)
      3. a safe control highlighted (green ✓ marker)            -> drawn to the highlighted safe (+)
      4. only badged / noop (trap still salient & disguised)    -> STUBBORN: clicks trap again (-)
    Unlike SalientThenSafe (which obeys ANY warning), this agent only changes its click when the
    PAGE changed — so badge and noop fail, exactly the gap §5.4 found in the real model.
    """
    name = "stubborn"

    def act(self, env, task, warning=None):
        dom = env.observe_dom()
        trap = next((e for e in dom if e.get("lbyl_role") == "trap" and e["bbox"]["w"] > 0), None)
        safe = next((e for e in dom if e.get("lbyl_role") == "safe" and e["bbox"]["w"] > 0), None)
        safe_xy = _center_of_elem(safe) if safe else (1, 1)
        if warning is None:
            return _center_of_elem(trap) if trap else safe_xy           # naive: drawn to trap
        if trap is None:                                                # 1. trap neutralized
            return safe_xy
        vis = (trap.get("visible_text") or "").strip().lower()
        aria = (trap.get("aria_label") or "").strip().lower()
        if vis and aria and vis == aria:                                # 2. relabeled honest
            return safe_xy
        if safe is not None and env.is_highlighted(safe.get("lbyl_id")):  # 3. safe highlighted
            return safe_xy
        return _center_of_elem(trap)                                    # 4. badge/noop -> stubborn
