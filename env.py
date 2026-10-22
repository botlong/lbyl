"""
LBYL M0 — Playwright GUI environment.

Renders a dark-pattern page and exposes the two channels the LBYL method needs:
  * screenshot()   -> PNG bytes (the pixel channel the agent acts on)
  * observe_dom()  -> per-element {bbox, visible_text, aria_label, lbyl_role}
                      (the structural channel; ground truth for the mismatch head)
A real mouse click at (x, y) dispatches to the topmost element, so transparent
overlays (类2) resolve correctly. outcome() reads window.__lbyl_outcome.

Design: faithful computer-use setting — the agent sees ONLY the screenshot and
returns a pixel click. Targets are large to keep grounding noise low so ASR
reflects *deception*, not mis-clicks.
"""
import os, json
from playwright.sync_api import sync_playwright

PAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pages")
VW, VH = 1280, 720

_DOM_JS = r"""
() => {
  const out = [];
  const els = document.querySelectorAll('button,a,input,[onclick],[data-lbyl]');
  els.forEach((el, i) => {
    el.setAttribute('data-lbyl-id', i);   // stable handle for recheck actions
    const r = el.getBoundingClientRect();
    if (r.width < 1 || r.height < 1) {
      // still record zero-area/offscreen interactables that are clickable overlays
    }
    const cs = getComputedStyle(el);
    out.push({
      lbyl_id: i,
      tag: el.tagName.toLowerCase(),
      bbox: {x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height)},
      visible_text: (el.innerText || el.value || '').trim().slice(0, 80),
      aria_label: el.getAttribute('aria-label') || '',
      lbyl_role: el.getAttribute('data-lbyl') || '',
      opacity: cs.opacity,
      bg: cs.backgroundColor,
      // what element actually receives a click at this element's center (overlay detection):
      topmost_role: (() => {
        const cx = r.x + r.width/2, cy = r.y + r.height/2;
        const t = document.elementFromPoint(cx, cy);
        return t ? (t.getAttribute('data-lbyl') || '') : '';
      })()
    });
  });
  return out;
}
"""


class GUIEnv:
    def __init__(self, headless=True):
        self._pw = sync_playwright().start()
        self.browser = self._pw.chromium.launch(headless=headless,
                                                 args=["--no-sandbox", "--disable-dev-shm-usage"])
        self.ctx = self.browser.new_context(viewport={"width": VW, "height": VH},
                                            device_scale_factor=1)
        self.page = self.ctx.new_page()
        self.task = None

    def load(self, task):
        self.task = task
        self.page.goto("file://" + os.path.join(PAGES_DIR, task["file"]))
        self.page.wait_for_timeout(80)
        return self

    def screenshot(self):
        return self.page.screenshot(type="png")

    def screenshot_to(self, path):
        self.page.screenshot(path=path, type="png")
        return path

    def observe_dom(self):
        return self.page.evaluate(_DOM_JS)

    def apply_recheck(self, actions):
        """Execute low-level re-observation ops grounded in g_phi's findings / pi_theta's
        choice, then the agent re-observes the corrected page. This is a DUMB executor: the
        caller (heuristic scan() or the learned pi_theta) decides which op goes on which
        element. actions = [{lbyl_id, kind, text}]:
          kind="remove"   : neutralize a deceptive/overlay element (display:none + no clicks)
          kind="badge"    : draw a red warning badge over the element (known weak: a frozen
                            agent tends to ignore it — §5.4)
          kind="relabel"  : rewrite the element's VISIBLE text to its true (structural) action
                            so pixels match structure — turns a disguised control honest
          kind="highlight": positively emphasize a SAFE candidate (green outline + ✓), steering
                            the agent toward it rather than warning it away from the trap
        """
        self.page.evaluate(
            """(acts) => {
              acts.forEach(a => {
                const el = document.querySelector('[data-lbyl-id="' + a.lbyl_id + '"]');
                if (!el) return;
                if (a.kind === 'remove') {
                  el.style.display = 'none';
                  el.style.pointerEvents = 'none';
                } else if (a.kind === 'badge') {
                  const r = el.getBoundingClientRect();
                  const b = document.createElement('div');
                  b.style.cssText = 'position:absolute;z-index:99999;border:3px solid #dc2626;'
                    + 'background:rgba(220,38,38,.92);color:#fff;font:bold 13px Arial;padding:3px 6px;'
                    + 'border-radius:4px;pointer-events:none;left:' + (r.x) + 'px;top:'
                    + (Math.max(0, r.y - 22)) + 'px;max-width:' + Math.max(120, r.width) + 'px;';
                  b.textContent = '⚠ deceptive: really ' + (a.text || 'a hidden action');
                  document.body.appendChild(b);
                  el.style.outline = '3px solid #dc2626';
                } else if (a.kind === 'relabel') {
                  const t = a.text || '';
                  if (el.tagName.toLowerCase() === 'input') { el.value = t; }
                  else { el.textContent = t; }
                  el.style.outline = '2px solid #2563eb';   // marks a corrected control
                } else if (a.kind === 'highlight') {
                  el.style.outline = '4px solid #16a34a';
                  el.style.outlineOffset = '2px';
                  el.setAttribute('data-lbyl-hl', '1');   // detectable marker (smoke agent)
                  const r = el.getBoundingClientRect();
                  const g = document.createElement('div');
                  g.style.cssText = 'position:absolute;z-index:99999;background:#16a34a;color:#fff;'
                    + 'font:bold 12px Arial;padding:2px 5px;border-radius:3px;pointer-events:none;left:'
                    + (r.x) + 'px;top:' + (Math.max(0, r.y - 20)) + 'px;';
                  g.textContent = '✓ safe';
                  document.body.appendChild(g);
                }
              });
            }""", actions)
        self.page.wait_for_timeout(40)

    def outcome(self):
        o = self.page.evaluate("window.__lbyl_outcome")
        return o if o is not None else "none"

    def click(self, x, y):
        x = max(0, min(VW - 1, int(x))); y = max(0, min(VH - 1, int(y)))
        self.page.mouse.click(x, y)
        self.page.wait_for_timeout(60)
        return self.outcome()

    # --- helpers for oracle / fake agents (harness validation, no model needed) ---
    def center_of(self, lbyl_role):
        for e in self.observe_dom():
            if e["lbyl_role"] == lbyl_role and e["bbox"]["w"] > 0:
                b = e["bbox"]; return (b["x"] + b["w"] // 2, b["y"] + b["h"] // 2)
        return None

    def is_highlighted(self, lbyl_id):
        """True if apply_recheck('highlight') marked this element. Used by the CPU smoke
        agent to detect a positive steer; the real VLM agent just sees the green outline."""
        return bool(self.page.evaluate(
            "(id)=>!!document.querySelector('[data-lbyl-id=\"'+id+'\"][data-lbyl-hl=\"1\"]')",
            str(lbyl_id)))

    def reset(self):
        # reload clears window.__lbyl_outcome
        self.page.goto("file://" + os.path.join(PAGES_DIR, self.task["file"]))
        self.page.wait_for_timeout(60)

    def close(self):
        try: self.ctx.close(); self.browser.close()
        finally: self._pw.stop()


def load_manifest(name="manifest.json"):
    with open(os.path.join(PAGES_DIR, name)) as f:
        return json.load(f)
