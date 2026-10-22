"""
LBYL real-transfer probe — render SusBench's dark-pattern injections OFFLINE and emit
(crop, structural-channel, label) rows so the synthetically-trained g_phi can be tested
ZERO-SHOT on EXTERNALLY-AUTHORED real dark patterns.

SusBench (arXiv 2510.11035, github.com/SusBench-creator/SusBench) ships its dark patterns as
browser-extension code (`injection-mapping.js`) that BUILDS each widget at runtime. We never
touch the 55 live sites: we load that script in a blank Playwright page over a generic
product-page STUB and call each entry's executeInjection() directly (bypassing shouldInject).
Ground-truth labels come from SusBench's OWN avoidance hooks: an element whose onclick calls
agentNotAvoid() is the DECEPTIVE one (label 1); agentAvoid() is the SAFE one (label 0). No
clicking/navigation needed — we read element.onclick.toString().

Two structural-channel variants are recorded per element (BOTH parsed from the DOM, NEITHER
peeks at the label):
  aria : the element's real aria-label (the channel g_phi was TRAINED on; usually EMPTY in the
         wild -> tests whether the as-trained signal even EXISTS on real dark patterns).
  dest : a short phrase for the element's real navigation/effect, parsed from href / onclick
         window.location (the "what it ACTUALLY does" channel that IS in the real DOM) -> tests
         whether the cross-modal MECHANISM transfers when the structural channel is matched to
         the real threat.

dp_class tags each entry:
  action_mismatch : disguised ads (visible label vs navigation target)   -> g_phi's TARGET class
  visual          : false hierarchy / preselect / pop-up / social / confirm-shaming /
                    forced_action / hidden_information (honest labels, deceptive layout)
  linguistic      : double negatives (confusing wording; aria == visible)

Output: gphi_data_susbench/{crops/*.png, gphi_data.jsonl}        (structural channel = real aria)
        gphi_data_susbench_dest/{crops -> same, gphi_data.jsonl} (structural channel = dest)
Then:   GPHI_DATA_DIR=gphi_data_susbench      python extract_feats.py
        GPHI_DATA_DIR=gphi_data_susbench_dest python extract_feats.py
        SB_DIR=gphi_data_susbench      python susbench_eval.py
        SB_DIR=gphi_data_susbench_dest python susbench_eval.py
Pure Playwright + PIL -> GPU-free.
"""
import os, sys, io, json
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from PIL import Image
from playwright.sync_api import sync_playwright

SUS = os.environ.get("SUSBENCH_DIR", "/root/autodl-tmp/SusBench/dark-pattern-browser-extension")
INJECTION_JS = os.path.join(SUS, "injection-mapping.js")
OUT = os.path.join(HERE, "gphi_data_susbench")
OUT_DEST = os.path.join(HERE, "gphi_data_susbench_dest")
VW, VH = 1280, 1400   # tall viewport so body-appended modals/overlays sit fully on screen


def classify(pattern):
    if pattern.startswith("disguised"):
        return "action_mismatch"
    if pattern.startswith(("double_neg", "doube_negative")):
        return "linguistic"
    return "visual"   # false_hierarchy/preselect/pop_up/social/confirm*/forced_action/hidden_information


# Generous product-page stub so host-anchored injections (disguised/preselect insertion points)
# find their hooks. 118/123 entries append to body and ignore this; the rest insert near these
# anchors. Missing-anchor entries simply no-op (reported as empty).
STUB_HTML = """<!doctype html><html><head><meta charset=utf-8>
<style>body{font-family:Arial,Helvetica,sans-serif;margin:0;padding:20px;background:#fff;color:#111}
.wrap{max-width:760px;margin:0 auto}.price{font-size:28px;color:#B12704;margin:10px 0}
button{font-size:15px;padding:10px 16px;margin:6px 4px}</style></head>
<body><div class="wrap">
  <h1 id="title">Premium Wireless Headphones</h1>
  <div class="price">$129.99</div>
  <div id="dp-upgrade-section"></div>
  <div id="mbb-offeringID-1"></div>
  <div id="addToCart_feature_div">
    <button id="add-to-cart-btn" aria-label="Add to cart">Add to Cart</button>
    <button id="add-to-cart" aria-label="Add to cart">Add to Cart</button>
    <button aria-label="Buy now">Buy Now</button>
    <button aria-label="Add to bag" data-test="addToCartButton" class="add-to-cart">Add to bag</button>
  </div>
  <div id="fake-advertisement-container"></div>
  <div class="rs-cto-main-panel"></div>
  <div class="rf-configuration-productsummary"></div>
  <div class="js-product-pip"></div>
  <form id="checkout"><input type=checkbox><button>Continue</button></form>
</div></body></html>"""

# Installed BEFORE inj_src loads: wrap addEventListener so that when a widget attaches a click
# listener whose SOURCE calls agentNotAvoid()/agentAvoid(), we tag the element with data-sb-label.
# This is what catches the MAJORITY of SusBench widgets (they attach handlers via addEventListener,
# not .onclick=), without any clicking/navigation.
INSTALL_JS = r"""
() => {
  if (window.__sbPatched) return; window.__sbPatched = true;
  const _add = EventTarget.prototype.addEventListener;
  EventTarget.prototype.addEventListener = function(type, fn, opts){
    try {
      if (type === 'click' && typeof fn === 'function' && this.setAttribute) {
        const s = fn.toString();
        if (s.includes('agentNotAvoid')) this.setAttribute('data-sb-label', '1');
        else if (s.includes('agentAvoid')) this.setAttribute('data-sb-label', '0');
        const m = s.match(/location(?:\.href)?\s*=\s*['"]([^'"]+)['"]/);
        if (m) this.setAttribute('data-sb-nav', m[1]);
      }
    } catch(e){}
    return _add.call(this, type, fn, opts);
  };
}
"""

# Collect every SusBench-LABELED interactive element after an injection runs.
COLLECT_JS = r"""
() => {
  const SAFE = 'agentAvoid', BAD = 'agentNotAvoid';
  const MAXAREA = 0.45 * %d * %d;   // drop full-screen overlays (out-of-distribution for a button head)
  const out = [];
  const els = document.querySelectorAll(
    'button,a,[onclick],[data-sb-label],input[type=checkbox],input[type=radio],div[id*=overlay],div[id*=popup],[role=button]');
  let idx = 0;
  els.forEach((el) => {
    let src = '';
    try { if (typeof el.onclick === 'function') src = el.onclick.toString(); } catch(e){}
    let label = null;
    const tagged = el.getAttribute('data-sb-label');      // set by the addEventListener wrapper
    if (tagged === '1') label = 1;
    else if (tagged === '0') label = 0;
    else if (src.includes(BAD)) label = 1;
    else if (src.includes(SAFE)) label = 0;
    if (label === null) return;                       // keep only SusBench-labeled elements
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) return;
    if (r.width * r.height > MAXAREA) return;          // skip overlay divs
    // dest phrase: real navigation target / effect, parsed from href or onclick (NOT the label)
    let dest = '';
    const href = (el.getAttribute && el.getAttribute('href')) || '';
    const navAttr = (el.getAttribute && el.getAttribute('data-sb-nav')) || '';   // from addEventListener wrapper
    const mSrc = src.match(/location(?:\.href)?\s*=\s*['"]([^'"]+)['"]/);
    const m = mSrc || (navAttr ? [null, navAttr] : null);
    if (m) { try { dest = 'navigate to ' + new URL(m[1]).hostname; } catch(e){ dest = 'navigate to ' + m[1]; } }
    else if (href && href !== '#') { try { dest = 'navigate to ' + new URL(href, location.href).hostname; } catch(e){ dest = 'navigate to ' + href; } }
    else if (/removeChild|closePopup|display\s*=\s*['"]none|\.remove\(\)/.test(src)) dest = 'close dialog';
    else if (/checked\s*=\s*true|\.check\(/.test(src)) dest = 'select option';
    else dest = (el.innerText || '').trim().slice(0, 40);   // no nav -> structural ~ visible (no extra signal)
    out.push({
      sb_id: idx, tag: el.tagName.toLowerCase(), label: label,
      bbox: {x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height)},
      visible_text: (el.innerText || el.value || '').trim().slice(0, 80),
      aria: el.getAttribute('aria-label') || '', dest: dest,
      opacity: getComputedStyle(el).opacity,
    });
    idx++;
  });
  return out;
}
""" % (VW, VH)


def main(limit=0):
    for d in (OUT, OUT_DEST):
        os.makedirs(os.path.join(d, "crops"), exist_ok=True)
        for f in os.listdir(os.path.join(d, "crops")):
            if f.endswith(".png"):
                os.remove(os.path.join(d, "crops", f))

    inj_src = open(INJECTION_JS).read()
    pw = sync_playwright().start()
    browser = pw.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
    ctx = browser.new_context(viewport={"width": VW, "height": VH}, device_scale_factor=1)
    ctx.route("**/*", lambda route: route.abort())   # block all nav/network: injected location.href can't leave
    page = ctx.new_page()
    page.on("dialog", lambda d: d.dismiss())

    page.set_content(STUB_HTML)
    page.add_script_tag(content=inj_src)
    keys = page.evaluate("() => Object.keys(window.injectionMapping || {})")
    print(f"discovered {len(keys)} injection entries", flush=True)
    if limit:
        keys = keys[:limit]

    rows = []
    cov = Counter()
    for k in keys:
        site, _, pattern = k.partition("_")
        dpc = classify(pattern)
        page.set_content(STUB_HTML)
        page.evaluate(INSTALL_JS)            # wrap addEventListener BEFORE the widget attaches handlers
        page.add_script_tag(content=inj_src)
        err = page.evaluate(
            "(k) => { try { window.injectionMapping[k].executeInjection(); return ''; }"
            " catch(e){ return String(e); } }", k)
        page.wait_for_timeout(120)
        if err:
            cov[f"err:{pattern}"] += 1
            continue
        els = page.evaluate(COLLECT_JS)
        if not els:
            cov[f"empty:{pattern}"] += 1
            continue
        img = Image.open(io.BytesIO(page.screenshot(type="png"))).convert("RGB")
        for el in els:
            b = el["bbox"]; pad = 6
            box = (max(0, b["x"] - pad), max(0, b["y"] - pad),
                   min(img.width, b["x"] + b["w"] + pad), min(img.height, b["y"] + b["h"] + pad))
            cid = f"{k}_e{el['sb_id']}.png"
            crop = img.crop(box)
            crop.save(os.path.join(OUT, "crops", cid))
            crop.save(os.path.join(OUT_DEST, "crops", cid))
            rows.append(dict(
                crop=cid, template=pattern, site=site, pattern=pattern, dp_class=dpc,
                level=0, threat_class=1, role=el["tag"], visible_text=el["visible_text"],
                opacity=el["opacity"], bbox=[b["x"], b["y"], b["w"], b["h"]],
                aria_real=el["aria"], dest=el["dest"], susbench_key=k, label=el["label"]))
        cov[f"ok:{pattern}"] += 1
    browser.close(); pw.stop()

    def write(d, aria_field):
        with open(os.path.join(d, "gphi_data.jsonl"), "w") as f:
            for r in rows:
                rr = dict(r); rr["aria"] = r[aria_field]
                f.write(json.dumps(rr) + "\n")
    write(OUT, "aria_real")
    write(OUT_DEST, "dest")

    pos = sum(r["label"] for r in rows)
    print(f"\nwrote {len(rows)} elements ({pos} deceptive / {len(rows) - pos} safe)")
    byc = Counter((r["dp_class"], r["label"]) for r in rows)
    print("by (class,label):", dict(byc))
    print("coverage:", dict(cov))
    nonempty = sum(1 for r in rows if r["aria_real"].strip())
    print(f"elements with a real aria-label: {nonempty}/{len(rows)}  "
          f"(few -> as-trained aria channel is inert in the wild)")
    print("SUSBENCH_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
