"""
LBYL (Neurocomputing revision, P1-1 fix) — channel-balanced realistic set v2 with real instance variation.

The v1 set (realistic_crossmodal.py) rendered one fixed wording per action and rotated three skins,
so 1344 rows held only 168 distinct (crop, structure) inputs and every "unseen seed" test input also
occurred in training. v2 keeps the same 2x2 look-by-action factorial per dark-pattern type (aria
alone and pixels alone each predict the label at exactly 50%) but varies what is inside the crop:

  wording family f in {0,1,2,3}: four phrasings of each visible label and each accessible name
                                  (f = 0 is the v1 phrasing); a page uses one family
  skin k in {0..4}              : font, palette, button shape (v1's three plus two new)
  geometry                      : button width / height / font size drawn per page
  cell order                    : the four factorial cells are shuffled per page (no fixed layout)

Splits (crossmodal_eval.py, SPLIT=...): the held-out test uses wording family 3 AND skin 4, which
never occur in training, so no test input can be a training input (asserted on features there).

Page s uses family s % 4 and skin (s // 4) % 5, so 80 seeds give every (family, skin) cell 4 pages.
Output: pages/rv2_*.html + pages/manifest_realistic_v2.json; render with make_crossmodal.py 80 realistic_v2.
"""
import json, os, random
import realistic_crossmodal as rcm

PAGES_DIR = rcm.PAGES_DIR

# type -> four families of (A_visible, A_aria, B_visible, B_aria); A = harmful, B = safe.
FAMILIES = {
    "disguised_decline": [
        ("Accept all cookies", "Accept all cookies and ad tracking", "Reject non-essential", "Decline non-essential cookies"),
        ("Allow all cookies", "Consent to all cookies including advertising", "Only necessary cookies", "Keep only strictly necessary cookies"),
        ("Agree to all", "Agree to cookies and advertising trackers", "Decline optional cookies", "Reject all optional cookies"),
        ("Yes, allow everything", "Allow all cookies and targeted-ad tracking", "Essential cookies only", "Refuse cookies that are not required")],
    "transparent_overlay": [
        ("Subscribe now", "Subscribe for a monthly fee", "Continue without subscribing", "Continue free without subscribing"),
        ("Start subscription", "Pay a monthly fee to subscribe", "Not now, continue", "Keep reading without a subscription"),
        ("Join now", "Start a subscription charged every month", "Read without subscribing", "Proceed without paying for a plan"),
        ("Become a subscriber", "Enroll in a monthly paid plan", "Skip subscription", "Skip the subscription and continue for free")],
    "fake_close_ad": [
        ("Install the app", "Download and install the companion app", "No thanks, keep reading", "Close the ad and keep reading"),
        ("Get our app", "Install the companion mobile app", "Close", "Dismiss this ad and return to the article"),
        ("Install now", "Download our app to this device", "Maybe later", "Hide the advertisement"),
        ("Download app", "Begin installing the partner app", "Back to article", "Close the promotion and continue reading")],
    "misdirected_download": [
        ("Sponsored download", "Download sponsored third-party software", "Official installer", "Download the official verified installer"),
        ("Fast download (ad)", "Download a partner's bundled program", "Download from developer", "Download the developer's signed installer"),
        ("Partner download", "Get software from an advertiser", "Verified download", "Get the authentic official build"),
        ("Download now (sponsor)", "Download a third-party sponsored installer", "Get official version", "Download the verified first-party package")],
    "confirmshaming": [
        ("Subscribe to premium", "Start a paid premium subscription", "Continue without the discount", "Continue without subscribing or paying"),
        ("Upgrade to premium", "Upgrade to the paid premium plan", "No, I'll pay full price later", "Turn down premium and keep the free account"),
        ("Go premium", "Begin paying for premium", "Skip this offer", "Leave without purchasing anything"),
        ("Yes, upgrade me", "Purchase a premium subscription", "Not interested", "Close the offer with no payment")],
    "free_trial_paid": [
        ("Start paid subscription", "Start a paid subscription billed monthly", "Continue with the free plan", "Continue on the free plan with no payment"),
        ("Subscribe and pay", "Begin a plan that charges every month", "Use free version", "Use the free version without charges"),
        ("Begin paid plan", "Subscribe with monthly charges", "Stay free", "Keep the no-cost plan"),
        ("Start billing", "Activate a monthly billed plan", "Keep the free plan", "Continue with the plan that costs nothing")],
    "fake_system_update": [
        ("Download security software", "Download third-party security software", "Dismiss this message", "Dismiss the page notice and keep browsing"),
        ("Install antivirus", "Install an antivirus from another vendor", "Ignore", "Ignore this warning and continue"),
        ("Get security tool", "Download a third-party protection tool", "Close notice", "Close the pop-up and stay on the page"),
        ("Install protection", "Install external security software", "Not now", "Hide the message and keep browsing")],
    "preselected_addon": [
        ("Complete order with insurance", "Complete order and add paid insurance", "Complete order, no extras", "Complete the order without any paid extras"),
        ("Buy with protection plan", "Buy and add a paid protection plan", "Order without insurance", "Buy only the selected item"),
        ("Order + insurance", "Finish purchase with paid coverage added", "Buy item only", "Finish purchase with no add-ons"),
        ("Checkout with coverage", "Complete checkout plus insurance charge", "Checkout, no add-ons", "Complete checkout without insurance")],
    "disguised_unsubscribe": [
        ("Keep my subscription", "Resubscribe and keep receiving all emails", "Confirm: stop all emails", "Confirm unsubscribe and stop all emails"),
        ("Stay subscribed", "Stay on every mailing list", "Unsubscribe me", "Remove me from all mailing lists"),
        ("Keep emails coming", "Continue receiving all newsletters", "Stop emails", "Stop sending me any emails"),
        ("Remain on list", "Keep my email subscription active", "Leave mailing list", "Cancel every email subscription")],
    "bait_play_button": [
        ("Sponsored ad", "Open a sponsored third-party advertisement", "Play the video", "Play the actual video content"),
        ("Advertisement", "Go to an advertiser's website", "Watch video", "Start playback of this video"),
        ("Visit sponsor", "Open a promoted partner page", "Play", "Watch the video on this page"),
        ("Promoted link", "Follow a sponsored ad link", "Start video", "Play the embedded video")],
    "fake_captcha_allow": [
        ("Allow notifications", "Allow browser notifications and ad tracking", "Skip, I'm human", "Continue without allowing notifications"),
        ("Allow", "Grant notification permission and tracking", "Block", "Block notifications and proceed"),
        ("Enable alerts", "Turn on push alerts and ad tracking", "Continue without allowing", "Proceed with notifications off"),
        ("Turn on notifications", "Permit notifications and advertising cookies", "No, just continue", "Decline the notification request")],
    "sneak_into_basket": [
        ("Checkout with extra item", "Checkout and add a recommended paid item", "Checkout current items only", "Checkout with only the current cart items"),
        ("Checkout + recommended", "Pay for the cart and an extra product", "Checkout my cart", "Buy just the items I selected"),
        ("Buy with bonus item", "Checkout with an added paid item", "Pay for my items", "Checkout the cart as it is"),
        ("Checkout incl. add-on", "Buy the cart plus a suggested product", "Checkout without extras", "Complete purchase of current items only")],
    "inverted_consent_toggle": [
        ("Save with tracking on", "Save settings with ad tracking ON", "Turn tracking off and save", "Disable ad tracking and save settings"),
        ("Save, tracking enabled", "Save preferences and leave ad tracking on", "Disable tracking", "Turn off advertising tracking then save"),
        ("Keep tracking and save", "Keep advertising tracking enabled and save", "Save, tracking off", "Save preferences with tracking disabled"),
        ("Save (ads on)", "Save with personalized-ad tracking active", "Opt out and save", "Opt out of ad tracking and save")],
    "countdown_claim": [
        ("Claim paid membership", "Sign up for a paid monthly membership", "No thanks, just continue", "Continue without signing up or paying"),
        ("Join membership", "Enroll in a paid membership plan", "Continue as guest", "Continue as a guest with no membership"),
        ("Claim offer now", "Start a membership billed monthly", "Dismiss offer", "Dismiss the offer without paying"),
        ("Become a member", "Register for a paid member subscription", "Keep browsing", "Keep browsing without joining")],
}
TYPE_IDS = [t[0] for t in rcm.TYPES]
assert list(FAMILIES) == TYPE_IDS
assert all(tuple(FAMILIES[t[0]][0]) == tuple(t[1:]) for t in rcm.TYPES), "family 0 must be the v1 phrasing"

SKINS = rcm._SKINS + [
    ("dark", "Verdana,Geneva,sans-serif", "#111827", "#f9fafb", "#1f2937",
     "background:#374151;color:#f9fafb;border:1px solid #6b7280;border-radius:6px;"),
    ("mono", "'Courier New',Courier,monospace", "#fffbeb", "#1c1917", "#ffffff",
     "background:#fef3c7;color:#78350f;border:2px solid #d97706;border-radius:0px;"),
]
N_FAM, N_SKIN = 4, len(SKINS)


def _btn(vis, aria, cv, ca, cmid, btn_css, w, h, fs):
    b = rcm._btn(vis, aria, cv, ca, cmid, btn_css)
    return (b.replace("font-size:18px;", f"font-size:{fs}px;")
             .replace("width:320px;height:80px;", f"width:{w}px;height:{h}px;"))


def all_strings(families=range(N_FAM)):
    """Every visible label and accessible name used by the given wording families."""
    return {s for fams in FAMILIES.values() for f in families for s in fams[f]}


def generate(seeds=80, clear=True):
    os.makedirs(PAGES_DIR, exist_ok=True)
    if clear:
        for f in os.listdir(PAGES_DIR):
            if f.startswith("rv2_") and f.endswith(".html"):
                os.remove(os.path.join(PAGES_DIR, f))
    manifest = []
    for s in range(seeds):
        fam, k = s % N_FAM, (s // N_FAM) % N_SKIN
        skin = SKINS[k]
        _, font, bg, fg, card_bg, btn_css = skin
        brand = rcm._BRANDS[s % len(rcm._BRANDS)]
        for ti, tp in enumerate(TYPE_IDS):
            rng = random.Random(1000 * s + ti)
            a_vis, a_aria, b_vis, b_aria = FAMILIES[tp][fam]
            cvA, cvB = f"{tp}:A", f"{tp}:B"
            cells = [(a_vis, a_aria, cvA, cvA), (b_vis, a_aria, cvB, cvA),
                     (b_vis, b_aria, cvB, cvB), (a_vis, b_aria, cvA, cvB)]
            order = list(range(4)); rng.shuffle(order)
            w, h, fs = rng.choice([280, 320, 360]), rng.choice([70, 80, 92]), rng.choice([16, 18, 20])
            buttons = [_btn(*cells[c], c, btn_css, w, h, fs) for c in order]
            pid = f"rv2_{s:02d}_{ti:02d}"
            title = rcm._CARDS[(s + ti) % len(rcm._CARDS)].format(b=brand)
            with open(os.path.join(PAGES_DIR, pid + ".html"), "w") as f:
                f.write(rcm._page(title, brand, font, bg, fg, card_bg, buttons))
            manifest.append(dict(id=pid, file=pid + ".html", template=tp, threat_class=1, level=0,
                                 skin=skin[0], skin_idx=k, family=fam))
    with open(os.path.join(PAGES_DIR, "manifest_realistic_v2.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest, seeds * len(TYPE_IDS) * 4, seeds * len(TYPE_IDS) * 2


if __name__ == "__main__":
    import sys
    man, total, nm = generate(int(sys.argv[1]) if len(sys.argv) > 1 else 80)
    held = all_strings([3]); seen = all_strings([0, 1, 2])
    print(f"generated {len(man)} pages, {total} buttons; family-3 strings shared with families 0-2: "
          f"{sorted(held & seen)}")
