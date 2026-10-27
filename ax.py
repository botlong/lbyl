"""
LBYL — browser-computed accessible name and role (the structural channel, P1-2 fix).

The earlier pipeline encoded the raw `aria-label` attribute plus the tag name. That is not the
accessible name: when `aria-label` is absent the browser derives the name from the content,
`aria-labelledby`, an associated <label>, `title`, and so on. Here we ask Chromium itself, through
the DevTools protocol, for each element's computed accessibility node:

  ax_name : the accessible name the browser exposes to assistive technology (and to agents that
            read the accessibility tree)
  ax_role : the computed role (button, link, generic, ...)
  ax_from : which source produced the name ("attribute:aria-label", "contents", ...; "" if none)

Elements are addressed by an attribute the caller has already set on them (e.g. data-lbyl-id).
"""


def _name_source(name):
    for s in (name or {}).get("sources", []):
        if "value" in s and not s.get("superseded"):
            return s.get("type", "") + (":" + s["attribute"] if s.get("attribute") else "")
    return ""


def ax_by_attr(page, attr):
    """{attr value (str): {ax_name, ax_role, ax_from}} for every element carrying `attr`."""
    cdp = page.context.new_cdp_session(page)
    out = {}
    try:
        cdp.send("DOM.enable")
        cdp.send("Accessibility.enable")
        root = cdp.send("DOM.getDocument", {"depth": 0})["root"]["nodeId"]
        ids = cdp.send("DOM.querySelectorAll", {"nodeId": root, "selector": f"[{attr}]"})["nodeIds"]
        for nid in ids:
            a = cdp.send("DOM.getAttributes", {"nodeId": nid})["attributes"]
            key = dict(zip(a[::2], a[1::2])).get(attr)
            bid = cdp.send("DOM.describeNode", {"nodeId": nid})["node"]["backendNodeId"]
            nodes = cdp.send("Accessibility.getPartialAXTree",
                             {"backendNodeId": bid, "fetchRelatives": False})["nodes"]
            n = next((x for x in nodes if x.get("backendDOMNodeId") == bid), nodes[0] if nodes else {})
            name = n.get("name") or {}
            out[key] = dict(ax_name=(name.get("value") or "").strip(),
                            ax_role=(n.get("role") or {}).get("value", "") or "",
                            ax_from=_name_source(name),
                            ax_ignored=bool(n.get("ignored", False)))
    finally:
        try:
            cdp.detach()
        except Exception:
            pass
    return out


def struct_text(row):
    """The structural-channel string the frozen text encoder sees: computed name + computed role."""
    if "ax_name" not in row:
        raise KeyError("row has no ax_name: re-render the dataset with the accessibility-tree pipeline")
    return ((row.get("ax_name") or "").strip() or " ") + " " + (row.get("ax_role") or "")
