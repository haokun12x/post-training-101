"""Render the final frame of every scene (all beats visible) for layout review."""
import os
import sys

from engine import background, header_layer, render_layer
from scenes import SCENES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = os.path.join(ROOT, "build", "preview")
os.makedirs(out, exist_ok=True)
langs = sys.argv[1:] or ["en", "zh"]
only = os.environ.get("ONLY")
bg = background()
for lang in langs:
    for i, sc in enumerate(SCENES):
        if only and sc.sid not in only.split(","):
            continue
        els = [e for b in sc.beats for e in b.add]
        im = bg.copy()
        im.alpha_composite(header_layer(sc, lang, i, len(SCENES) - 1))
        im.alpha_composite(render_layer(els, lang))
        im.convert("RGB").save(os.path.join(out, f"{lang}_{i:02d}_{sc.sid}.png"))
        print("ok", lang, i, sc.sid)
