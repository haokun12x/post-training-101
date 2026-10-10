"""Export chapter titles, narration and timestamps from the video scenes for the website."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "video", "src"))
from engine import L  # noqa: E402
from scenes import SCENES  # noqa: E402

tl = {}
for lang in ("en", "zh"):
    with open(os.path.join(ROOT, "video", "output", f"timeline_{lang}.json")) as fh:
        d = json.load(fh)
    tl[lang] = d

chapters = []
for i, sc in enumerate(SCENES):
    start = {lg: next(m["start"] for m in tl[lg]["scenes"] if m["sid"] == sc.sid) for lg in tl}
    chapters.append(dict(
        sid=sc.sid,
        part={lg: L(sc.part, lg) or "" for lg in ("en", "zh")},
        title={lg: L(sc.title, lg) or ("Post-Training Systems, Layer by Layer" if lg == "en" else "后训练工具体系：分层拆解")
               for lg in ("en", "zh")},
        start=start,
        beats=[{lg: b.say[lg] for lg in ("en", "zh")} for b in sc.beats],
    ))
out = dict(total={lg: tl[lg]["total"] for lg in tl}, chapters=chapters)
path = os.path.join(ROOT, "docs", "assets", "chapters.js")
with open(path, "w", encoding="utf-8") as fh:
    fh.write("window.CHAPTERS = " + json.dumps(out, ensure_ascii=False, indent=1) + ";\n")
print("wrote", path, len(chapters))
