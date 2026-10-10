"""Write the bilingual (EN | 中文) narration script with scene timestamps."""
import json
import os

from engine import L
from scenes import SCENES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")


def mmss(t):
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


tl = {}
for lang in ("en", "zh"):
    p = os.path.join(OUT, f"timeline_{lang}.json")
    if os.path.exists(p):
        with open(p) as fh:
            d = json.load(fh)
        tl[lang] = {m["sid"]: m["start"] for m in d["scenes"]}
        tl[lang]["_total"] = d["total"]

lines = [
    "# Post-Training Systems, Layer by Layer · 后训练工具体系：分层拆解",
    "",
    "Bilingual narration script for `post_training_systems_en.mp4` and `post_training_systems_zh.mp4`.",
    "中英对照解说稿，对应英文版与中文版视频。时间戳格式 EN / 中文。",
    "",
]
if tl:
    lines.append(f"Duration 时长: EN {mmss(tl.get('en', {}).get('_total', 0))} · 中文 {mmss(tl.get('zh', {}).get('_total', 0))}")
    lines.append("")
lines += ["## Contents 目录", ""]
for i, sc in enumerate(SCENES):
    title = L(sc.title, "en") or "Title"
    ztitle = L(sc.title, "zh") or "标题"
    ts = " / ".join(mmss(tl[lg][sc.sid]) for lg in ("en", "zh") if lg in tl and sc.sid in tl[lg])
    lines.append(f"{i:02d}. {title} · {ztitle}" + (f"  `{ts}`" if ts else ""))
lines.append("")
for i, sc in enumerate(SCENES):
    title = L(sc.title, "en") or "Post-Training Systems, Layer by Layer"
    ztitle = L(sc.title, "zh") or "后训练工具体系：分层拆解"
    lines.append(f"## {i:02d} · {title} · {ztitle}")
    part = L(sc.part, "en")
    if part:
        lines.append(f"*{part} · {L(sc.part, 'zh')}*")
    ts = [f"{lg.upper() if lg == 'en' else '中文'} {mmss(tl[lg][sc.sid])}" for lg in ("en", "zh")
          if lg in tl and sc.sid in tl[lg]]
    if ts:
        lines.append("")
        lines.append("⏱ " + " · ".join(ts))
    lines.append("")
    lines.append("| English | 中文 |")
    lines.append("|---|---|")
    for b in sc.beats:
        en = b.say["en"].replace("|", "\\|")
        zh = b.say["zh"].replace("|", "\\|")
        lines.append(f"| {en} | {zh} |")
    lines.append("")

with open(os.path.join(OUT, "script_bilingual.md"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
print("wrote", os.path.join(OUT, "script_bilingual.md"))
