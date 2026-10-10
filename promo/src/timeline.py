"""Lay the narration onto a musical grid.

Shot boundaries are shared by both languages (so one score fits both cuts);
each language places its own lines inside every shot. Times are in seconds.
"""
import math
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "video", "src"))

from tts import SR, synth  # noqa: E402

from script import SHOTS, lines  # noqa: E402

BPM = 120
BEAT = 60 / BPM
BAR = 4 * BEAT
LEAD = {"open": BAR}             # opening shot: one bar of atmosphere first
TAIL = {"outro": 2.5 * BAR, "rule": 1.0}
GAP = 0.32


def ceil_to(x, q):
    return math.ceil(x / q - 1e-6) * q


def _place(shot, lang):
    lead = LEAD.get(shot["id"], 0.45)
    t, out = lead, []
    for say, sub in lines(shot, lang):
        a = synth(say, lang)
        d = len(a) / SR
        if shot.get("on_beats"):
            t = ceil_to(t, BEAT)
        out.append(dict(t=t, d=d, say=say, sub=sub, audio=a))
        t += d + (0.12 if shot.get("on_beats") else GAP)
    end = t - (0.12 if shot.get("on_beats") else GAP) + TAIL.get(shot["id"], 0.55)
    return out, end


def build():
    shots, t0 = [], 0.0
    for sh in SHOTS:
        placed = {lg: _place(sh, lg) for lg in ("en", "zh")}
        dur = ceil_to(max(p[1] for p in placed.values()), 2 * BEAT)
        shots.append(dict(id=sh["id"], label=sh["label"], start=t0, dur=dur,
                          cues={lg: [dict(c, t=t0 + c["t"]) for c in placed[lg][0]] for lg in placed}))
        t0 += dur
    return dict(bpm=BPM, beat=BEAT, bar=BAR, total=t0, shots=shots)


if __name__ == "__main__":
    tl = build()
    for s in tl["shots"]:
        print(f"{s['id']:10s} start {s['start']:6.2f}  dur {s['dur']:5.2f}  bars {s['dur'] / BAR:4.1f}")
    print("total", round(tl["total"], 2))
