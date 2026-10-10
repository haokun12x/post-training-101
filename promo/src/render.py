"""Render the promo video frames (and stills for review).

usage: python render.py en|zh [--fps 60] [--still 12.5,30] [--out path.mp4]
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np
from PIL import Image

import shots as S
from gfx import H, W, Frame, finish, ramp, smooth
from timeline import build

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "build")

_bg = S.Backdrop()


def frame_at(tl, t, lang):
    n = len(tl["shots"])
    si = max(i for i, s in enumerate(tl["shots"]) if s["start"] <= t + 1e-9)
    sh = tl["shots"][si]
    cx = S.Ctx(sh, lang, tl["beat"])
    lt = t - sh["start"]
    f = Frame()
    grid = {"chaos": 0.6, "stack": 0.8, "loop": 0.5, "landscape": 0.7, "outro": 0.4}.get(sh["id"], 0.0)
    intro = smooth(ramp(t, 0.0, 2.0))
    _bg.draw(f, t, dust=intro, grid=grid * smooth(ramp(lt, 0, 0.6)) * smooth(ramp(sh["dur"] - lt, 0, 0.4)))
    S.SHOT_FN[sh["id"]](f, lt, cx)
    # captions
    if sh["id"] in S.CAPTIONED or sh["id"] == "landscape":
        for i, c in enumerate(cx.cues):
            if sh["id"] == "landscape" and i < 6:
                continue
            nxt = cx.cues[i + 1]["t"] if i + 1 < len(cx.cues) else sh["dur"]
            t1 = min(c["t"] + c["d"] + 0.5, nxt)
            a = smooth(ramp(lt, c["t"] - 0.08, 0.15)) * (1 - smooth(ramp(lt, t1 - 0.15, 0.15)))
            S.caption(f, c["sub"], lang, a)
    # HUD
    ha = smooth(ramp(t, tl["shots"][1]["start"], 0.6)) * (1 - smooth(ramp(t, tl["shots"][-1]["start"], 0.8)))
    S.hud(f, t, si + 1, n, sh["label"][0] if lang == "en" else sh["label"][1], lang, ha)
    # cut accent: tiny flash + chromatic split at each shot start (not the first)
    k = lt if si > 0 else 9
    flash = 0.05 * np.exp(-k * 14)
    aberr = 5 * np.exp(-k * 10)
    end = 1.0
    return finish(f, t, flash=float(flash), aberr=float(aberr) if aberr > 0.5 else 0, exposure=end)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lang")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--still", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    tl = build()
    if a.still:
        os.makedirs(os.path.join(BUILD, "stills"), exist_ok=True)
        for s in a.still.split(","):
            t = float(s)
            t0 = time.time()
            img = frame_at(tl, t, a.lang)
            p = os.path.join(BUILD, "stills", f"{a.lang}_{t:07.2f}.png")
            Image.fromarray(img).save(p)
            print(p, f"{time.time() - t0:.2f}s")
        return
    out = a.out or os.path.join(BUILD, f"promo_video_{a.lang}.mp4")
    n = int(round(tl["total"] * a.fps))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "17",
           "-pix_fmt", "yuv420p", "-x264-params", "aq-mode=3", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    for i in range(n):
        proc.stdin.write(frame_at(tl, i / a.fps, a.lang).tobytes())
        if i % 300 == 0:
            print(f"[{a.lang}] {i}/{n}  {time.time() - t0:.0f}s", flush=True)
    proc.stdin.close()
    proc.wait()
    with open(os.path.join(BUILD, f"timeline_{a.lang}.json"), "w") as fh:
        json.dump({"total": tl["total"], "shots": [dict(id=s["id"], start=s["start"], dur=s["dur"]) for s in tl["shots"]]},
                  fh, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
