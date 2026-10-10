"""Build the finished promo: audio mix + rendered frames + subtitles → mp4.

usage: python make.py en|zh|both [--fps 60] [--audio-only] [--mux-only]

The Chinese cut is music-only (no narration; captions are burned in).
"""
import argparse
import os
import subprocess
import sys

import soundfile as sf

from music import SR, make_audio
from timeline import build

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "build")
OUT = os.path.join(ROOT, "output")
os.makedirs(OUT, exist_ok=True)


def srt_time(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(tl, lang, path):
    items = []
    for s in tl["shots"]:
        cs = s["cues"][lang]
        for i, c in enumerate(cs):
            end = cs[i + 1]["t"] if i + 1 < len(cs) else s["start"] + s["dur"]
            txt = c["sub"].rstrip("。，；") if lang == "zh" else c["sub"]
            items.append((c["t"], min(c["t"] + c["d"] + 0.5, end), txt))
    with open(path, "w", encoding="utf-8") as fh:
        for i, (a, b, t) in enumerate(items, 1):
            fh.write(f"{i}\n{srt_time(a)} --> {srt_time(b - 0.02)}\n{t}\n\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lang", default="both")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--audio-only", action="store_true")
    ap.add_argument("--mux-only", action="store_true", help="reuse the rendered frames in build/")
    a = ap.parse_args()
    tl = build()
    for lang in (["en", "zh"] if a.lang == "both" else [a.lang]):
        wav = os.path.join(BUILD, f"promo_audio_{lang}.wav")
        sf.write(wav, make_audio(tl, lang, narration=(lang != "zh")), SR, subtype="FLOAT")
        srt = os.path.join(OUT, f"promo_{lang}.srt")
        write_srt(tl, lang, srt)
        print("audio", wav, flush=True)
        if a.audio_only:
            continue
        vid = os.path.join(BUILD, f"promo_video_{lang}.mp4")
        if not a.mux_only:
            subprocess.run([sys.executable, os.path.join(ROOT, "src", "render.py"), lang, "--fps", str(a.fps), "--out", vid],
                           check=True, cwd=os.path.join(ROOT, "src"))
        out = os.path.join(OUT, f"promo_{lang}.mp4")
        title = "Post-Training Systems, Layer by Layer — Trailer" if lang == "en" else "后训练工具体系：分层拆解 · 预告"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", vid, "-i", wav, "-i", srt,
                        "-map", "0:v", "-map", "1:a", "-map", "2:s",
                        "-c:v", "libx264", "-preset", "slow", "-crf", "20",
                        "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.2",
                        "-af", "loudnorm=I=-14:TP=-1.0:LRA=9", "-c:a", "aac", "-b:a", "256k", "-ar", "48000",
                        "-c:s", "mov_text", "-metadata:s:s:0", "language=" + ("eng" if lang == "en" else "chi"),
                        "-metadata", "title=" + title, "-movflags", "+faststart", out], check=True)
        print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
