"""Assemble the narrated video for one or both languages.

usage: python build.py [en|zh|both] [--scenes title,unified] [--fps 30]

Outputs (in ../output):
  post_training_systems_<lang>.mp4   video + narration + soft subtitles (EN and ZH tracks)
  post_training_systems_<lang>.<sub>.srt
  timeline_<lang>.json               scene start times (used by the script document)
"""
import argparse
import json
import os
import re
import subprocess

import numpy as np
import soundfile as sf

from engine import AMBER, H, W, background, header_layer, render_layer
from scenes import SCENES
from tts import SR, beat_audio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
BUILD = os.path.join(ROOT, "build")
os.makedirs(OUT, exist_ok=True)

SCENE_FADE = 0.8    # crossfade between scenes (s)
BEAT_FADE = 0.5     # fade/slide-in of new elements (s)
LEAD_SCENE = 0.8    # silence before narration on a new scene
LEAD_BEAT = 0.4     # silence before narration on a new beat
TAIL = 0.45         # silence after narration
END_FADE = 1.2


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


# ------------------------------------------------------------------ subtitles
def split_chunks(text, lang):
    """Split one sentence into subtitle-sized chunks."""
    maxlen = 26 if lang == "zh" else 84
    if len(text) <= maxlen:
        return [text]
    if lang == "zh":
        pieces = [p for p in re.split(r"(?<=[，、：])", text) if p]
    else:
        pieces = [p for p in re.split(r"(?<=[,:])\s+", text) if p]
    out, cur = [], ""
    for p in pieces:
        sep = "" if lang == "zh" or not cur else " "
        if len(cur) + len(sep) + len(p) <= maxlen:
            cur = cur + sep + p
        else:
            if cur:
                out.append(cur)
            cur = p
    if cur:
        out.append(cur)
    final = []
    for c in out:  # hard-wrap pieces that are still too long
        while len(c) > maxlen * 1.25:
            if lang == "zh":
                cut = maxlen
            else:
                cut = c.rfind(" ", 0, maxlen)
                cut = cut if cut > 20 else maxlen
            final.append(c[:cut].strip())
            c = c[cut:].strip()
        final.append(c)
    return final


def timed_chunks(t0, t1, text, lang):
    chunks = split_chunks(text, lang)
    total = sum(len(c) for c in chunks)
    out, t = [], t0
    for c in chunks:
        d = (t1 - t0) * len(c) / total
        out.append((t, t + d, c))
        t += d
    return out


def srt_time(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(path, items):
    with open(path, "w", encoding="utf-8") as fh:
        for i, (a, b, txt) in enumerate(items, 1):
            fh.write(f"{i}\n{srt_time(a)} --> {srt_time(b - 0.02)}\n{txt}\n\n")


def sentences_of(text, lang):
    from tts import sentences
    return sentences(text, lang)


# ------------------------------------------------------------------ main
def build(lang, scenes, fps):
    other = "zh" if lang == "en" else "en"
    bg = background()
    timeline, t = [], 0.0
    print(f"[{lang}] synthesizing narration ...", flush=True)
    for si, sc in scenes:
        for bi, beat in enumerate(sc.beats):
            audio, segs = beat_audio(beat.say[lang], lang)
            lead = LEAD_SCENE if bi == 0 else LEAD_BEAT
            tail = TAIL + (0.4 if bi == len(sc.beats) - 1 else 0.0)
            dur = lead + len(audio) / SR + tail
            timeline.append(dict(si=si, bi=bi, start=t, dur=dur, speech=t + lead,
                                 alen=len(audio) / SR, audio=audio, segs=segs))
            t += dur
    total_t = t + END_FADE
    nframes = int(round(total_t * fps))
    print(f"[{lang}] duration {total_t / 60:.1f} min, {nframes} frames", flush=True)

    # ---- audio track
    track = np.zeros(int(round(total_t * SR)) + SR, "float32")
    for e in timeline:
        s0 = int(round(e["speech"] * SR))
        track[s0:s0 + len(e["audio"])] += e["audio"]
    wav = os.path.join(BUILD, f"narration_{lang}.wav")
    sf.write(wav, track, SR)

    # ---- subtitles: own language follows real sentence audio; the other
    # language is aligned beat-by-beat across the same speech window.
    own, oth = [], []
    for e in timeline:
        for a, b, s in e["segs"]:
            own += timed_chunks(e["speech"] + a, e["speech"] + b, s, lang)
        beat = SCENES[e["si"]].beats[e["bi"]]
        sents = sentences_of(beat.say[other], other)
        tot = sum(len(s) for s in sents)
        ta = e["speech"]
        for s in sents:
            d = e["alen"] * len(s) / tot
            oth += timed_chunks(ta, ta + d, s, other)
            ta += d
    base = os.path.join(OUT, f"post_training_systems_{lang}")
    write_srt(f"{base}.{lang}.srt", own)
    write_srt(f"{base}.{other}.srt", oth)

    # ---- scene table for the script document
    marks = []
    for e in timeline:
        if e["bi"] == 0:
            sc = SCENES[e["si"]]
            marks.append(dict(sid=sc.sid, start=e["start"]))
    with open(os.path.join(OUT, f"timeline_{lang}.json"), "w") as fh:
        json.dump(dict(total=total_t, scenes=marks), fh, indent=1)

    # ---- video frames → ffmpeg
    sub_lang = {"en": "eng", "zh": "chi"}
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
           "-i", wav, "-i", f"{base}.{lang}.srt", "-i", f"{base}.{other}.srt",
           "-map", "0:v", "-map", "1:a", "-map", "2:s", "-map", "3:s",
           "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-tune", "stillimage",
           "-pix_fmt", "yuv420p",
           "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
           "-c:s", "mov_text",
           "-metadata:s:s:0", f"language={sub_lang[lang]}",
           "-metadata:s:s:1", f"language={sub_lang[other]}",
           "-disposition:s:0", "default", "-disposition:s:1", "0",
           "-metadata", "title=" + ("Post-Training Systems, Layer by Layer" if lang == "en"
                                    else "后训练工具体系：分层拆解"),
           "-shortest", "-movflags", "+faststart", f"{base}.mp4"]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    written = 0
    track_col = np.array((26, 34, 60), np.uint8)
    bar_col = np.array(AMBER, np.uint8)

    def emit(arr):
        nonlocal written
        fr = arr.copy()
        fr[H - 5:H, :] = track_col
        fill = int(W * written / max(1, nframes - 1))
        fr[H - 5:H, :fill] = bar_col
        proc.stdin.write(fr.tobytes())
        written += 1

    bg_arr = np.asarray(bg.convert("RGB")).astype(np.float32)
    prev = np.zeros((H, W, 3), np.uint8)
    state = None  # PIL RGBA composite of the current scene so far
    n_scenes = len(SCENES) - 1
    for e in timeline:
        sc = SCENES[e["si"]]
        beat = sc.beats[e["bi"]]
        f_end = int(round((e["start"] + e["dur"]) * fps))
        n = f_end - written
        layer = render_layer(beat.add, lang)
        if e["bi"] == 0:
            state = bg.copy()
            state.alpha_composite(header_layer(sc, lang, e["si"], n_scenes))
            state.alpha_composite(layer)
            target = np.asarray(state.convert("RGB"))
            # dip through the empty backdrop so two slides never overlap
            k = min(n, int(SCENE_FADE * fps))
            h = k // 2
            a0 = prev.astype(np.float32)
            a1 = target.astype(np.float32)
            for i in range(k):
                if i < h:
                    a = ease((i + 1) / h)
                    emit((a0 * (1 - a) + bg_arr * a).astype(np.uint8))
                else:
                    a = ease((i - h + 1) / (k - h))
                    emit((bg_arr * (1 - a) + a1 * a).astype(np.uint8))
        else:
            base_arr = np.asarray(state.convert("RGB")).astype(np.float32)
            lay = np.asarray(layer).astype(np.float32)
            rgb, al = lay[..., :3], lay[..., 3:4] / 255.0
            ys, xs = np.nonzero(al[..., 0] > 0)
            k = min(n, int(BEAT_FADE * fps))
            if len(ys):
                y0, y1 = max(0, ys.min() - 30), min(H, ys.max() + 30)
                for i in range(k):
                    a = ease((i + 1) / k)
                    dy = int(round((1 - a) * 22))
                    fr = base_arr.copy()
                    src_rgb = np.roll(rgb[y0:y1], dy, axis=0)
                    src_al = np.roll(al[y0:y1], dy, axis=0) * a
                    fr[y0:y1] = fr[y0:y1] * (1 - src_al) + src_rgb * src_al
                    emit(fr.astype(np.uint8))
            else:
                k = 0
            state.alpha_composite(layer)
            target = np.asarray(state.convert("RGB"))
        for _ in range(n - k):
            emit(target)
        prev = target
        print(f"[{lang}] {sc.sid}/{e['bi']}  {written}/{nframes}", flush=True)
    # fade to black
    rest = nframes - written
    a0 = prev.astype(np.float32)
    for i in range(rest):
        emit((a0 * (1 - ease((i + 1) / rest))).astype(np.uint8))
    proc.stdin.close()
    proc.wait()
    if proc.returncode:
        raise SystemExit("ffmpeg failed")
    print(f"[{lang}] wrote {base}.mp4", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("lang", nargs="?", default="both")
    ap.add_argument("--scenes", default="")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()
    sel = [(i, s) for i, s in enumerate(SCENES) if not args.scenes or s.sid in args.scenes.split(",")]
    for lg in (["en", "zh"] if args.lang == "both" else [args.lang]):
        build(lg, sel, args.fps)
