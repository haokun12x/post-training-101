"""Procedural score + sound design + narration mix for the promo (torch, 48 kHz stereo).

Everything is synthesized here (no samples): additive detuned-saw pads with a moving
low-pass, a plucked arpeggio with ping-pong delay, sub bass, kick/clap/hats, risers,
impacts, UI ticks, a convolution reverb, and voice-keyed ducking.
"""
import math

import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SR = 48000
G = torch.Generator(device=DEV).manual_seed(1234)
DEBUG = {}


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def noise(n):
    return torch.randn(n, device=DEV, generator=G)


def fft_filter(x, fn):
    """Static frequency-domain filter; fn(freqs_tensor) -> gain."""
    n = x.shape[-1]
    X = torch.fft.rfft(x, n=n)
    f = torch.fft.rfftfreq(n, 1 / SR).to(DEV)
    return torch.fft.irfft(X * fn(f), n=n)


def lp(fc, order=2):
    return lambda f: 1 / torch.sqrt(1 + (f / fc) ** (2 * order))


def hp(fc, order=2):
    return lambda f: 1 / torch.sqrt(1 + (fc / f.clamp(min=1)) ** (2 * order))


def bp(fc, q=1.5):
    return lambda f: 1 / torch.sqrt(1 + q * q * (f / fc - fc / f.clamp(min=1)) ** 2)


class Bus:
    def __init__(self, n):
        self.x = torch.zeros(2, n, device=DEV)
        self.n = n

    def add(self, sig, t0, gain=1.0, pan=0.0):
        """Add mono (n,) or stereo (2,n) at time t0 (s)."""
        i0 = int(round(t0 * SR))
        if sig.dim() == 1:
            l = math.cos((pan + 1) * math.pi / 4)
            r = math.sin((pan + 1) * math.pi / 4)
            sig = torch.stack([sig * l * 1.414, sig * r * 1.414])
        j0, j1 = max(0, i0), min(self.n, i0 + sig.shape[1])
        if j1 > j0:
            self.x[:, j0:j1] += sig[:, j0 - i0:j1 - i0] * gain


# ------------------------------------------------------------------ instruments
def pad_note(f0, dur, fc_fn, t_abs0, attack=1.2, release=2.2, voices=4, spread=0.11):
    n = int((dur + release) * SR)
    t = torch.arange(n, device=DEV) / SR
    fc = fc_fn(t + t_abs0)                                  # time-varying cutoff (n,)
    out = torch.zeros(2, n, device=DEV)
    kmax = max(1, min(48, int(9000 / f0)))
    k = torch.arange(1, kmax + 1, device=DEV, dtype=torch.float32).view(-1, 1)
    for v in range(voices):
        det = (v - (voices - 1) / 2) / max(1, voices - 1) * spread       # semitones
        f = f0 * 2 ** (det / 12)
        ph0 = torch.rand(kmax, 1, device=DEV, generator=G) * 6.283
        gain = (1 / k) / torch.sqrt(1 + (k * f / fc.view(1, -1)) ** 4)
        s = (torch.sin(2 * math.pi * f * k * t.view(1, -1) + ph0) * gain).sum(0)
        pan = -0.7 + 1.4 * v / max(1, voices - 1)
        out[0] += s * math.cos((pan + 1) * math.pi / 4)
        out[1] += s * math.sin((pan + 1) * math.pi / 4)
    e = torch.clamp(t / attack, max=1.0) ** 2
    e = e * torch.where(t > dur, torch.exp(-(t - dur) * 4.0 / release), torch.ones_like(t))
    return out * e / voices


def pluck(f0, dur=0.42, bright=1.0):
    n = int(dur * SR)
    t = torch.arange(n, device=DEV) / SR
    kmax = max(1, min(24, int(8000 / f0)))
    k = torch.arange(1, kmax + 1, device=DEV, dtype=torch.float32).view(-1, 1)
    env_k = torch.exp(-t.view(1, -1) * (5 + 2.2 * k / bright))
    s = (torch.sin(2 * math.pi * f0 * k * t.view(1, -1)) / k * env_k).sum(0)
    return s * torch.clamp(t / 0.003, max=1) * torch.exp(-t * 4)


def kick():
    n = int(0.5 * SR)
    t = torch.arange(n, device=DEV) / SR
    f = 46 + 115 * torch.exp(-t * 32)
    ph = torch.cumsum(2 * math.pi * f / SR, 0)
    s = torch.sin(ph) * torch.exp(-t * 5.5)
    click = noise(n) * torch.exp(-t * 400) * 0.25
    return torch.tanh((s + click) * 1.6) * 0.9


def clap():
    n = int(0.4 * SR)
    t = torch.arange(n, device=DEV) / SR
    e = sum(torch.exp(-(t - d).clamp(min=0) * 90) * (t >= d) for d in (0, 0.011, 0.022))
    e = e + torch.exp(-t * 14) * 0.5
    return fft_filter(noise(n) * e, bp(1400, 1.2)) * 0.8


def hat(open_=False):
    n = int((0.3 if open_ else 0.08) * SR)
    t = torch.arange(n, device=DEV) / SR
    s = fft_filter(noise(n), hp(7000, 3)) * torch.exp(-t * (14 if open_ else 70))
    return s * 0.5


def tick(freq=2200):
    n = int(0.12 * SR)
    t = torch.arange(n, device=DEV) / SR
    s = torch.sin(2 * math.pi * freq * t) * torch.exp(-t * 60) + torch.sin(2 * math.pi * freq * 1.5 * t) * torch.exp(-t * 90) * 0.4
    return s * 0.5


def thump(f=62):
    n = int(0.6 * SR)
    t = torch.arange(n, device=DEV) / SR
    fr = f + 40 * torch.exp(-t * 25)
    return torch.sin(torch.cumsum(2 * math.pi * fr / SR, 0)) * torch.exp(-t * 7)


def impact(length=3.0):
    n = int(length * SR)
    t = torch.arange(n, device=DEV) / SR
    fr = 30 + 40 * torch.exp(-t * 4)
    boom = torch.sin(torch.cumsum(2 * math.pi * fr / SR, 0)) * torch.exp(-t * 1.6)
    nz = fft_filter(noise(n), lp(2500)) * torch.exp(-t * 5) * 0.5
    return torch.tanh((boom + nz) * 1.3)


def riser(length):
    n = int(length * SR)
    t = torch.arange(n, device=DEV) / SR
    x = t / length
    out = torch.zeros(n, device=DEV)
    hop = int(0.04 * SR)
    win = torch.hann_window(2 * hop, device=DEV)
    for i0 in range(0, n - 2 * hop, hop):
        c = 300 * (8000 / 300) ** ((i0 + hop) / n)
        seg = fft_filter(noise(2 * hop), bp(c, 2.0)) * win
        out[i0:i0 + 2 * hop] += seg
    sweep = torch.sin(torch.cumsum(2 * math.pi * (180 * 2 ** (x * 3)) / SR, 0)) * 0.15
    return (out * 1.4 + sweep) * x ** 2.2


def whoosh(length=0.6):
    n = int(length * SR)
    t = torch.arange(n, device=DEV) / SR
    x = t / length
    out = torch.zeros(n, device=DEV)
    hop = int(0.03 * SR)
    win = torch.hann_window(2 * hop, device=DEV)
    for i0 in range(0, n - 2 * hop, hop):
        c = 600 * 10 ** ((i0 + hop) / n)
        out[i0:i0 + 2 * hop] += fft_filter(noise(2 * hop), bp(c, 1.2)) * win
    return out * torch.sin(math.pi * x) ** 2 * 0.6


def reverb(x, rt=2.8, wet=1.0, seed=5):
    """Stereo convolution reverb with a synthetic exponential IR."""
    n_ir = int(rt * SR)
    t = torch.arange(n_ir, device=DEV) / SR
    g = torch.Generator(device=DEV).manual_seed(seed)
    ir = torch.randn(2, n_ir, device=DEV, generator=g) * torch.exp(-t * 6.9 / rt)
    ir = fft_filter(ir, lp(5500, 1))
    ir[:, :int(0.018 * SR)] = 0
    ir = ir / ir.abs().sum(1, keepdim=True).clamp(min=1e-6) * 18
    n = x.shape[1] + n_ir
    nfft = 1 << (n - 1).bit_length()
    y = torch.fft.irfft(torch.fft.rfft(x, nfft) * torch.fft.rfft(ir, nfft), nfft)[:, :x.shape[1]]
    return y * wet


def resample_24_48(a):
    a = torch.as_tensor(a, device=DEV, dtype=torch.float32)
    n = a.shape[0]
    X = torch.fft.rfft(a)
    Y = torch.zeros(n + 1, dtype=X.dtype, device=DEV)
    Y[:X.shape[0]] = X
    return torch.fft.irfft(Y, n=2 * n) * 2


# ------------------------------------------------------------------ score
PROG = [  # (pad voicing, bass root), two bars each
    ([50, 57, 60, 64, 65], 38),   # Dm9
    ([46, 53, 57, 60, 62], 34),   # Bbmaj9
    ([53, 57, 60, 64, 67], 41),   # Fmaj9
    ([48, 55, 62, 64, 67], 36),   # Cadd9
]


def make_audio(tl, lang):
    beat, bar = tl["beat"], tl["bar"]
    total = tl["total"]
    n = int((total + 1.0) * SR)
    S = {s["id"]: s for s in tl["shots"]}
    st = lambda k: S[k]["start"]          # noqa: E731
    en = lambda k: S[k]["start"] + S[k]["dur"]  # noqa: E731
    cues = {s["id"]: s["cues"][lang] for s in tl["shots"]}

    pads, arp, drums, sfx, sub, voice = (Bus(n) for _ in range(6))

    # filter automation for the pad (Hz) by section
    keys = [(0, 380), (st("chaos"), 700), (st("equation"), 1300), (st("fork"), 2000), (st("math"), 2600),
            (st("differ"), 1100), (st("landscape"), 1700), (st("rule"), 3000), (st("outro"), 2000),
            (total, 700)]
    kt = torch.tensor([k[0] for k in keys], device=DEV)
    kv = torch.tensor([math.log(k[1]) for k in keys], device=DEV)

    def fc_fn(t):
        idx = torch.searchsorted(kt, t.clamp(max=float(kt[-1]) - 1e-3), right=True).clamp(1, len(keys) - 1)
        t0, t1 = kt[idx - 1], kt[idx]
        a = ((t - t0) / (t1 - t0)).clamp(0, 1)
        a = a * a * (3 - 2 * a)
        return torch.exp(kv[idx - 1] + (kv[idx] - kv[idx - 1]) * a)

    # ---- harmony: chord every two bars
    nch = int(math.ceil(total / (2 * bar)))
    final_chord_t = st("outro")
    for c in range(nch):
        t0 = c * 2 * bar
        if t0 >= final_chord_t - 1e-6:
            break
        voicing, root = PROG[c % 4]
        dur = min(2 * bar, final_chord_t - t0) + 0.15
        for m in voicing:
            pads.add(pad_note(mtof(m), dur, fc_fn, t0), t0, 0.16)
        sub_n = int((dur + 0.3) * SR)
        tt = torch.arange(sub_n, device=DEV) / SR
        sb = (torch.sin(2 * math.pi * mtof(root) * tt) + 0.18 * torch.sin(4 * math.pi * mtof(root) * tt))
        sb = sb * torch.clamp(tt / 0.25, max=1) * torch.clamp((dur + 0.3 - tt) / 0.3, max=1)
        level = 0.0 if t0 < st("equation") - 1e-6 else 0.22
        if st("open") <= t0 < st("chaos"):
            level = 0.18
        sub.add(sb, t0, level)
    # final chord: Dm(add9) held through the outro
    for m in [38, 50, 57, 62, 64, 65, 69]:
        pads.add(pad_note(mtof(m), total - final_chord_t - 2.0, fc_fn, final_chord_t, attack=0.3, release=3.0),
                 final_chord_t, 0.15)

    # ---- arpeggio (16ths), sections where it plays and its brightness
    pattern = [0, 2, 4, 1, 3, 2, 4, 3, 0, 2, 4, 1, 3, 4, 2, 1]
    arp_on = [(st("chaos") + 2 * bar, st("equation"), 0.45, 0.10), (st("equation"), st("rule"), 1.0, 0.14),
              (st("outro"), st("outro") + 4 * bar, 0.6, 0.08)]
    step = beat / 4
    for a0, a1, bright, g in arp_on:
        k = int(math.ceil(a0 / step))
        while k * step < a1 - 1e-6:
            t = k * step
            voicing, _ = PROG[int(t // (2 * bar)) % 4]
            tones = [m + 12 for m in voicing[1:]] + [voicing[1] + 24]
            m = tones[pattern[k % 16] % len(tones)]
            vel = (1.0 if k % 4 == 0 else 0.72) * g
            pl = pluck(mtof(m), bright=bright)
            arp.add(pl, t, vel, pan=-0.25 if k % 2 else 0.25)
            k += 1
    # ping-pong delay on the arp
    d = int(0.375 * SR)
    dl = torch.zeros_like(arp.x)
    src = arp.x.clone()
    for i in range(1, 5):
        sh = src.roll(i * d, 1)
        sh[:, :i * d] = 0
        side = 0 if i % 2 else 1
        dl[side] += (sh[0] + sh[1]) * 0.5 * (0.42 ** i)
    arp.x += fft_filter(dl, lp(3500))

    # ---- drums
    K, C, Hc, Ho = kick(), clap(), hat(), hat(True)
    groove = [(st("equation"), st("fork"), "half"), (st("fork"), st("math"), "four"), (st("math"), st("differ"), "full"),
              (st("landscape") + 0, st("rule") - 2 * beat, "four")]
    kicks = []
    for g0, g1, mode in groove:
        b = int(math.ceil(g0 / beat - 1e-6))
        while b * beat < g1 - 1e-6:
            t = b * beat
            pos = b % 4
            if mode == "half":
                if pos in (0, 2):
                    drums.add(K, t, 0.85); kicks.append(t)
                drums.add(Hc, t + beat / 2, 0.10, pan=0.3)
            else:
                drums.add(K, t, 0.9); kicks.append(t)
                drums.add(Hc, t + beat / 2, 0.16, pan=0.3)
                if pos in (1, 3):
                    drums.add(C, t, 0.30, pan=-0.05)
                if mode == "full":
                    drums.add(Hc, t + beat / 4, 0.06, pan=-0.3)
                    drums.add(Hc, t + 3 * beat / 4, 0.06, pan=-0.3)
                    if pos == 3:
                        drums.add(Ho, t + beat / 2, 0.08, pan=0.2)
            b += 1
    # sidechain pump on pads/sub/arp from the kicks
    pump = torch.ones(n, device=DEV)
    pl = int(0.32 * SR)
    shape = 1 - 0.45 * torch.exp(-torch.arange(pl, device=DEV) / SR * 9)
    for t in kicks:
        i0 = int(t * SR)
        j = min(n, i0 + pl)
        pump[i0:j] = torch.minimum(pump[i0:j], shape[:j - i0])

    # ---- sound design
    for c in cues["open"]:
        sfx.add(thump(58), c["t"], 0.55)
        sfx.add(tick(2400), c["t"], 0.22, pan=0.0)
    for i, c in enumerate(cues["landscape"][:6]):
        sfx.add(tick(1800 + 160 * i), c["t"], 0.2, pan=-0.4 + 0.16 * i)
        sfx.add(thump(66), c["t"], 0.3)
    # risers into big moments + impacts on arrival
    for target, length in ((st("equation"), 2 * bar), (st("math"), bar), (st("rule"), 2 * bar)):
        sfx.add(riser(length), target - length, 0.22)
    for target, g in ((st("equation"), 0.75), (st("rule"), 0.8), (st("outro") + 0.0, 0.5),
                      (cues["outro"][0]["t"], 0.65)):
        sfx.add(impact(), target, g)
    # whoosh into each cut
    for s in tl["shots"][1:]:
        if s["id"] in ("equation", "rule"):
            continue
        sfx.add(whoosh(0.55), s["start"] - 0.42, 0.18)
    # UI ticks for slabs landing / cards / chips
    for i in range(6):
        sfx.add(tick(1300 + 120 * i), st("stack") + 0.25 + i * beat, 0.12, pan=0.2)
    for c in cues["differ"][1:]:
        sfx.add(tick(1600), c["t"] - 0.15, 0.14)
    # opening swell: low drone fading in
    dn = int(st("chaos") * SR)
    tt = torch.arange(dn, device=DEV) / SR
    drone = (torch.sin(2 * math.pi * 36.7 * tt) * 0.6 + torch.sin(2 * math.pi * 73.4 * tt) * 0.25) * \
        torch.clamp(tt / 3.0, max=1) * torch.clamp((st("chaos") - tt) / 1.0, max=1)
    sfx.add(drone, 0.0, 0.22)

    # ---- narration
    for s in tl["shots"]:
        for c in s["cues"][lang]:
            voice.add(resample_24_48(c["audio"]), c["t"], 1.0)
    voice.x = fft_filter(voice.x, hp(85, 2))

    # ---- mix
    music = pads.x * 1.0 + arp.x * 0.9 + sub.x
    music = music * pump
    music = music + drums.x * 0.6
    wet = reverb(pads.x * 0.35 + arp.x * 0.45 + drums.x * 0.08 + sfx.x * 0.35, rt=3.2)
    music = music + wet + sfx.x
    music = fft_filter(music, hp(32, 2))
    vwet = reverb(voice.x, rt=1.1, seed=9) * 0.05
    # voice-keyed ducking
    env = voice.x.abs().mean(0)
    w = int(0.18 * SR)
    ker = torch.ones(1, 1, w, device=DEV) / w
    env = torch.nn.functional.conv1d(env.view(1, 1, -1), ker, padding=w // 2)[0, 0, :n]
    env = env / (env.max() + 1e-9)
    duck = 1 - 0.42 * torch.clamp(env * 6, max=1)
    music = music * duck
    # level balance
    mrms = music.pow(2).mean().sqrt()
    vrms = voice.x[voice.x.abs() > 1e-4].pow(2).mean().sqrt() if (voice.x.abs() > 1e-4).any() else torch.tensor(1.0)
    music = music / (mrms + 1e-9) * vrms * 0.6
    mix = music + voice.x + vwet
    DEBUG["music"], DEBUG["voice"] = music, voice.x
    # end fade
    fade_n = int(2.0 * SR)
    end_i = int(total * SR)
    mix[:, end_i - fade_n:end_i] *= torch.linspace(1, 0, fade_n, device=DEV)
    mix[:, end_i:] = 0
    mix = torch.tanh(mix / (mix.abs().max() + 1e-9) * 1.2) * 0.9
    return mix[:, :int(total * SR)].T.contiguous().cpu().numpy()
