"""Shot designs for the promo. Every shot draws into a Frame at local time t."""
import math
import random

import torch

from gfx import (AMBER, BLUE, CORAL, CYAN, DEV, DIM, FAINT, GREEN, H, INK, PINK, TEAL, VIOLET,
                 W, WHITE, Frame, clamp, env, in_out, lerp, out_back, out_cubic, out_expo, ramp,
                 smooth, tex_sprite, text_sprite)

LAYERS = [  # (en, zh, color, tools)
    ("ALGORITHM", "算法与实验", AMBER, "your code · TRL · SkyRL · slime · verl"),
    ("DATA & RUNTIME", "数据与运行", CORAL, "verl · OpenRLHF · SkyRL · slime · AReaL"),
    ("TRAINING EXECUTION", "训练执行", BLUE, "FSDP2 · DeepSpeed · Megatron-Core"),
    ("INFERENCE GENERATION", "推理生成", TEAL, "vLLM · SGLang"),
    ("PARAMETERIZATION & KERNELS", "参数化与 kernel", VIOLET, "PEFT · LoRA · FlashAttention · Liger"),
    ("DATA QUALITY & OBSERVABILITY", "数据质量与观测", GREEN, "Math-Verify · lm-eval · W&B · Nsight"),
]
PICK = ["your code", "SkyRL", "FSDP2", "vLLM", "LoRA", "lm-eval"]

EQ = (r"\nabla_\theta J=\mathbb{E}_{(x,y)\sim %s}\Big[\sum_t %s\,"
      r"\nabla_\theta\log\pi_\theta(y_t\mid x,y_{<t})\Big]")
EQ_W = EQ % (r"\mu", r"\psi_t")
EQ_C = EQ % (r"{\color{cy}\mu}", r"{\color{am}\psi}_t")


def L(lang, en, zh):
    return zh if lang == "zh" else en


class Ctx:
    def __init__(self, shot, lang, beat):
        self.shot, self.lang, self.beat = shot, lang, beat
        self.dur = shot["dur"]
        self.cues = [dict(c, t=c["t"] - shot["start"]) for c in shot["cues"][lang]]

    def c(self, i):
        return self.cues[min(i, len(self.cues) - 1)]["t"]

    def ce(self, i):
        c = self.cues[min(i, len(self.cues) - 1)]
        return c["t"] + c["d"]

    def kw(self, i, word):
        """Approximate time a keyword is spoken inside cue i (by character position)."""
        c = self.cues[i]
        s = c["sub"]
        k = s.find(word)
        if k < 0:
            return c["t"]
        return c["t"] + c["d"] * (k + 0.3 * len(word)) / max(1, len(s))


# ------------------------------------------------------------------ helpers
def reveal(f, sp, x, y, p, anchor="lt", alpha=1.0, color=CYAN, scan=True):
    """Left-to-right wipe of a sprite with a glowing scan edge."""
    if p <= 0:
        return
    ox = {"l": 0, "c": sp.w / 2, "r": sp.w}[anchor[0]]
    oy = {"t": 0, "m": sp.h / 2, "b": sp.h}[anchor[1]]
    x0 = x - ox
    edge = x0 + sp.w * clamp(p)
    f.blit(sp, x, y, alpha, anchor, clip=(x0 - 2, edge))
    if scan and p < 1:
        a = math.sin(math.pi * clamp(p)) * alpha
        f.line((edge, y - oy - 14), (edge, y - oy + sp.h + 14), 2, color, a, glow=0.9, gr=14)


def rect_path(x, y, w, h):
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]


def draw_box(f, x, y, w, h, p, color, alpha=1.0, fill_alpha=0.55):
    """A panel whose border draws itself on (p: 0→1) and then gets a dark fill."""
    if p <= 0:
        return
    if p >= 1:
        f.rrect(x, y, w, h, 14, 1.2, color, alpha * 0.85, glow=0.18, gr=14,
                fill=(0.025, 0.035, 0.06), fill_alpha=fill_alpha)
    else:
        f.rrect(x, y, w, h, 14, 0, color, alpha * smooth(p), fill=(0.025, 0.035, 0.06),
                fill_alpha=fill_alpha * smooth(p))
        f.polyline(rect_path(x, y, w, h), out_cubic(p), width=1.4, color=color, alpha=alpha,
                   glow=0.5, gr=10)


def counter(v, fmt):
    return fmt.format(v)


def caption(f, text, lang, a):
    if a <= 0.01:
        return
    size = 34 if lang == "zh" else 32
    face = "reg"
    if lang == "zh":
        text = text.rstrip("。，；")
    sp = text_sprite(text, size, face, WHITE, 0.0, lang)
    y = H - 118
    shadow = text_sprite(text, size, face, (0, 0, 0), 0.0, lang)
    f.blit(shadow, W / 2, y, a * 0.9, "ct", blur=6)
    f.blit(sp, W / 2, y, a * 0.94, "ct")


# ------------------------------------------------------------------ background
class Backdrop:
    def __init__(self, seed=3):
        rnd = random.Random(seed)
        n = 520
        self.p = torch.tensor([[rnd.uniform(-1.6, 1.6), rnd.uniform(-1.0, 1.0), rnd.uniform(0.3, 4.0)]
                               for _ in range(n)], device=DEV)
        self.tw = torch.tensor([rnd.uniform(0, 6.28) for _ in range(n)], device=DEV)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, device=DEV), torch.linspace(-1, 1, W, device=DEV),
                                indexing="ij")
        r = torch.sqrt((xx * 0.9) ** 2 + (yy * 1.25) ** 2)
        base = torch.tensor(INK, device=DEV).view(3, 1, 1)
        center = torch.tensor((0.045, 0.07, 0.13), device=DEV).view(3, 1, 1)
        self.grad = base + (center - base) * torch.exp(-r * r * 1.6)

    def draw(self, f, t, dust=1.0, grid=0.0, speed=1.0):
        f.c += self.grad
        if dust > 0:
            z = torch.remainder(self.p[:, 2] - t * 0.05 * speed, 3.7) + 0.3
            x = self.p[:, 0] + 0.02 * torch.sin(t * 0.3 + self.tw)
            y = self.p[:, 1] + 0.015 * torch.cos(t * 0.23 + self.tw) - t * 0.004
            y = torch.remainder(y + 1.0, 2.0) - 1.0
            sx = W / 2 + x / z * W * 0.35
            sy = H / 2 + y / z * W * 0.35
            fade = torch.clamp(1.2 - z / 3.5, 0, 1) * torch.clamp((z - 0.3) * 3, 0, 1)
            inten = (0.10 + 0.06 * torch.sin(t * 1.7 + self.tw * 3)) * fade * dust
            f.dots(torch.stack([sx, sy], 1), inten, (0.7, 0.82, 1.0), size=1.3)
        if grid > 0:
            # receding floor grid
            hy = H * 0.62
            for i in range(-12, 13):
                x0 = W / 2 + i * 70
                x1 = W / 2 + i * 520
                f.line((x0, hy), (x1, H + 40), 1, (0.35, 0.5, 0.9), 0.07 * grid, glow=0)
            for k in range(14):
                z = ((k + (t * 0.5 * speed) % 1.0) / 14)
                yz = hy + (H + 40 - hy) * z ** 2.2
                f.line((0, yz), (W, yz), 1, (0.35, 0.5, 0.9), 0.07 * grid * z, glow=0)


def hud(f, t, idx, n, label, lang, a):
    if a <= 0.01:
        return
    m, ln = 54, 26
    c = (0.55, 0.65, 0.85)
    for (x, y, sx, sy) in ((m, m, 1, 1), (W - m, m, -1, 1), (m, H - m, 1, -1), (W - m, H - m, -1, -1)):
        f.line((x, y), (x + sx * ln, y), 1.2, c, 0.45 * a, glow=0)
        f.line((x, y), (x, y + sy * ln), 1.2, c, 0.45 * a, glow=0)
    f.text("POST-TRAINING SYSTEMS · 101", m + 40, m - 7, 15, "mono", DIM, 0.75 * a, tracking=0.18)
    f.text(f"{idx:02d} / {n:02d}", W - m - 40, m - 7, 15, "mono", AMBER, 0.85 * a, "rt", tracking=0.18)
    f.text(label, W - m - 140, m - 7, 15, "mono" if lang == "en" else "reg", DIM, 0.75 * a, "rt",
           tracking=0.18 if lang == "en" else 0.05, lang=lang)
    mm, ss = divmod(t, 60)
    f.text(f"T+{int(mm):02d}:{ss:05.2f}", m + 40, H - m - 12, 15, "mono", DIM, 0.6 * a, tracking=0.12)
    f.text("EN" if lang == "en" else "中文", W - m - 40, H - m - 12, 15, "mono" if lang == "en" else "reg",
           DIM, 0.6 * a, "rt", tracking=0.12, lang=lang)


# ================================================================== shots
OPEN_SUB = {
    "en": ["supervised fine-tuning", "direct preference optimization", "proximal policy optimization",
           "group relative policy optimization", "on-policy distillation"],
    "zh": ["监督微调", "直接偏好优化", "近端策略优化", "组相对策略优化", "在线蒸馏"],
}


def shot_open(f, t, cx):
    lang = cx.lang
    # horizon line of light
    grow = out_expo(ramp(t, 0.25, 1.6))
    half = 700 * grow
    a_end = 1 - smooth(ramp(t, cx.dur - 0.5, 0.45))
    f.line((W / 2 - half, 612), (W / 2 + half, 612), 1.2, CYAN, 0.8 * a_end, glow=0.8, gr=16)
    f.circle((W / 2, 612), 3, 0, WHITE, a_end * (1 - grow * 0.6), glow=2.5, gr=18, fill=True)
    for i, c in enumerate(cx.cues):
        word = c["sub"].rstrip(".。")
        t0 = c["t"] - 0.06
        t1 = cx.c(i + 1) - 0.06 if i + 1 < len(cx.cues) else cx.dur - 0.35
        if t < t0 - 0.05 or t > t1 + 0.2:
            continue
        p = ramp(t, t0, 0.28)
        a = smooth(p) * (1 - smooth(ramp(t, t1 - 0.04, 0.16)))
        last = i == len(cx.cues) - 1
        size = 150 if lang == "en" else 150
        face = "xlight"
        if lang == "zh" and not word.isascii():
            face = "light"
        f.text(word, W / 2, 560, size, face, WHITE, a, "cb", tracking=0.02, lang=lang,
               blur=(1 - p) * 10, scale=1.0 + 0.06 * (1 - out_cubic(p)), glow=0.25)
        sub = OPEN_SUB[lang][i]
        f.text(sub.upper() if lang == "en" else sub, W / 2, 648, 20 if lang == "en" else 24,
               "mono" if lang == "en" else "reg", AMBER, a * 0.9, "ct",
               tracking=0.28 if lang == "en" else 0.3, lang=lang)
        # tick marks for each method so far
    n = len(cx.cues)
    for i in range(n):
        on = t >= cx.c(i) - 0.06
        x = W / 2 + (i - (n - 1) / 2) * 34
        f.circle((x, 712), 3.2, 1.2, AMBER if on else FAINT, (1.0 if on else 0.6) * a_end * smooth(ramp(t, 0.8, 0.6)),
                 glow=0.6 if on else 0, gr=8, fill=on)


CHAOS_ITEMS = [  # (text, row, kind)
    ("GRPO", 0, "n"), ("PPO", 0, "n"), ("DPO", 0, "n"), ("GAE", 0, "n"), ("kl_coef", 0, "k"), ("clip_ratio", 0, "k"),
    ("verl", 1, "n"), ("OpenRLHF", 1, "n"), ("SkyRL", 1, "n"), ("slime", 1, "n"), ("Miles", 1, "n"), ("AReaL", 1, "n"),
    ("ROLL", 1, "n"), ("NeMo RL", 1, "n"), ("staleness", 1, "k"), ("rollout_n", 1, "k"),
    ("FSDP2", 2, "n"), ("DeepSpeed", 2, "n"), ("Megatron-Core", 2, "n"), ("ZeRO-3", 2, "n"), ("micro_batch", 2, "k"),
    ("vLLM", 3, "n"), ("SGLang", 3, "n"), ("KV cache", 3, "n"), ("top_p", 3, "k"), ("max_len", 3, "k"),
    ("LoRA", 4, "n"), ("QLoRA", 4, "n"), ("PEFT", 4, "n"), ("FlashAttention", 4, "n"), ("Liger", 4, "n"),
    ("Math-Verify", 5, "n"), ("lm-eval", 5, "n"), ("W&B", 5, "n"), ("Nsight", 5, "n"), ("DataTrove", 5, "n"),
]


class Chaos:
    def __init__(self):
        rnd = random.Random(11)
        self.items = []
        rows = {}
        for txt, row, kind in CHAOS_ITEMS:
            rows.setdefault(row, []).append(txt)
        sc, gap = 0.62, 34
        cells = [(i, j) for i in range(6) for j in range(6)]
        rnd.shuffle(cells)
        self.cells = cells
        ci = 0
        for row, names in rows.items():
            ws = []
            for txt in names:
                kind = next(k for t_, r_, k in CHAOS_ITEMS if t_ == txt)
                ws.append(text_sprite(txt, 44, "mono" if kind == "k" else "light", WHITE).w * sc)
            x = 1110 - (sum(ws) + gap * (len(ws) - 1)) / 2
            for txt, w in zip(names, ws):
                kind = next(k for t_, r_, k in CHAOS_ITEMS if t_ == txt)
                gi, gj = cells[ci % len(cells)]
                ci += 1
                z = rnd.uniform(1.0, 3.0)
                px = 300 + gi * 264 + rnd.uniform(-50, 50)
                py = 170 + gj * 110 + rnd.uniform(-25, 25)
                self.items.append(dict(txt=txt, row=row, kind=kind, x=(px - W / 2) * z / 620,
                                       y=(py - H / 2 + 40) * z / 620, z=z, ph=rnd.uniform(0, 6.28),
                                       ox=x + w / 2, oy=300 + row * 82))
                x += w + gap


_chaos = Chaos()


def shot_chaos(f, t, cx):
    morph = in_out(ramp(t, cx.c(2) + 0.2, 1.5))
    collapse = in_out(ramp(t, cx.dur - 0.9, 0.85))
    for it in _chaos.items:
        appear = cx.c(0) - 0.25 if it["kind"] == "n" else cx.c(1) - 0.2
        a = smooth(ramp(t, appear + (it["z"] - 0.9) * 0.15, 0.5))
        if a <= 0:
            continue
        z = it["z"] * (1 - 0.025 * t)
        sx = W / 2 + it["x"] / z * 620 + math.sin(t * 0.5 + it["ph"]) * 6
        sy = H / 2 - 40 + it["y"] / z * 620 + math.cos(t * 0.4 + it["ph"]) * 5
        scale = clamp(1.25 / z, 0.35, 1.4)
        blur = abs(z - 1.4) * 2.2
        dep = clamp(1.25 - z / 3.4, 0.25, 1)
        x = lerp(sx, it["ox"], morph)
        y = lerp(sy, it["oy"], morph)
        sc = lerp(scale, 0.62, morph)
        bl = lerp(blur, 0, morph)
        x = lerp(x, W / 2, collapse)
        y = lerp(y, H / 2, collapse)
        mono = it["kind"] == "k"
        col = LAYERS[it["row"]][2]
        base = AMBER if mono else WHITE
        size = 44
        face = "mono" if mono else "light"
        aa = a * lerp(dep, 0.95, morph) * (1 - collapse)
        f.text(it["txt"], x, y, size, face, base, aa * (1 - morph), "cm", scale=sc, blur=bl)
        if morph > 0:
            f.text(it["txt"], x, y, size, face, col, aa * morph, "cm", scale=sc, glow=0.15 * morph)
    if morph > 0:
        for r, (en, zh, col, _) in enumerate(LAYERS):
            y = 300 + r * 82
            a = morph * (1 - collapse)
            f.line((150, y + 30), (W - 150, y + 30), 1, col, 0.22 * a, glow=0)
            f.text(L(cx.lang, en, zh), 150, y, 16 if cx.lang == "en" else 20, "mono" if cx.lang == "en" else "reg", col, 0.9 * a,
                   "lm", tracking=0.2, lang=cx.lang)
    if collapse > 0:
        f.circle((W / 2, H / 2), 2 + 10 * collapse, 0, WHITE, collapse, glow=2.0 * collapse, gr=30, fill=True)


class EqInfo:
    def __init__(self, size):
        self.white = tex_sprite(EQ_W, size, WHITE)
        self.col = tex_sprite(EQ_C, size, WHITE)
        import numpy as np
        a = np.asarray(self.col.img).astype(int)
        m = a[..., 3] > 100
        mu = m & (a[..., 0] < 160) & (a[..., 2] > 180)
        ps = m & (a[..., 2] < 150) & (a[..., 0] > 200)

        def bb(mask):
            ys, xs = np.nonzero(mask)
            return xs.min(), ys.min(), xs.max(), ys.max()
        self.mu = bb(mu)
        self.psi = bb(ps)


_eq = {}


def eqinfo(size=70):
    if size not in _eq:
        _eq[size] = EqInfo(size)
    return _eq[size]


def draw_rings(f, cxy, t, a):
    for k, r in enumerate((260, 340, 430)):
        f.circle(cxy, r, 1, (0.4, 0.55, 0.9), 0.10 * a, glow=0)
        n = 60 + 24 * k
        for i in range(0, n, 3 + k):
            ang = i / n * 2 * math.pi + t * (0.05 if k % 2 == 0 else -0.04)
            p0 = (cxy[0] + r * math.cos(ang), cxy[1] + r * math.sin(ang))
            p1 = (cxy[0] + (r + 7) * math.cos(ang), cxy[1] + (r + 7) * math.sin(ang))
            f.line(p0, p1, 1, (0.45, 0.6, 0.95), 0.16 * a, glow=0)


def shot_equation(f, t, cx):
    lang = cx.lang
    eq = eqinfo(70)
    x0 = W / 2 - eq.white.w / 2
    y0 = 420 - eq.white.h / 2
    a_out = 1 - smooth(ramp(t, cx.dur - 0.45, 0.4))
    draw_rings(f, (W / 2, 420), t, smooth(ramp(t, 0.0, 1.0)) * a_out)
    p = in_out(ramp(t, cx.c(0) - 0.1, 1.5))
    f.blit(eq.white, x0, y0, 0.18 * smooth(ramp(t, 0, 0.6)) * (1 - p) * a_out, blur=8)
    reveal(f, eq.white, x0, y0, p, alpha=a_out)
    if p >= 1:
        f.glow_sprite(eq.white, x0, y0, 0.12 * a_out)
    for k, (bbx, col, ci, en, zh) in enumerate((
            (eq.mu, CYAN, 1, "where the data comes from", "数据从哪里来"),
            (eq.psi, AMBER, 2, "how each token is weighted", "每个 token 如何加权"))):
        hp = ramp(t, cx.c(ci) - 0.1, 0.6)
        if hp <= 0:
            continue
        a = smooth(hp) * a_out
        bx0, by0, bx1, by1 = bbx
        f.blit(eq.col, x0, y0, a, clip=(x0 + bx0 - 3, x0 + bx1 + 4))
        f.glow_sprite(eq.col, x0, y0, 0.0)
        cxm = x0 + (bx0 + bx1) / 2
        e = 8 + 10 * (1 - out_cubic(hp))
        f.rrect(x0 + bx0 - e, y0 + by0 - e, bx1 - bx0 + 2 * e, by1 - by0 + 2 * e, 8, 1.2, col, a * 0.9, glow=0.7, gr=12)
        ytop = y0 + by1 + 8
        yl = 640 + k * 0
        xl = W / 2 - 330 if k == 0 else W / 2 + 330
        lp = out_cubic(ramp(t, cx.c(ci), 0.7))
        f.polyline([(cxm, ytop), (cxm, yl - 40), (xl, yl - 40), (xl, yl - 10)], lp, width=1.2, color=col,
                   alpha=a * 0.8, glow=0.4, gr=8)
        sym = tex_sprite(r"\mu" if k == 0 else r"\psi_t", 64, col)
        ta = smooth(ramp(t, cx.c(ci) + 0.35, 0.5)) * a_out
        f.blit(sym, xl, yl + 6, ta, "ct")
        f.text(L(lang, en.upper(), zh), xl, yl + 90, 22 if lang == "en" else 28,
               "mono" if lang == "en" else "reg", col, ta, "ct", tracking=0.18 if lang == "en" else 0.1, lang=lang)


def shot_fork(f, t, cx):
    lang = cx.lang
    eq = eqinfo(70)
    a_out = 1 - smooth(ramp(t, cx.dur - 0.4, 0.38))
    # equation parks at the top
    f.blit(eq.white, W / 2, 150, 0.55 * a_out, "cm", scale=0.6)
    # divider
    g = out_expo(ramp(t, 0.1, 0.8))
    f.line((W / 2, 560 - 300 * g), (W / 2, 560 + 300 * g), 1, (0.5, 0.6, 0.9), 0.5 * a_out, glow=0.4, gr=10)
    # left: fixed data
    lp = ramp(t, cx.c(0) - 0.15, 0.6)
    focus_l = 1 - 0.55 * smooth(ramp(t, cx.c(1), 0.6))
    al = smooth(lp) * a_out * focus_l
    if lp > 0:
        cxl = W / 4 + 10
        if lang == "en":
            f.blit(tex_sprite(r"\mu\ \text{fixed}", 84, BLUE), cxl, 300, al, "cm")
        else:
            mu = tex_sprite(r"\mu", 84, BLUE)
            zt = text_sprite("固定", 64, "light", BLUE, 0.05, "zh")
            tw = mu.w + 24 + zt.w
            f.blit(mu, cxl - tw / 2, 300, al, "lm")
            f.blit(zt, cxl - tw / 2 + mu.w + 24, 296, al, "lm")
        # dataset rows + reading scan line
        for r in range(7):
            w = 360 - (r * 37) % 120
            f.rrect(cxl - 190, 400 + r * 34, w, 14, 4, 0, BLUE, al, fill=BLUE, fill_alpha=0.22)
        sy = 400 + ((t * 120) % 238)
        f.line((cxl - 210, sy), (cxl + 210, sy), 1.2, BLUE, al * 0.8, glow=0.8, gr=10)
        f.text(L(lang, "LOAD · LOSS · STEP", "读数据 · 算损失 · 更新"), cxl, 680, 18 if lang == "en" else 24,
               "mono" if lang == "en" else "reg", DIM, al, "ct", tracking=0.25 if lang == "en" else 0.1, lang=lang)
        f.text(L(lang, "TRAINER PROBLEM", "训练器问题"), cxl, 740, 36 if lang == "en" else 42, "light",
               BLUE, al, "ct", tracking=0.22 if lang == "en" else 0.15, lang=lang, glow=0.3)
    # right: the loop
    rp = ramp(t, cx.c(1) - 0.15, 0.6)
    ar = smooth(rp) * a_out
    if rp > 0:
        cxr = 3 * W / 4 - 10
        f.blit(tex_sprite(r"\mu=\pi_\theta", 84, AMBER), cxr, 300, ar, "cm")
        c = (cxr, 520)
        R = 110
        f.circle(c, R, 1.4, AMBER, ar * 0.5, glow=0.3, gr=10, a0=-math.pi / 2,
                 a1=-math.pi / 2 + 2 * math.pi * out_cubic(rp))
        for k, (en, zh) in enumerate((("generate", "生成"), ("reward", "奖励"), ("update", "更新"), ("sync", "同步"))):
            ang = -math.pi / 2 + k * math.pi / 2
            px, py = c[0] + R * math.cos(ang), c[1] + R * math.sin(ang)
            f.circle((px, py), 5, 0, AMBER, ar, glow=1.0, gr=10, fill=True)
            off = 40
            tx, ty = c[0] + (R + off) * math.cos(ang), c[1] + (R + off) * math.sin(ang)
            anc = "cm" if k in (0, 2) else ("lm" if k == 1 else "rm")
            f.text(L(lang, en.upper(), zh), tx, ty, 16 if lang == "en" else 22, "mono" if lang == "en" else "reg",
                   DIM, ar, anc, tracking=0.2 if lang == "en" else 0.05, lang=lang)
        # comet
        hd = t * 2.4
        pts = torch.tensor([[c[0] + R * math.cos(hd - i * 0.04), c[1] + R * math.sin(hd - i * 0.04)]
                            for i in range(30)], device=DEV)
        inten = torch.linspace(1.6, 0.0, 30, device=DEV) * ar
        f.dots(pts, inten, AMBER, size=2.2, glow=1.0)
        f.text(L(lang, "SYSTEMS PROBLEM", "系统问题"), cxr, 740, 36 if lang == "en" else 42, "light",
               AMBER, ar, "ct", tracking=0.22 if lang == "en" else 0.15, lang=lang, glow=0.3)


def slab(f, x, y, w, col, a, lift=0.0, label=None, lang="en", k=0, glow=1.0):
    dx, dy, th = 120, -46, 18
    y = y - lift
    top = [(x, y), (x + w, y), (x + w + dx, y + dy), (x + dx, y + dy)]
    front = [(x, y), (x + w, y), (x + w, y + th), (x, y + th)]
    side = [(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + dy + th), (x + w, y + th)]
    f.poly(top, tuple(c * 0.16 for c in col), a * 0.95)
    f.poly(front, tuple(c * 0.30 for c in col), a)
    f.poly(side, tuple(c * 0.20 for c in col), a)
    for p0, p1 in ((top[0], top[1]), (top[1], top[2]), (top[2], top[3]), (top[3], top[0])):
        f.line(p0, p1, 1.3, col, a * 0.9, glow=0.35 * glow, gr=10)
    f.line(front[2], front[3], 1, col, a * 0.5, glow=0)
    if label:
        f.text(f"{k + 1:02d}", x + 92, y + dy / 2, 16, "mono", col, a, "lm", tracking=0.1)
        f.text(label, x + 140, y + dy / 2, 19 if lang == "en" else 22, "mono" if lang == "en" else "reg",
               WHITE, a * 0.92, "lm", tracking=0.16 if lang == "en" else 0.08, lang=lang)


def shot_stack(f, t, cx):
    lang = cx.lang
    a_out = 1 - smooth(ramp(t, cx.dur - 0.4, 0.38))
    x, w = 260, 760
    pick = ramp(t, cx.c(1) + 0.3, 1.4)
    for i in range(6):
        k = 5 - i               # build from the bottom layer up
        tl = 0.25 + i * cx.beat  # lands on successive beats
        p = ramp(t, tl - 0.35, 0.35)
        if p <= 0:
            continue
        en, zh, col, tools = LAYERS[k]
        y = 300 + k * 96
        lift = (1 - out_cubic(p)) * 140
        a = smooth(p) * a_out
        land = math.exp(-max(0, t - tl) * 6) * (t >= tl)
        slab(f, x, y, w, col, a, lift, L(lang, en, zh), lang, k, glow=1 + 3 * land)
        # tool list to the right
        ta = smooth(ramp(t, tl + 0.1, 0.4)) * a_out
        tx = x + w + 170
        parts = tools.split(" · ")
        cxp = tx
        for j, part in enumerate(parts):
            sel = part == PICK[k]
            sp = text_sprite(part, 21, "mono", WHITE)
            hl = smooth(pick) if sel else 0
            alpha = ta * (0.75 - 0.4 * smooth(pick) * (not sel))
            f.blit(sp, cxp, y - 23, alpha * (1 - hl), "lm")
            if sel and hl > 0:
                f.blit(text_sprite(part, 21, "mono", col), cxp, y - 23, ta * hl, "lm")
                f.rrect(cxp - 8, y - 23 - 19, sp.w + 16, 38, 8, 1.2, col, ta * hl, glow=0.6, gr=10)
                PICKPOS[k] = (cxp + sp.w / 2 - 4, y - 23)
            cxp += sp.w + 18
    hp = smooth(ramp(t, cx.c(1) + 0.1, 0.6)) * a_out
    f.text(L(lang, "ONE COMPONENT PER ROLE", "每个角色 · 一个部件"), x, 200, 20 if lang == "en" else 26,
           "mono" if lang == "en" else "reg", AMBER, hp, "lm", tracking=0.3 if lang == "en" else 0.15, lang=lang)


PICKPOS = {}


def shot_math(f, t, cx):
    lang = cx.lang
    a_out = 1 - smooth(ramp(t, cx.dur - 0.4, 0.38))
    f.text(L(lang, "EVERY LAYER HAS ITS ARITHMETIC", "每一层都有自己的算术"), W / 2, 150, 20 if lang == "en" else 26,
           "mono" if lang == "en" else "reg", DIM, smooth(ramp(t, cx.c(0), 0.5)) * a_out, "ct",
           tracking=0.3 if lang == "en" else 0.15, lang=lang)
    # ---- LoRA panel
    px, py, pw, ph = 130, 230, 800, 620
    bp = ramp(t, cx.c(0) + 0.2, 0.9)
    draw_box(f, px, py, pw, ph, bp, AMBER, a_out)
    if bp > 0.5:
        a = smooth(ramp(t, cx.c(0) + 0.6, 0.5)) * a_out
        f.text("LoRA", px + 40, py + 36, 26, "mono", AMBER, a, tracking=0.2)
        # frozen W0 grid + trainable low-rank pair
        gx, gy, gs = px + 60, py + 110, 220
        n = 11
        for i in range(n + 1):
            f.line((gx + i * gs / n, gy), (gx + i * gs / n, gy + gs), 1, FAINT, a * 0.7, glow=0)
            f.line((gx, gy + i * gs / n), (gx + gs, gy + i * gs / n), 1, FAINT, a * 0.7, glow=0)
        f.text(L(lang, "W₀ frozen · d×k", "W0 冻结 · d×k"), gx + gs / 2, gy + gs + 20, 17,
               "mono" if lang == "en" else "reg", DIM, a, "ct", lang=lang)
        f.text("+", gx + gs + 40, gy + gs / 2, 40, "xlight", WHITE, a, "cm")
        lp = out_cubic(ramp(t, cx.c(1) - 0.2, 0.7))
        bx = gx + gs + 80
        f.rrect(bx, gy, 16, gs * lp, 3, 1.2, AMBER, a, glow=0.8, gr=12, fill=AMBER, fill_alpha=0.35)
        f.rrect(bx + 30, gy, 160 * lp, 16, 3, 1.2, AMBER, a, glow=0.8, gr=12, fill=AMBER, fill_alpha=0.35)
        f.text("B · d×r", bx, gy + gs + 20, 17, "mono", AMBER, a * lp, "lt")
        f.text("A · r×k", bx + 30, gy + 30, 17, "mono", AMBER, a * lp, "lt")
        f.blit(tex_sprite(r"h=W_0x+\tfrac{\alpha}{r}BAx", 40, WHITE), px + 60, py + ph - 70, a * 0.9, "lm")
        # counter
        cp = out_expo(ramp(t, cx.c(1) + 0.2, 1.4))
        if cp > 0:
            v = 0.39 * cp
            f.text(f"{v:.2f}%", px + pw - 50, py + ph - 40, 128, "thin", AMBER, a, "rb", glow=0.35)
            f.text(L(lang, "TRAINABLE PARAMETERS", "可训练参数占比"), px + pw - 50, py + ph - 250, 17 if lang == "en" else 22,
                   "mono" if lang == "en" else "reg", DIM, a * smooth(cp), "rt", tracking=0.2 if lang == "en" else 0.1,
                   lang=lang)
            f.text("r(d+k) / dk  ·  d = k = 4096, r = 8", px + pw - 50, py + ph - 214, 17, "mono", DIM, a * smooth(cp), "rt",
                   tracking=0.05)
    # ---- ZeRO panel
    qx = 990
    bq = ramp(t, cx.c(0) + 0.45, 0.9)
    draw_box(f, qx, py, pw, ph, bq, TEAL, a_out)
    if bq > 0.5:
        a = smooth(ramp(t, cx.c(0) + 0.8, 0.5)) * a_out
        f.text(L(lang, "MEMORY / GPU", "每卡显存"), qx + 40, py + 36, 26, "mono" if lang == "en" else "reg", TEAL, a,
               tracking=0.2 if lang == "en" else 0.1, lang=lang)
        rows = [("DDP", 120.0), ("ZeRO-1", 31.4), ("ZeRO-2", 16.6), ("ZeRO-3", 1.9)]
        maxw = 520
        for i, (nm, gb) in enumerate(rows):
            rp = out_cubic(ramp(t, cx.c(2) + i * cx.beat, 0.6))
            y = py + 140 + i * 74
            f.text(nm, qx + 40, y, 20, "mono", WHITE if i == 3 else DIM, a, "lm", tracking=0.08)
            bw = max(4, maxw * gb / 120) * rp
            col = TEAL if i == 3 else (0.35, 0.5, 0.7)
            f.rrect(qx + 160, y - 11, bw, 22, 4, 0, col, a, fill=col, fill_alpha=0.8 if i == 3 else 0.5)
            if i == 3:
                f.rrect(qx + 160, y - 11, bw, 22, 4, 1.2, TEAL, a * rp, glow=1.2, gr=12)
            if rp > 0.05:
                f.text(f"{gb * rp:.1f} GB", qx + 172 + bw, y, 18, "mono", WHITE if i == 3 else DIM, a * rp, "lm")
        cp = out_expo(ramp(t, cx.c(2) + 3 * cx.beat, 1.5))
        if cp > 0:
            v = 120 + (1.9 - 120) * cp
            f.text(f"{v:.1f}", qx + pw - 140, py + ph - 70, 110, "thin", TEAL, a, "rb", glow=0.35)
            f.text("GB", qx + pw - 50, py + ph - 84, 30, "light", TEAL, a, "rb")
            f.text("Ψ = 7.5B · K = 12 · N = 64", qx + 40, py + ph - 84, 17, "mono", DIM, a * smooth(cp), "lb",
                   tracking=0.1)


def shot_loop(f, t, cx):
    lang = cx.lang
    a_out = 1 - smooth(ramp(t, cx.dur - 0.4, 0.38))
    c = (560, 545)
    R = 230
    rp = out_cubic(ramp(t, 0.1, 1.2))
    f.circle(c, R, 1.3, (0.6, 0.7, 0.95), 0.5 * a_out, glow=0.2, gr=10, a0=-math.pi / 2,
             a1=-math.pi / 2 + 2 * math.pi * max(rp, 0.001))
    nodes = [  # (angle, en, zh, sub, col, keyword en, keyword zh)
        (-90, "GENERATE", "生成", "vLLM · SGLang", TEAL, "generate", "生成"),
        (0, "SCORE", "打分", "reward · verifier", AMBER, "score", "打分"),
        (90, "UPDATE", "更新", "FSDP2 · Megatron", BLUE, "update", "更新"),
        (180, "SYNC", "同步", "weights v → v+1", CORAL, "sync", "同步"),
    ]
    bottleneck = ramp(t, cx.c(1), 0.5)
    for k, (deg, en, zh, sub, col, kwe, kwz) in enumerate(nodes):
        ang = math.radians(deg)
        p = (c[0] + R * math.cos(ang), c[1] + R * math.sin(ang))
        ap = smooth(ramp(t, 0.25 + k * 0.25, 0.4)) * a_out
        tk = cx.kw(0, kwe if lang == "en" else kwz)
        lit = math.exp(-max(0, t - tk) * 2.5) * (t >= tk - 0.05)
        dimk = 1 - 0.55 * smooth(bottleneck) * (k != 3)
        f.circle(p, 44, 1.4, col, ap * dimk, glow=0.3 + 1.5 * lit, gr=12, fill=False)
        f.circle(p, 43, 0, (0.02, 0.03, 0.05), ap, glow=0, fill=True)
        f.circle(p, 6, 0, col, ap * dimk, glow=1 + 2 * lit, gr=10, fill=True)
        off = 72
        tx, ty = c[0] + (R + off) * math.cos(ang), c[1] + (R + off) * math.sin(ang)
        anc = {-90: "cb", 0: "lm", 90: "ct", 180: "rm"}[deg]
        dy = {-90: -8, 0: -14, 90: 0, 180: -14}[deg]
        f.text(L(lang, en, zh), tx, ty + dy, 22 if lang == "en" else 28, "mono" if lang == "en" else "reg", col,
               ap * dimk, anc, tracking=0.22 if lang == "en" else 0.1, lang=lang, glow=0.4 * lit)
        f.text(sub, tx, ty + dy + (30 if deg != -90 else -0) + (0 if deg != -90 else -34), 16, "mono", DIM,
               ap * dimk * 0.9, anc, tracking=0.05)
        if k == 3 and bottleneck > 0:
            pulse = 0.5 + 0.5 * math.sin((t - cx.c(1)) * 6)
            f.circle(p, 58 + 8 * pulse, 1.2, CORAL, smooth(bottleneck) * a_out * (0.5 + 0.5 * pulse), glow=0.8, gr=14)
    # comet around the ring
    if rp >= 1:
        hd = (t - 1.3) * 1.6 - math.pi / 2
        pts = torch.tensor([[c[0] + R * math.cos(hd - i * 0.02), c[1] + R * math.sin(hd - i * 0.02)]
                            for i in range(60)], device=DEV)
        inten = torch.linspace(1.8, 0.0, 60, device=DEV) * a_out
        f.dots(pts, inten, WHITE, size=2.0, glow=1.2)
    f.text(L(lang, "ONE RL STEP", "一个 RL 步"), c[0], c[1], 18 if lang == "en" else 24, "mono" if lang == "en" else "reg",
           DIM, 0.7 * a_out * smooth(ramp(t, 0.6, 0.5)), "cm", tracking=0.25, lang=lang)
    # ---- right: the weight-update phase
    bp = ramp(t, cx.c(1) + 0.2, 0.8)
    if bp > 0:
        x0, y0 = 1060, 330
        a = smooth(bp) * a_out
        f.text(L(lang, "WEIGHT-UPDATE PHASE", "权重更新阶段"), x0, y0, 20 if lang == "en" else 26,
               "mono" if lang == "en" else "reg", CORAL, a, "lm", tracking=0.25 if lang == "en" else 0.1, lang=lang)
        f.text("Miles · Kimi K2 1T-A32B · 32 + 32 H100 nodes", x0, y0 + 36, 16, "mono", DIM, a * 0.9, "lm",
               tracking=0.05)
        sh = in_out(ramp(t, cx.c(2) + 0.4, 2.2))
        full = 700
        v = 53.28 + (7.23 - 53.28) * sh
        bw = full * v / 53.28
        f.rrect(x0, y0 + 90, full, 40, 6, 1, (0.4, 0.45, 0.6), a * 0.5, glow=0)
        f.rrect(x0, y0 + 90, bw, 40, 6, 1.4, CORAL, a, glow=0.6, gr=12, fill=CORAL, fill_alpha=0.45)
        f.text(f"{v:5.2f}", x0, y0 + 270, 120, "thin", CORAL if sh < 1 else WHITE, a, "lb", glow=0.3)
        f.text(L(lang, "s / update", "秒 / 次"), x0 + 330, y0 + 255, 24, "light", DIM, a, "lb", lang=lang)
        la = smooth(ramp(t, cx.c(2) + 0.2, 0.4)) * a
        f.text(L(lang, "NCCL broadcast → P2P transfer", "NCCL 广播 → P2P 传输"), x0, y0 + 310, 20, "mono" if lang == "en" else "reg",
               DIM, la, "lm", tracking=0.05, lang=lang)
        ga = smooth(ramp(t, cx.c(2) + 2.8, 0.5)) * a
        f.text("≈ 7.4×", x0 + full, y0 + 255, 54, "light", WHITE, ga, "rb", glow=0.4)
        f.text(L(lang, "official, per phase — not end-to-end", "官方数据，仅该阶段，非端到端"), x0, y0 + 360, 16,
               "mono" if lang == "en" else "reg", FAINT, ga, "lm", tracking=0.05, lang=lang)


def shot_differ(f, t, cx):
    lang = cx.lang
    a_out = 1 - smooth(ramp(t, cx.dur - 0.4, 0.38))
    f.text(L(lang, "WHERE FRAMEWORKS REALLY DIFFER", "框架真正的差异"), W / 2, 170, 22 if lang == "en" else 30,
           "mono" if lang == "en" else "reg", DIM, smooth(ramp(t, cx.c(0), 0.5)) * a_out, "ct",
           tracking=0.3 if lang == "en" else 0.2, lang=lang)
    cards = [
        ("01", "ASYNCHRONY", "异步调度", "not a boolean · 5 mechanisms", "不是布尔开关 · 5 种机制", TEAL),
        ("02", "CONSISTENCY", "训推一致性", "5 sources of mismatch", "5 个误差来源", AMBER),
        ("03", "ALGORITHMIC FREEDOM", "算法自由度", "6 freedoms to own", "6 类可掌控的自由度", VIOLET),
    ]
    cw, ch, gap = 500, 470, 40
    x0 = (W - 3 * cw - 2 * gap) / 2
    y0 = 270
    ph_a = smooth(ramp(t, cx.c(0) + 0.2, 0.6)) * a_out
    for i in range(3):
        f.rrect(x0 + i * (cw + gap), y0, cw, ch, 14, 1, FAINT, ph_a * 0.5, glow=0)
    for i, (num, en, zh, sen, szh, col) in enumerate(cards):
        p = ramp(t, cx.c(i + 1) - 0.2, 0.7)
        if p <= 0:
            continue
        x = x0 + i * (cw + gap)
        draw_box(f, x, y0, cw, ch, p, col, a_out)
        a = smooth(ramp(t, cx.c(i + 1) + 0.1, 0.5)) * a_out
        f.text(num, x + 36, y0 + 34, 18, "mono", col, a, tracking=0.2)
        f.text(L(lang, en, zh), x + 36, y0 + 74, 30 if lang == "en" else 36, "light", WHITE, a, tracking=0.08 if lang == "en" else 0.05,
               lang=lang)
        f.text(L(lang, sen.upper(), szh), x + 36, y0 + ch - 50, 16 if lang == "en" else 22, "mono" if lang == "en" else "reg",
               col, a, tracking=0.15 if lang == "en" else 0.05, lang=lang)
        vx, vy = x + 36, y0 + 170
        if i == 0:
            # two lanes: generation runs ahead of training by a bounded lag
            for lane, (nm, off) in enumerate(((L(lang, "gen", "生成"), 0.0), (L(lang, "train", "训练"), 0.5))):
                yy = vy + lane * 70
                f.text(nm, vx, yy + 16, 16, "mono" if lang == "en" else "reg", DIM, a, "lm", lang=lang)
                for b in range(5):
                    bx = vx + 80 + ((b * 72 - (t * 50) + off * 72) % 360)
                    vcol = [TEAL, BLUE, VIOLET, AMBER, CORAL][b]
                    fade = clamp((bx - vx - 80) / 30) * clamp((vx + 364 - bx) / 30)
                    f.rrect(bx, yy, 64, 32, 5, 1, vcol, a * fade * 0.9, glow=0.3, gr=8, fill=vcol, fill_alpha=0.25)
            f.text(L(lang, "staleness ≤ k", "陈旧度 ≤ k"), vx, vy + 190, 20, "mono" if lang == "en" else "reg", WHITE,
                   a * 0.85, "lm", lang=lang)
        elif i == 1:
            f.blit(tex_sprite(r"\log\mu=\log\pi-\log Z_S", 38, WHITE), vx, vy + 30, a, "lm")
            f.text(L(lang, "top-p: renormalize on the sampled support", "top-p：在实际采样支持集上归一化"), vx, vy + 110, 17,
                   "mono" if lang == "en" else "reg", DIM, a, "lm", lang=lang)
            f.text(L(lang, "tokens · versions · routes · kernels", "token · 版本 · 路由 · kernel"), vx, vy + 150, 17,
                   "mono" if lang == "en" else "reg", DIM, a, "lm", lang=lang)
        else:
            names = [("objective", "目标函数"), ("estimator", "估计器"), ("data lifetime", "数据寿命"),
                     ("model state", "模型状态"), ("update rule", "更新规则"), ("execution", "执行调度")]
            for j, (ne, nz) in enumerate(names):
                yy = vy - 10 + j * 36
                fp = out_cubic(ramp(t, cx.c(3) + 0.2 + j * 0.12, 0.5))
                f.text(f"{j + 1}", vx, yy, 16, "mono", col, a, "lm")
                f.text(L(lang, ne, nz), vx + 30, yy, 18, "mono" if lang == "en" else "reg", WHITE, a * 0.85, "lm", lang=lang)
                f.rrect(vx + 230, yy - 5, 190 * fp, 10, 3, 0, col, a, fill=col, fill_alpha=0.6)


ATOMS = [("FSDP2", BLUE), ("ZeRO", BLUE), ("Megatron", BLUE), ("vLLM", TEAL), ("SGLang", TEAL), ("Ray", CORAL)]
RECIPES = [("OpenRLHF", ["Ray", "ZeRO", "vLLM"]), ("slime", ["Ray", "Megatron", "SGLang"]),
           ("SkyRL", ["Ray", "FSDP2", "vLLM"]), ("verl", ["Ray", "FSDP2", "Megatron", "vLLM", "SGLang"]),
           ("NeMo RL", ["Ray", "Megatron", "vLLM", "SGLang"])]
ADDR = [  # (number, tex, color, address)
    ("01", r"x\sim\mathcal D", CYAN, "custom_async_data_buffer_path"),
    ("06", r"w_t=\pi_{\mathrm{old}}/\mu", CYAN, "custom TIS"),
    ("07", r"\hat A_t", AMBER, "register_adv_est"),
    ("09", r"\pi_\theta(\cdot\mid h_t)", AMBER, "custom_loss(logits)"),
    ("12", r"\theta\leftarrow\mathrm{Opt}", PINK, "your learner"),
]
DEPTH = [("L0", "CONFIG", "配置", TEAL), ("L1", "HOOK", "钩子", GREEN), ("L2", "OVERRIDE", "覆写", AMBER),
         ("L3", "FORK", "改源码", CORAL), ("L4", "CROSS BOUNDARIES", "跨边界", PINK)]


def shot_anatomy(f, t, cx):
    lang = cx.lang
    a_out = 1 - smooth(ramp(t, cx.dur - 0.4, 0.38))
    mono = "mono" if lang == "en" else "reg"
    # phase envelopes: recipes → addresses → depth
    a1 = smooth(ramp(t, cx.c(0) - 0.2, 0.5)) * (1 - smooth(ramp(t, cx.c(1) - 0.35, 0.35)))
    a2 = smooth(ramp(t, cx.c(1) - 0.1, 0.5)) * a_out
    a3 = smooth(ramp(t, cx.c(2) - 0.1, 0.5)) * a_out
    f.text(L(lang, "ANATOMY", "拆解"), W / 2, 150, 22 if lang == "en" else 30, mono, DIM,
           smooth(ramp(t, 0.0, 0.5)) * a_out, "ct", tracking=0.3 if lang == "en" else 0.2, lang=lang)
    # ---- phase 1: atoms → recipes
    if a1 > 0.01:
        aw, ag = 190, 22
        ax0 = (W - len(ATOMS) * aw - (len(ATOMS) - 1) * ag) / 2
        apos = {}
        for i, (nm, col) in enumerate(ATOMS):
            x = ax0 + i * (aw + ag)
            p = smooth(ramp(t, cx.c(0) + i * 0.08, 0.4)) * a1
            f.rrect(x, 230, aw, 52, 10, 1.2, col, p, glow=0.5, gr=10, fill=col, fill_alpha=0.12)
            f.text(nm, x + aw / 2, 256, 21, "mono", col, p, "cm", tracking=0.06)
            apos[nm] = (x + aw / 2, 282)
        for r, (fw, parts) in enumerate(RECIPES):
            y = 400 + r * 92
            t0 = cx.c(0) + 0.7 + r * 0.32
            p = smooth(ramp(t, t0, 0.45)) * a1
            if p <= 0:
                continue
            f.text(fw, 420, y, 34, "light", WHITE, p, "rm")
            for j, nm in enumerate(parts):
                col = dict(ATOMS)[nm]
                q = out_cubic(ramp(t, t0 + 0.07 * j, 0.45))
                x = 480 + j * 168
                yy = y - 22 - (1 - q) * 30
                f.rrect(x, yy, 152, 44, 10, 1.1, col, p * q, glow=0.4, gr=8, fill=col, fill_alpha=0.10)
                f.text(nm, x + 76, yy + 22, 18, "mono", col, p * q, "cm")
                if q < 1:  # a faint trace back to the atom it came from
                    sx, sy = apos[nm]
                    f.line((sx, sy), (x + 76, yy), 1, col, p * 0.35 * (1 - q), glow=0.3)
            f.text(L(lang, "+ its own connective layer", "+ 自己的连接层"), 480 + len(parts) * 168 + 14, y, 18, mono, AMBER,
                   p * smooth(ramp(t, t0 + 0.4, 0.4)) * 0.85, "lm", lang=lang)
    # ---- phase 2: every term has an address
    if a2 > 0.01:
        n = len(ADDR)
        colw = 345
        x0 = (W - n * colw) / 2 + colw / 2
        for i, (num, tx, col, addr) in enumerate(ADDR):
            x = x0 + i * colw
            t0 = cx.c(1) + 0.1 + i * 0.22
            p = smooth(ramp(t, t0, 0.45)) * a2
            if p <= 0:
                continue
            f.text(num, x, 300, 18, "mono", col, p, "cm", tracking=0.2)
            f.blit(tex_sprite(tx, 46, col), x, 380, p, "cm")
            lp = out_cubic(ramp(t, t0 + 0.25, 0.5))
            f.line((x, 430), (x, 430 + 150 * lp), 1.2, col, p * 0.7, glow=0.6, gr=10)
            if lp > 0.9:
                f.circle((x, 584), 4, 0, col, p, glow=2, gr=10, fill=True)
            q = smooth(ramp(t, t0 + 0.6, 0.4)) * p
            f.rrect(x - 162, 606, 324, 52, 10, 1.1, col, q * 0.9, glow=0.3, gr=8, fill=(0.025, 0.035, 0.06), fill_alpha=0.7)
            f.text(addr, x, 632, 17, "mono", WHITE, q, "cm")
        f.text(L(lang, "DATA RULE", "数据规则"), x0 + 0.5 * colw, 730, 18, mono, CYAN, a2 * (1 - a3) * 0.9, "cm",
               tracking=0.25 if lang == "en" else 0.1, lang=lang)
        f.text(L(lang, "LEARNING RULE", "学习规则"), x0 + 3.0 * colw, 730, 18, mono, AMBER, a2 * (1 - a3) * 0.9, "cm",
               tracking=0.25 if lang == "en" else 0.1, lang=lang)
    # ---- phase 3: how deep must you dig
    if a3 > 0.01:
        for i, (k, en, zh, col) in enumerate(DEPTH):
            y = 694 + i * 46
            bw = 300 + 110 * i
            bx = W / 2 - bw / 2
            p = smooth(ramp(t, cx.c(2) + 0.1 + i * 0.12, 0.35)) * a3
            hot = math.exp(-((t - (cx.c(2) + 0.5 + i * 0.25)) * 3.0) ** 2)
            f.rrect(bx, y, bw, 36, 8, 1.1, col, p, glow=0.4 + hot, gr=10, fill=col, fill_alpha=0.10 + 0.25 * hot)
            f.text(k, bx + 16, y + 18, 18, "mono", col, p, "lm")
            f.text(L(lang, en, zh), W / 2, y + 18, 17, mono, WHITE, p * 0.9, "cm", tracking=0.18 if lang == "en" else 0.05, lang=lang)


LAND = [
    ("verl", "FSDP · Megatron · vLLM/SGLang"), ("OpenRLHF", "Ray · DeepSpeed · vLLM"), ("slime", "Megatron · SGLang"),
    ("SkyRL", "FSDP2/Megatron · vLLM"), ("AReaL", "async-native · FSDP2 · Megatron"), ("NeMo RL", "Megatron-Core · DTensor"),
    ("Miles", "slime fork"), ("ROLL", "Ray roles · multi-domain"), ("RLinf", "embodied / VLA"), ("Prime-RL", "FSDP2 · vLLM · async"),
]


def shot_landscape(f, t, cx):
    lang = cx.lang
    a_out = 1 - smooth(ramp(t, cx.dur - 0.4, 0.38))
    c = (W / 2, 520)
    rx, ry = 640, 270
    order = [0, 2, 4, 6, 8, 1, 3, 5, 7, 9]  # spread the spoken six around the ellipse
    slots = {}
    for j, idx in enumerate(order):
        ang = -math.pi / 2 + j * 2 * math.pi / 10 + t * 0.02
        slots[idx] = (c[0] + rx * math.cos(ang), c[1] + ry * math.sin(ang))
    hub = smooth(ramp(t, 0.0, 0.6)) * a_out
    f.circle(c, 70, 1.2, AMBER, hub * 0.6, glow=0.5, gr=14)
    f.blit(tex_sprite(r"\nabla J", 50, AMBER), c[0], c[1], hub, "cm")
    final = ramp(t, cx.c(6) - 0.2, 0.8)
    for idx, (nm, tag) in enumerate(LAND):
        spoken = idx < 6
        t0 = cx.c(idx) - 0.08 if spoken else cx.c(6) + (idx - 6) * 0.18
        p = ramp(t, t0, 0.35)
        if p <= 0:
            continue
        x, y = slots[idx]
        a = smooth(p) * a_out * (1 if spoken else 0.7)
        lp = out_cubic(ramp(t, t0, 0.5))
        ux, uy = x - c[0], y - c[1]
        un = math.hypot(ux, uy)
        s0 = (c[0] + ux / un * 76, c[1] + uy / un * 76)
        f.line(s0, (lerp(s0[0], x, lp), lerp(s0[1], y, lp)), 1, AMBER if spoken else FAINT, a * 0.45, glow=0.3, gr=8)
        flash = math.exp(-max(0, t - t0) * 5)
        f.circle((x, y), 5, 0, WHITE, a, glow=1 + 3 * flash, gr=12, fill=True)
        cur = spoken and (idx == 5 or t < cx.c(idx + 1) - 0.08) and final <= 0
        size = 40 if spoken else 30
        f.text(nm, x, y - 18, size, "light" if not cur else "reg", WHITE, a, "cb", glow=0.4 if cur else 0.1,
               scale=1 + 0.08 * flash)
        f.text(tag, x, y + 20, 15, "mono", DIM, a * (0.9 if (cur or final > 0) else 0.6), "ct", tracking=0.05)
    if final > 0:
        pts = [slots[i] for i in order] + [slots[order[0]]]
        f.polyline(pts, out_cubic(final), width=1, color=(0.5, 0.6, 0.9), alpha=0.35 * a_out, glow=0.2, gr=8)


def shot_rule(f, t, cx):
    lang = cx.lang
    a_out = 1 - smooth(ramp(t, cx.dur - 0.35, 0.33))
    lines = [(L(lang, "Own the math.", "掌握数学"), AMBER, 0), (L(lang, "Reuse the runtime.", "复用运行时"), TEAL, 1)]
    for txt, col, i in lines:
        p = ramp(t, cx.c(i) - 0.1, 0.55)
        if p <= 0:
            continue
        face = "xlight" if lang == "en" else "zhserif"
        sp = text_sprite(txt, 110 if lang == "en" else 116, face, col, 0.01, lang)
        y = 470 + i * 160
        f.blit(sp, W / 2, y, smooth(p) * a_out, "cm", blur=(1 - out_cubic(p)) * 8)
        f.glow_sprite(sp, W / 2, y, 0.25 * smooth(p) * a_out, "cm")
        lw = out_expo(ramp(t, cx.c(i), 0.8)) * (sp.w / 2 + 40)
        f.line((W / 2 - lw, y + 78), (W / 2 + lw, y + 78), 1, col, 0.5 * a_out, glow=0.6, gr=10)


def shot_outro(f, t, cx):
    lang = cx.lang
    a_end = 1 - smooth(ramp(t, cx.dur - 2.2, 2.0))
    eq = eqinfo(70)
    f.blit(eq.white, W / 2, 300, 0.22 * smooth(ramp(t, 0.2, 1.5)) * a_end, "cm", scale=0.7, blur=1)
    p = ramp(t, cx.c(0) - 0.15, 0.9)
    if lang == "en":
        t1 = text_sprite("POST-TRAINING SYSTEMS", 92, "xlight", WHITE, 0.14)
        t2 = text_sprite("Layer by Layer", 50, "light", AMBER, 0.04)
    else:
        t1 = text_sprite("后训练工具体系", 116, "zhserif", WHITE, 0.12, "zh")
        t2 = text_sprite("分层拆解", 54, "zhserif", AMBER, 0.3, "zh")
    f.blit(t1, W / 2, 500, smooth(p) * a_end, "cm", blur=(1 - out_cubic(p)) * 10,
           scale=1 + 0.03 * (1 - out_cubic(p)))
    f.glow_sprite(t1, W / 2, 500, 0.2 * smooth(p) * a_end, "cm")
    p2 = ramp(t, cx.c(0) + 0.8, 0.8)
    f.blit(t2, W / 2, 610, smooth(p2) * a_end, "cm", blur=(1 - out_cubic(p2)) * 6)
    lw = out_expo(ramp(t, cx.c(0) + 0.4, 1.2)) * 420
    f.line((W / 2 - lw, 676), (W / 2 + lw, 676), 1, AMBER, 0.6 * a_end, glow=0.7, gr=12)
    p3 = smooth(ramp(t, cx.c(1) - 0.1, 0.6)) * a_end
    f.text(L(lang, "FULL DEEP DIVE  ·  ~32 MIN  ·  ENGLISH + CHINESE", "完整版深度解读  ·  约 32 分钟  ·  中文 / ENGLISH"),
           W / 2, 720, 20 if lang == "en" else 24, "mono" if lang == "en" else "reg", DIM, p3, "ct",
           tracking=0.25 if lang == "en" else 0.12, lang=lang)
    p4 = smooth(ramp(t, cx.c(1) + 0.9, 0.6)) * a_end
    f.text("haokun12x.github.io/post-training-101", W / 2, 790, 26, "mono", AMBER, p4, "ct", tracking=0.08, glow=0.3)


SHOT_FN = dict(open=shot_open, chaos=shot_chaos, equation=shot_equation, fork=shot_fork, stack=shot_stack,
               math=shot_math, loop=shot_loop, differ=shot_differ, anatomy=shot_anatomy, landscape=shot_landscape, rule=shot_rule,
               outro=shot_outro)
CAPTIONED = {"chaos", "equation", "fork", "stack", "math", "loop", "differ", "anatomy"}
