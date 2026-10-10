"""A small GPU motion-graphics engine (torch).

Frame = linear RGB float tensor on the GPU. Shapes are drawn as signed-distance
fields (antialiased core + exponential glow), text and LaTeX as cached sprites.
A bloom pass, vignette, grain and a soft shoulder finish every frame.
"""
import functools
import hashlib
import math
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(ROOT, "build", "tex")
os.makedirs(CACHE, exist_ok=True)

W, H = 1920, 1080
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)

# palette (0..1 floats)
def rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


INK = rgb("#04060c")
WHITE = rgb("#eef2fb")
DIM = rgb("#8a96b2")
FAINT = rgb("#3a4560")
AMBER = rgb("#ffb547")
CYAN = rgb("#4fd8ff")
TEAL = rgb("#3fe0b8")
VIOLET = rgb("#a98bff")
CORAL = rgb("#ff6f73")
BLUE = rgb("#5f9dff")
GREEN = rgb("#7fe28a")
PINK = rgb("#ff7cc8")

# ---------------------------------------------------------------- easing
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ramp(t, t0, d):
    """0 → 1 linearly over [t0, t0 + d]."""
    return clamp((t - t0) / d) if d > 0 else float(t >= t0)


def smooth(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def out_cubic(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def out_expo(x):
    x = clamp(x)
    return 1.0 if x >= 1 else 1 - 2 ** (-10 * x)


def in_out(x):
    x = clamp(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def out_back(x, s=1.4):
    x = clamp(x)
    return 1 + (s + 1) * (x - 1) ** 3 + s * (x - 1) ** 2


def lerp(a, b, x):
    return a + (b - a) * x


def env(t, t0, t1, fi=0.35, fo=0.35):
    """Fade in at t0, out at t1."""
    return smooth(ramp(t, t0, fi)) * (1 - smooth(ramp(t, t1 - fo, fo)))


# ---------------------------------------------------------------- fonts
FONTS = {
    "thin": "/usr/share/fonts/opentype/inter/InterDisplay-Thin.otf",
    "xlight": "/usr/share/fonts/opentype/inter/InterDisplay-ExtraLight.otf",
    "light": "/usr/share/fonts/opentype/inter/InterDisplay-Light.otf",
    "reg": "/usr/share/fonts/opentype/inter/InterDisplay-Regular.otf",
    "med": "/usr/share/fonts/opentype/inter/InterDisplay-Medium.otf",
    "semi": "/usr/share/fonts/opentype/inter/InterDisplay-SemiBold.otf",
    "bold": "/usr/share/fonts/opentype/inter/InterDisplay-Bold.otf",
    "mono": "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
    "zh": ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 2),
    "zhb": ("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 2),
    "zhserif": ("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc", 2),
    "zhserifb": ("/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc", 2),
}


@functools.lru_cache(None)
def pil_font(face, size):
    f = FONTS[face]
    if isinstance(f, tuple):
        return ImageFont.truetype(f[0], size, index=f[1])
    return ImageFont.truetype(f, size)


def zh_face(face):
    """Map a latin weight to the matching CJK face."""
    return {"thin": "zh", "xlight": "zh", "light": "zh", "reg": "zh", "med": "zh",
            "semi": "zhb", "bold": "zhb", "mono": "mono"}.get(face, face)


# ---------------------------------------------------------------- sprites
class Sprite:
    """Premultiplied RGBA on the GPU plus its anchor-independent size."""

    def __init__(self, arr):
        t = torch.from_numpy(np.asarray(arr, np.float32) / 255.0).to(DEV).permute(2, 0, 1)
        a = t[3:4]
        self.t = torch.cat([t[:3] * a, a], 0)
        self.h, self.w = self.t.shape[1:]
        self._blur = {}

    def blurred(self, s):
        s = round(s * 2) / 2
        if s <= 0.25:
            return self.t
        if s not in self._blur:
            pad = int(3 * s) + 1
            x = F.pad(self.t, (pad, pad, pad, pad))
            self._blur[s] = (gblur(x[None], s)[0], pad)
        return self._blur[s]


_sprites = {}


def text_sprite(text, size, face="light", color=WHITE, tracking=0.0, lang="en"):
    """Single-line text as a sprite. tracking = extra letter spacing in em."""
    if lang == "zh":
        face = zh_face(face)
    key = ("t", text, size, face, color, tracking)
    if key in _sprites:
        return _sprites[key]
    f = pil_font(face, size)
    col = tuple(int(c * 255) for c in color) + (255,)
    asc, desc = f.getmetrics()
    if tracking:
        widths = [f.getlength(ch) for ch in text]
        total = sum(widths) + tracking * size * (len(text) - 1)
    else:
        total = f.getlength(text)
    pad = 4
    img = Image.new("RGBA", (int(math.ceil(total)) + 2 * pad, asc + desc + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if tracking:
        x = pad
        for ch, w in zip(text, widths):
            d.text((x, pad), ch, font=f, fill=col)
            x += w + tracking * size
    else:
        d.text((pad, pad), text, font=f, fill=col)
    s = Sprite(img)
    s.asc = asc + pad
    _sprites[key] = s
    return s


TEX_PRE = r"""\documentclass[12pt]{article}
\usepackage{amsmath,amssymb,bm}
\usepackage[dvipsnames]{xcolor}
\pagestyle{empty}
\begin{document}
\definecolor{fgc}{RGB}{%d,%d,%d}\color{fgc}
\definecolor{cy}{RGB}{79,216,255}\definecolor{am}{RGB}{255,181,71}
\mbox{$\displaystyle %s$}
\end{document}
"""


def tex_sprite(tex, size, color=WHITE):
    key = ("x", tex, size, color)
    if key in _sprites:
        return _sprites[key]
    c255 = tuple(int(c * 255) for c in color)
    dpi = int(size * 72.27 / 12)
    h = hashlib.md5(f"{tex}|{dpi}|{c255}".encode()).hexdigest()
    path = os.path.join(CACHE, h + ".png")
    if not os.path.exists(path):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "m.tex"), "w") as fh:
                fh.write(TEX_PRE % (c255 + (tex,)))
            r = subprocess.run(["latex", "-interaction=nonstopmode", "m.tex"], cwd=td,
                               capture_output=True, text=True)
            if r.returncode:
                raise RuntimeError(tex + "\n" + r.stdout[-1200:])
            fg = "rgb %.4f %.4f %.4f" % tuple(color)
            subprocess.run(["dvipng", "-T", "tight", "-D", str(dpi), "-bg", "Transparent",
                            "-fg", fg, "-o", path, "m.dvi"], cwd=td, capture_output=True, check=True)
    img = Image.open(path).convert("RGBA")
    s = Sprite(img)
    s.img = img
    _sprites[key] = s
    return s


# ---------------------------------------------------------------- blur
@functools.lru_cache(None)
def _gk(s):
    r = int(3 * s) + 1
    x = torch.arange(-r, r + 1, device=DEV, dtype=torch.float32)
    k = torch.exp(-x * x / (2 * s * s))
    return k / k.sum(), r


def gblur(x, s):
    """Separable gaussian blur of a (N, C, H, W) tensor."""
    if s <= 0:
        return x
    k, r = _gk(float(s))
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)
    return x


# ---------------------------------------------------------------- frame
_YY, _XX = torch.meshgrid(torch.arange(H, device=DEV, dtype=torch.float32) + 0.5,
                          torch.arange(W, device=DEV, dtype=torch.float32) + 0.5, indexing="ij")


def _col(c):
    return torch.tensor(c, device=DEV, dtype=torch.float32).view(3, 1, 1)


class Frame:
    def __init__(self, base=None):
        self.c = base.clone() if base is not None else torch.zeros(3, H, W, device=DEV)
        self.glow = torch.zeros(3, H, W, device=DEV)   # extra light fed to the bloom only

    # -- region helper -------------------------------------------------
    @staticmethod
    def _box(x0, y0, x1, y1):
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(W, int(math.ceil(x1))), min(H, int(math.ceil(y1)))
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    def _emit(self, box, d, width, color, alpha, glow, gr, fill=False):
        """Composite an SDF d (pixels) into the region. width = full stroke width."""
        x0, y0, x1, y1 = box
        if fill:
            core = (0.5 - d).clamp(0, 1)
        else:
            core = (width / 2 + 0.5 - d.abs()).clamp(0, 1)
        col = _col(color)
        reg = self.c[:, y0:y1, x0:x1]
        a = core * alpha
        reg.mul_(1 - a * 0.0).add_(col * a)    # additive light core
        if glow > 0:
            dd = (d if fill else d.abs() - width / 2).clamp(min=0)
            g = torch.exp(-dd / gr) * glow * alpha
            reg.add_(col * g)

    # -- primitives ----------------------------------------------------
    def line(self, p0, p1, width=2.0, color=WHITE, alpha=1.0, glow=0.35, gr=10.0):
        if alpha <= 0.003:
            return
        m = width + 4 * gr * (glow > 0) + 2
        box = self._box(min(p0[0], p1[0]) - m, min(p0[1], p1[1]) - m,
                        max(p0[0], p1[0]) + m, max(p0[1], p1[1]) + m)
        if not box:
            return
        x0, y0, x1, y1 = box
        X, Y = _XX[y0:y1, x0:x1], _YY[y0:y1, x0:x1]
        ax, ay = p0
        bx, by = p1
        dx, dy = bx - ax, by - ay
        l2 = dx * dx + dy * dy + 1e-9
        h = (((X - ax) * dx + (Y - ay) * dy) / l2).clamp(0, 1)
        d = torch.sqrt((X - ax - h * dx) ** 2 + (Y - ay - h * dy) ** 2)
        self._emit(box, d, width, color, alpha, glow, gr)

    def polyline(self, pts, progress=1.0, **kw):
        """Draw the first `progress` fraction (by length) of a polyline."""
        if progress <= 0 or len(pts) < 2:
            return
        segs = list(zip(pts[:-1], pts[1:]))
        lens = [math.dist(a, b) for a, b in segs]
        target = sum(lens) * clamp(progress)
        for (a, b), L in zip(segs, lens):
            if target <= 0:
                break
            if L > target:
                f = target / L
                b = (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
            self.line(a, b, **kw)
            target -= L

    def circle(self, c, r, width=2.0, color=WHITE, alpha=1.0, glow=0.35, gr=10.0, fill=False,
               a0=None, a1=None):
        """Ring (or disc if fill). Optional arc from angle a0 to a1 (radians, clockwise from +x)."""
        if alpha <= 0.003:
            return
        m = r + width + 4 * gr + 2
        box = self._box(c[0] - m, c[1] - m, c[0] + m, c[1] + m)
        if not box:
            return
        x0, y0, x1, y1 = box
        X, Y = _XX[y0:y1, x0:x1] - c[0], _YY[y0:y1, x0:x1] - c[1]
        rr = torch.sqrt(X * X + Y * Y)
        d = rr - r
        if a0 is not None:
            ang = torch.atan2(Y, X)
            mid, half = (a0 + a1) / 2, (a1 - a0) / 2
            da = torch.remainder(ang - mid + math.pi, 2 * math.pi) - math.pi
            # distance to the arc endpoints outside the angular range
            out = da.abs() > half
            e0 = (c[0] + r * math.cos(a0), c[1] + r * math.sin(a0))
            e1 = (c[0] + r * math.cos(a1), c[1] + r * math.sin(a1))
            Xa, Ya = X + c[0], Y + c[1]
            de = torch.minimum(torch.sqrt((Xa - e0[0]) ** 2 + (Ya - e0[1]) ** 2),
                               torch.sqrt((Xa - e1[0]) ** 2 + (Ya - e1[1]) ** 2))
            d = torch.where(out, de, d.abs())
            self._emit(box, d, width, color, alpha, glow, gr)
            return
        self._emit(box, d, width, color, alpha, glow, gr, fill=fill)

    def rrect(self, x, y, w, h, r=12, width=1.5, color=WHITE, alpha=1.0, glow=0.25, gr=12.0,
              fill=None, fill_alpha=0.0):
        """Rounded rectangle: optional translucent fill (alpha-over) and glowing stroke."""
        if alpha <= 0.003:
            return
        m = width + 4 * gr + 2
        box = self._box(x - m, y - m, x + w + m, y + h + m)
        if not box:
            return
        x0, y0, x1, y1 = box
        cx, cy = x + w / 2, y + h / 2
        qx = (_XX[y0:y1, x0:x1] - cx).abs() - (w / 2 - r)
        qy = (_YY[y0:y1, x0:x1] - cy).abs() - (h / 2 - r)
        d = torch.sqrt(qx.clamp(min=0) ** 2 + qy.clamp(min=0) ** 2) + torch.minimum(torch.maximum(qx, qy), torch.zeros_like(qx)) - r
        if fill is not None and fill_alpha > 0:
            a = (0.5 - d).clamp(0, 1) * fill_alpha * alpha
            reg = self.c[:, y0:y1, x0:x1]
            reg.mul_(1 - a).add_(_col(fill) * a)
        if width > 0:
            self._emit(box, d, width, color, alpha, glow, gr)

    def poly(self, pts, color=WHITE, alpha=1.0, mode="over"):
        """Filled convex polygon (clockwise or not), antialiased."""
        if alpha <= 0.003:
            return
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        box = self._box(min(xs) - 2, min(ys) - 2, max(xs) + 2, max(ys) + 2)
        if not box:
            return
        x0, y0, x1, y1 = box
        X, Y = _XX[y0:y1, x0:x1], _YY[y0:y1, x0:x1]
        n = len(pts)
        area = sum(pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1] for i in range(n))
        sgn = 1 if area > 0 else -1
        d = None
        for i in range(n):
            ax, ay = pts[i]
            bx, by = pts[(i + 1) % n]
            ex, ey = bx - ax, by - ay
            L = math.hypot(ex, ey) + 1e-9
            # outward distance to the edge's line
            di = sgn * ((X - ax) * ey - (Y - ay) * ex) / L * -1
            d = di if d is None else torch.maximum(d, di)
        a = (0.5 - d).clamp(0, 1) * alpha
        reg = self.c[:, y0:y1, x0:x1]
        if mode == "add":
            reg.add_(_col(color) * a)
        else:
            reg.mul_(1 - a).add_(_col(color) * a)

    def dots(self, xy, inten, color=WHITE, size=1.6, glow=0.0):
        """Splat many points (N,2 tensor) with intensities (N,) as soft dots."""
        if xy.shape[0] == 0:
            return
        buf = torch.zeros(H * W, device=DEV)
        xi = xy[:, 0].round().long()
        yi = xy[:, 1].round().long()
        ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        buf.index_add_(0, (yi[ok] * W + xi[ok]), inten[ok])
        img = gblur(buf.view(1, 1, H, W), size)[0] * (2 * math.pi * size * size)
        self.c.add_(_col(color) * img)
        if glow:
            self.glow.add_(_col(color) * img * glow)

    # -- sprites -------------------------------------------------------
    def blit(self, sp, x, y, alpha=1.0, anchor="lt", scale=1.0, blur=0.0, mode="over", tint=None,
             clip=None):
        """Draw a sprite. anchor: two letters (l/c/r)(t/m/b/a); 'a' = text baseline.
        clip = (x0, x1) reveals only that horizontal span (screen space)."""
        if alpha <= 0.003:
            return
        t, pad = (sp.t, 0) if blur <= 0.25 else sp.blurred(blur)
        if scale != 1.0:
            t = F.interpolate(t[None], scale_factor=scale, mode="bilinear", align_corners=False,
                              antialias=scale < 1)[0]
            pad = pad * scale
        h, w = t.shape[1:]
        sw, sh = sp.w * scale, sp.h * scale
        ax, ay = anchor[0], anchor[1]
        ox = {"l": 0, "c": sw / 2, "r": sw}[ax]
        if ay == "a":
            oy = getattr(sp, "asc", sp.h) * scale
        else:
            oy = {"t": 0, "m": sh / 2, "b": sh}[ay]
        X0 = int(round(x - ox - pad))
        Y0 = int(round(y - oy - pad))
        bx0, by0 = max(0, X0), max(0, Y0)
        bx1, by1 = min(W, X0 + w), min(H, Y0 + h)
        if clip is not None:
            bx0, bx1 = max(bx0, int(clip[0])), min(bx1, int(clip[1]))
        if bx1 <= bx0 or by1 <= by0:
            return
        s = t[:, by0 - Y0:by1 - Y0, bx0 - X0:bx1 - X0]
        rgbp, a = s[:3], s[3:4] * alpha
        if tint is not None:
            rgbp = _col(tint) * s[3:4]
        reg = self.c[:, by0:by1, bx0:bx1]
        if mode == "add":
            reg.add_(rgbp * alpha)
        else:
            reg.mul_(1 - a).add_(rgbp * alpha)
        return (X0 + pad, Y0 + pad, X0 + pad + sw, Y0 + pad + sh)

    def text(self, s, x, y, size=40, face="light", color=WHITE, alpha=1.0, anchor="lt",
             tracking=0.0, lang="en", blur=0.0, scale=1.0, glow=0.0, clip=None):
        sp = text_sprite(s, size, face, color, tracking, lang)
        r = self.blit(sp, x, y, alpha, anchor, scale, blur, clip=clip)
        if glow > 0 and r:
            self.glow_sprite(sp, x, y, alpha * glow, anchor, scale)
        return r

    def glow_sprite(self, sp, x, y, k, anchor="lt", scale=1.0):
        """Feed a sprite into the bloom buffer only."""
        save = self.c
        self.c = self.glow
        self.blit(sp, x, y, k, anchor, scale, mode="add")
        self.c = save

    def fade(self, k):
        self.c.mul_(k)
        self.glow.mul_(k)


# ---------------------------------------------------------------- post
_vig = None
_grain = None


def finish(fr, t, flash=0.0, aberr=0.0, exposure=1.0):
    """Bloom → vignette → aberration → shoulder → grain. Returns uint8 HxWx3 (CPU numpy)."""
    global _vig, _grain
    c = fr.c * exposure
    if _vig is None:
        nx = (_XX / W - 0.5) * 2
        ny = (_YY / H - 0.5) * 2
        _vig = (1 - 0.32 * (nx * nx * 0.55 + ny * ny * 0.8)).clamp(0.4, 1)
        g = torch.Generator(device=DEV).manual_seed(7)
        _grain = torch.randn(8, 1, H // 2, W // 2, device=DEV, generator=g)
    # bloom from highlights + explicit glow buffer
    src = (c - 0.55).clamp(min=0) + fr.glow
    small = F.avg_pool2d(src[None], 4)
    half = F.interpolate(gblur(F.avg_pool2d(small, 2), 6), size=small.shape[2:], mode="bilinear",
                         align_corners=False)
    b = gblur(small, 3) * 0.55 + half * 0.45
    bloom = F.interpolate(b, size=(H, W), mode="bilinear", align_corners=False)[0]
    big = F.interpolate(gblur(F.avg_pool2d(src[None], 16), 4), size=(H, W), mode="bilinear",
                        align_corners=False)[0]
    c = c + bloom * 0.9 + big * 0.6
    c = c * _vig
    if aberr > 0:
        sh = int(round(aberr))
        c = torch.stack([torch.roll(c[0], sh, 1), c[1], torch.roll(c[2], -sh, 1)])
    if flash > 0:
        c = c + flash
    # soft shoulder above 0.8
    knee = 0.8
    c = torch.where(c < knee, c, knee + (1 - knee) * torch.tanh((c - knee) / (1 - knee)))
    # static dither texture: breaks up gradient banding without costing bitrate
    gr = F.interpolate(_grain[0:1], size=(H, W), mode="nearest")[0]
    c = c + gr * 0.007
    c = c.clamp(0, 1).pow(1 / 1.0)
    return (c * 255 + 0.5).to(torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy()
