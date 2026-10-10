"""Slide rendering engine: rich text, LaTeX math, diagrams, per-beat reveal.

Every visual element renders onto a transparent 1920x1080 RGBA layer so the
assembler can fade/slide new elements in on top of what is already shown.
"""
import hashlib
import os
import re
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "build", "math")
os.makedirs(CACHE, exist_ok=True)

# ---------------------------------------------------------------- palette
BG = (10, 15, 30)
PANEL = (21, 29, 52)
PANEL2 = (28, 39, 70)
LINE = (52, 66, 104)
TEXT = (232, 236, 246)
MUTED = (148, 160, 188)
BLUE = (96, 170, 255)
TEAL = (62, 212, 182)
AMBER = (255, 192, 84)
CORAL = (255, 112, 112)
VIOLET = (172, 142, 255)
GREEN = (126, 222, 126)
PINK = (255, 132, 200)

# ---------------------------------------------------------------- fonts
INTER = "/usr/share/fonts/opentype/inter/Inter-{}.otf"
CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-{}.ttc"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono{}.ttf"
_fc = {}


def font(size, bold=False, mono=False, lang="en"):
    key = (size, bold, mono, lang)
    if key not in _fc:
        if mono:
            _fc[key] = ImageFont.truetype(MONO.format("-Bold" if bold else ""), size)
        elif lang == "zh":
            # index 2 = Simplified Chinese face inside the collection
            _fc[key] = ImageFont.truetype(CJK.format("Bold" if bold else "Regular"), size, index=2)
        else:
            _fc[key] = ImageFont.truetype(INTER.format("SemiBold" if bold else "Regular"), size)
    return _fc[key]


def L(v, lang):
    """Pick the language variant of a {en:..., zh:...} value."""
    if isinstance(v, dict):
        return v.get(lang, v.get("en"))
    return v


# ---------------------------------------------------------------- rich text
_CJK_RE = re.compile(r"[⺀-鿿＀-￯　-〿]")
_MARK_RE = re.compile(r"(\*\*.+?\*\*|`.+?`)")


def _runs(text):
    """Split markup into (text, style) runs. **x** = accent bold, `x` = mono."""
    out = []
    for part in _MARK_RE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            out.append((part[2:-2], "b"))
        elif part.startswith("`") and part.endswith("`"):
            out.append((part[1:-1], "m"))
        else:
            out.append((part, "n"))
    return out


def _units(text):
    """Break opportunities: each CJK char alone; latin words keep trailing space."""
    units, buf = [], ""
    for ch in text:
        if _CJK_RE.match(ch):
            if buf:
                units.append(buf)
                buf = ""
            units.append(ch)
        elif ch == " ":
            buf += ch
            units.append(buf)
            buf = ""
        elif ch == "\n":
            if buf:
                units.append(buf)
                buf = ""
            units.append("\n")
        else:
            buf += ch
    if buf:
        units.append(buf)
    return units


_NO_LINE_START = set("，。、；：！？）》」』,.;:!?)%")


def layout_rich(text, size, maxw, lang, color, accent, bold=False):
    """Return list of lines; each line is a list of (x, text, font, color)."""
    lines, cur, x = [], [], 0
    fonts = {
        "n": font(size, bold, False, lang),
        "b": font(size, True, False, lang),
        "m": font(int(size * 0.9), False, True, lang),
    }
    colors = {"n": color, "b": accent, "m": TEAL}
    for txt, st in _runs(text):
        f = fonts[st]
        for u in _units(txt):
            if u == "\n":
                lines.append(cur)
                cur, x = [], 0
                continue
            w = f.getlength(u)
            fits = maxw is None or x + f.getlength(u.rstrip()) <= maxw
            if not fits and cur and u[0] not in _NO_LINE_START:
                lines.append(cur)
                cur, x = [], 0
                u = u.lstrip()
                w = f.getlength(u)
            cur.append((x, u, f, colors[st]))
            x += w
    lines.append(cur)
    return lines


def draw_rich(d, x, y, text, size, maxw=None, lang="en", color=TEXT, accent=AMBER,
              bold=False, align="left", lh=1.38):
    lines = layout_rich(text, size, maxw, lang, color, accent, bold)
    step = int(size * lh)
    for i, line in enumerate(lines):
        lw = (line[-1][0] + line[-1][2].getlength(line[-1][1].rstrip())) if line else 0
        ox = x
        if align == "center":
            ox = x + ((maxw or 0) - lw) / 2
        elif align == "right":
            ox = x + (maxw or 0) - lw
        for (dx, u, f, c) in line:
            dy = int(size * 0.06) if f.path.endswith(".ttf") and "Mono" in f.path else 0
            d.text((ox + dx, y + i * step + dy), u, font=f, fill=c)
    return len(lines) * step


def measure_rich(text, size, maxw=None, lang="en", bold=False, lh=1.38):
    lines = layout_rich(text, size, maxw, lang, TEXT, TEXT, bold)
    width = 0
    for line in lines:
        if line:
            width = max(width, line[-1][0] + line[-1][2].getlength(line[-1][1].rstrip()))
    return width, len(lines) * int(size * lh)


# ---------------------------------------------------------------- math
PREAMBLE = r"""\documentclass[12pt]{article}
\usepackage{amsmath,amssymb,bm,xcolor}\definecolor{cyan}{RGB}{110,230,255}\definecolor{orange}{RGB}{255,180,90}
\pagestyle{empty}
\begin{document}
\definecolor{fgc}{RGB}{%d,%d,%d}\color{fgc}\mbox{$\displaystyle %s$}
\end{document}
"""


def math_img(tex, size=40, color=TEXT):
    """Render LaTeX to an RGBA image whose em-height is roughly `size` px.

    The default color is passed to dvipng; explicit \\color{} in the TeX is kept.
    """
    dpi = int(size * 72.27 / 12)
    fg = "rgb %.4f %.4f %.4f" % tuple(c / 255 for c in color)
    key = hashlib.md5(f"{tex}|{dpi}|{fg}".encode()).hexdigest()
    path = os.path.join(CACHE, key + ".png")
    if not os.path.exists(path):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "m.tex"), "w") as fh:
                fh.write(PREAMBLE % (tuple(color) + (tex,)))
            r = subprocess.run(["latex", "-interaction=nonstopmode", "m.tex"], cwd=td,
                               capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError("LaTeX failed for: " + tex + "\n" + r.stdout[-1500:])
            subprocess.run(["dvipng", "-T", "tight", "-D", str(dpi), "-bg", "Transparent",
                            "-fg", fg, "-o", path, "m.dvi"], cwd=td,
                           capture_output=True, check=True)
    return Image.open(path).convert("RGBA")


# ---------------------------------------------------------------- elements
class El:
    def draw(self, layer, d, lang):
        raise NotImplementedError


class T(El):
    """Rich text block. x,y top-left; w wraps; align left/center/right."""

    def __init__(self, x, y, text, size=34, w=None, color=TEXT, accent=AMBER,
                 bold=False, align="left", lh=1.38):
        self.__dict__.update(locals())
        del self.__dict__["self"]

    def draw(self, layer, d, lang):
        draw_rich(d, self.x, self.y, L(self.text, lang), self.size, self.w, lang,
                  self.color, self.accent, self.bold, self.align, self.lh)


class Bul(El):
    """Bullet list with hanging indent."""

    def __init__(self, x, y, w, items, size=32, gap=14, color=TEXT, dot=AMBER,
                 accent=AMBER, marker="•", lh=1.36):
        self.__dict__.update(locals())
        del self.__dict__["self"]

    def draw(self, layer, d, lang):
        y = self.y
        ind = int(self.size * 1.15)
        for it in self.items:
            txt = L(it, lang)
            mf = font(self.size, True, False, lang) if self.marker == "•" else \
                ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", self.size)
            d.text((self.x, y), self.marker, font=mf, fill=self.dot)
            h = draw_rich(d, self.x + ind, y, txt, self.size, self.w - ind, lang,
                          self.color, self.accent, lh=self.lh)
            y += h + self.gap


class M(El):
    """LaTeX formula. anchor: 'lt' (left-top) or 'ct' (center-top) or 'cc'."""

    def __init__(self, x, y, tex, size=40, color=TEXT, anchor="lt", maxw=None):
        self.__dict__.update(locals())
        del self.__dict__["self"]

    def draw(self, layer, d, lang):
        im = math_img(L(self.tex, lang), self.size, self.color)
        if self.maxw and im.width > self.maxw:
            s = self.maxw / im.width
            im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
        x, y = self.x, self.y
        if self.anchor[0] == "c":
            x -= im.width // 2
        if self.anchor[1] == "c":
            y -= im.height // 2
        layer.alpha_composite(im, (int(x), int(y)))


class Box(El):
    """Rounded card with optional title and body text."""

    def __init__(self, x, y, w, h, title=None, body=None, fill=PANEL, border=LINE,
                 tcolor=AMBER, tsize=30, bsize=26, radius=16, bw=2, pad=22,
                 align="left", bcolor=TEXT, accent=AMBER, lh=1.34, bar=None):
        self.__dict__.update(locals())
        del self.__dict__["self"]

    def draw(self, layer, d, lang):
        d.rounded_rectangle([self.x, self.y, self.x + self.w, self.y + self.h], self.radius,
                            fill=self.fill, outline=self.border, width=self.bw)
        if self.bar:
            d.rounded_rectangle([self.x, self.y, self.x + 8, self.y + self.h], 4, fill=self.bar)
        y = self.y + self.pad - 4
        iw = self.w - 2 * self.pad
        if self.title:
            y += draw_rich(d, self.x + self.pad, y, L(self.title, lang), self.tsize, iw, lang,
                           self.tcolor, self.tcolor, True, self.align, 1.25) + 6
        if self.body:
            draw_rich(d, self.x + self.pad, y, L(self.body, lang), self.bsize, iw, lang,
                      self.bcolor, self.accent, False, self.align, self.lh)


class Chip(El):
    """Pill label sized to its text."""

    def __init__(self, x, y, text, size=26, fill=PANEL2, color=TEXT, border=None,
                 anchor="lt", padx=20, pady=10):
        self.__dict__.update(locals())
        del self.__dict__["self"]

    def size_of(self, lang):
        w, h = measure_rich(L(self.text, lang), self.size, None, lang, lh=1.2)
        return int(w + 2 * self.padx), int(self.size * 1.2 + 2 * self.pady)

    def draw(self, layer, d, lang):
        w, h = self.size_of(lang)
        x, y = self.x, self.y
        if self.anchor[0] == "c":
            x -= w // 2
        d.rounded_rectangle([x, y, x + w, y + h], h // 2, fill=self.fill,
                            outline=self.border, width=2 if self.border else 0)
        draw_rich(d, x + self.padx, y + self.pady, L(self.text, lang), self.size, None, lang,
                  self.color, self.color, False, lh=1.2)


class Rect(El):
    def __init__(self, x, y, w, h, fill=PANEL, border=None, radius=8, bw=2):
        self.__dict__.update(locals())
        del self.__dict__["self"]

    def draw(self, layer, d, lang):
        d.rounded_rectangle([self.x, self.y, self.x + self.w, self.y + self.h], self.radius,
                            fill=self.fill, outline=self.border, width=self.bw if self.border else 0)


class Arrow(El):
    """Polyline with an arrowhead at the end (and optionally the start)."""

    def __init__(self, pts, color=MUTED, width=4, head=16, both=False, dash=False):
        self.__dict__.update(locals())
        del self.__dict__["self"]

    @staticmethod
    def _head(d, a, b, color, head):
        import math
        ang = math.atan2(b[1] - a[1], b[0] - a[0])
        p1 = (b[0] - head * math.cos(ang - 0.42), b[1] - head * math.sin(ang - 0.42))
        p2 = (b[0] - head * math.cos(ang + 0.42), b[1] - head * math.sin(ang + 0.42))
        d.polygon([b, p1, p2], fill=color)

    def draw(self, layer, d, lang):
        pts = self.pts
        if self.dash:
            import math
            for a, b in zip(pts[:-1], pts[1:]):
                n = max(1, int(math.dist(a, b) / 18))
                for i in range(0, n, 2):
                    s = (a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n)
                    e = (a[0] + (b[0] - a[0]) * (i + 1) / n, a[1] + (b[1] - a[1]) * (i + 1) / n)
                    d.line([s, e], fill=self.color, width=self.width)
        else:
            d.line(pts, fill=self.color, width=self.width, joint="curve")
        self._head(d, pts[-2], pts[-1], self.color, self.head)
        if self.both:
            self._head(d, pts[1], pts[0], self.color, self.head)


class Code(El):
    """Monospace code panel with light keyword/comment highlighting."""

    KW = {"while", "for", "in", "if", "not", "def", "return", "else", "elif", "import", "from"}

    def __init__(self, x, y, w, code, size=26, title=None, hl=None):
        self.__dict__.update(locals())
        del self.__dict__["self"]

    def draw(self, layer, d, lang):
        lines = L(self.code, lang).strip("\n").split("\n")
        lh = int(self.size * 1.45)
        top = 54 if self.title else 22
        h = top + lh * len(lines) + 22
        d.rounded_rectangle([self.x, self.y, self.x + self.w, self.y + h], 14, fill=(14, 20, 38),
                            outline=LINE, width=2)
        if self.title:
            d.text((self.x + 22, self.y + 14), L(self.title, lang),
                   font=font(22, True, False, lang), fill=MUTED)
        f = font(self.size, False, True)
        fb = font(self.size, True, True)
        for i, line in enumerate(lines):
            y = self.y + top + i * lh
            if self.hl and i in self.hl:
                d.rectangle([self.x + 6, y - 4, self.x + self.w - 6, y + lh - 6], fill=(40, 52, 88))
            x = self.x + 24
            if "#" in line:
                code, com = line.split("#", 1)
                com = "#" + com
            else:
                code, com = line, ""
            for tok in re.split(r"(\W)", code):
                if not tok:
                    continue
                if tok in self.KW:
                    d.text((x, y), tok, font=fb, fill=VIOLET)
                elif re.match(r"^[A-Za-z_]\w*$", tok) and code[code.find(tok) + len(tok):].startswith("("):
                    d.text((x, y), tok, font=f, fill=BLUE)
                else:
                    d.text((x, y), tok, font=f, fill=TEXT)
                x += f.getlength(tok)
            if com:
                d.text((x, y), com, font=f, fill=(110, 170, 120))


class Tbl(El):
    """Table with wrapped rich-text cells; header row styled."""

    def __init__(self, x, y, cols, rows, size=24, hsize=None, pad=12, header=True,
                 colors=None, hcolor=AMBER, zebra=True, minh=0, lh=1.3):
        self.__dict__.update(locals())
        del self.__dict__["self"]

    def draw(self, layer, d, lang):
        y = self.y
        hs = self.hsize or self.size
        tw = sum(self.cols)
        for ri, row in enumerate(self.rows):
            is_h = self.header and ri == 0
            sz = hs if is_h else self.size
            hts, imgs = [], {}
            for ci, cell in enumerate(row):
                if isinstance(cell, tuple) and cell[0] == "$":
                    im = math_img(cell[1], cell[2] if len(cell) > 2 else int(sz * 1.25), TEXT)
                    lim = self.cols[ci] - 2 * self.pad
                    if im.width > lim:
                        s = lim / im.width
                        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
                    imgs[ci] = im
                    hts.append(im.height)
                    continue
                _, h = measure_rich(L(cell, lang), sz, self.cols[ci] - 2 * self.pad, lang,
                                    bold=is_h, lh=self.lh)
                hts.append(h)
            rh = max(max(hts) + 2 * self.pad, self.minh)
            if is_h:
                d.rectangle([self.x, y, self.x + tw, y + rh], fill=(30, 42, 76))
            elif self.zebra and ri % 2 == 0:
                d.rectangle([self.x, y, self.x + tw, y + rh], fill=(18, 25, 46))
            x = self.x
            for ci, cell in enumerate(row):
                col = self.hcolor if is_h else (self.colors[ci] if self.colors else TEXT)
                if ci in imgs:
                    im = imgs[ci]
                    layer.alpha_composite(im, (int(x + self.pad), int(y + (rh - im.height) / 2)))
                else:
                    th = hts[ci]
                    draw_rich(d, x + self.pad, y + max(self.pad, (rh - th) / 2), L(cell, lang), sz,
                              self.cols[ci] - 2 * self.pad, lang, col, AMBER, bold=is_h, lh=self.lh)
                x += self.cols[ci]
            y += rh
            d.line([self.x, y, self.x + tw, y], fill=LINE, width=1)


class Bars(El):
    """Horizontal bar chart: items = [(label, value, color, value_label)]."""

    def __init__(self, x, y, w, items, vmax, size=24, bh=34, gap=26, lw=260):
        self.__dict__.update(locals())
        del self.__dict__["self"]

    def draw(self, layer, d, lang):
        y = self.y
        for label, v, c, vl in self.items:
            draw_rich(d, self.x, y + 2, L(label, lang), self.size, self.lw - 10, lang, TEXT)
            bw = max(4, int((self.w - self.lw - 170) * v / self.vmax))
            d.rounded_rectangle([self.x + self.lw, y, self.x + self.lw + bw, y + self.bh], 6, fill=c)
            d.text((self.x + self.lw + bw + 14, y + 2), L(vl, lang),
                   font=font(self.size, True, False, lang), fill=c)
            y += self.bh + self.gap


class Custom(El):
    """Escape hatch: fn(layer, draw, lang)."""

    def __init__(self, fn):
        self.fn = fn

    def draw(self, layer, d, lang):
        self.fn(layer, d, lang)


# ---------------------------------------------------------------- scenes
class Beat:
    def __init__(self, say, add=(), pause=0.5):
        self.say = say
        self.add = list(add)
        self.pause = pause


class Scene:
    def __init__(self, sid, part, title, beats, header=True):
        self.sid, self.part, self.title, self.beats, self.header = sid, part, title, beats, header


def render_layer(elements, lang):
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for el in elements:
        el.draw(layer, d, lang)
    return layer


def header_layer(scene, lang, idx, total):
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    if not scene.header:
        return layer
    d = ImageDraw.Draw(layer)
    part = L(scene.part, lang)
    d.text((100, 44), part, font=font(24, True, False, lang), fill=TEAL)
    d.text((100, 78), L(scene.title, lang), font=font(50, True, False, lang), fill=TEXT)
    d.rectangle([100, 150, 220, 154], fill=AMBER)
    tag = "Post-Training Systems 101" if lang == "en" else "后训练系统 101"
    d.text((100, 1022), tag, font=font(20, False, False, lang), fill=(98, 110, 140))
    num = f"{idx:02d} / {total:02d}"
    f = font(20, False, False, lang)
    d.text((1820 - f.getlength(num), 1022), num, font=f, fill=(98, 110, 140))
    return layer


def background():
    """Static backdrop: subtle vertical gradient plus faint grid."""
    import numpy as np
    yy = np.linspace(0, 1, H)[:, None]
    top = np.array(BG, dtype=float)
    bot = np.array((14, 22, 44), dtype=float)
    arr = (top * (1 - yy[..., None]) + bot * yy[..., None]) * np.ones((H, W, 1))
    arr = arr.astype("uint8")
    im = Image.fromarray(arr, "RGB").convert("RGBA")
    d = ImageDraw.Draw(im)
    for gx in range(0, W, 80):
        d.line([gx, 0, gx, H], fill=(18, 25, 46, 255), width=1)
    for gy in range(0, H, 80):
        d.line([0, gy, W, gy], fill=(18, 25, 46, 255), width=1)
    return im
