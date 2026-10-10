# Post-Training Systems, Layer by Layer · 后训练工具体系：分层拆解

**Site · 网站:** https://haokun12x.github.io/post-training-101/

A senior post-training engineer's map of the LLM post-training tool stack, built from one equation:

$$\nabla_\theta J=\mathbb{E}_{(x,y)\sim\mu}\Big[\sum_t \psi_t\,\nabla_\theta\log\pi_\theta(y_t\mid x,y_{<t})\Big]$$

μ fixed → a trainer problem; μ = π<sub>θ</sub> → a systems problem. From there: six layers, the memory arithmetic, the RL loop,
where frameworks really differ, the anatomy of the frameworks (shared atoms, trade-off axes, and the code address of every
term of the objective in each framework), and how to compose them.

从一个公式出发的后训练工具体系：μ 固定是训练器问题，μ = π<sub>θ</sub> 是系统问题。由此展开六层工具栈、显存算术、强化学习闭环、
框架之间真正的差异，框架的拆解（共享原子、取舍轴、目标函数每一项在各框架代码中的地址与改动深度），以及如何选型与组合。

## What's here · 内容

| Path | What |
|---|---|
| `docs/` | The website (GitHub Pages): interactive knowledge map, calculators, both films with chapters and bilingual subtitles, the trailers |
| `docs/media/full_en.mp4`, `full_zh.mp4` | ~32-minute deep dives (24 chapters in six parts): English with narration; Chinese captioned, no voice-over |
| `docs/media/promo_en.mp4`, `promo_zh.mp4` | 1:56 trailers for video platforms (1080p60, captions burned in + soft subtitle track); the Chinese trailer is music-only |
| `docs/media/script_bilingual.md` | Side-by-side EN / 中文 narration script with timestamps |
| `reports/` | The two source research reports (framework survey + source-code engineering review, as of 2026-10-10) |
| `video/` | Source of the long films: slide engine, scenes, offline TTS, assembler |
| `promo/` | Source of the trailer: GPU motion-graphics engine, shot designs, procedural score and mix |
| `site_tools/` | Exports chapter titles, narration and timestamps for the website |

## Rebuild · 重新生成

Python 3.12 with `kokoro misaki[zh,en] torch soundfile pillow numpy`, a TeX install (`latex`, `dvipng`), `ffmpeg`,
Inter and Noto Sans/Serif CJK fonts, and system `libespeak-ng`. A CUDA GPU makes the trailer render in a few minutes.

```bash
# long films (see video/README.md)
cd video/src && python build.py both && python make_script.py
# the published Chinese film carries no voice-over: drop its audio track
ffmpeg -i video/output/post_training_systems_zh.mp4 -map 0:v -map 0:s -c copy -movflags +faststart docs/media/full_zh.mp4
# trailers: narration → timeline → score + frames → mux
cd promo/src && python make.py both
# website chapter data
python site_tools/export_chapters.py
```

English narration is synthesized offline with Kokoro-82M (`af_heart`). The Chinese film and trailer are published without voice-over (captions only).
The trailer's music and sound design are synthesized procedurally in `promo/src/music.py` (no samples).

No unified GPU benchmark was run for the comparisons: recommendations are engineering judgments, not a speed ranking.
