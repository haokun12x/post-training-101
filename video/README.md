# Post-Training Systems, Layer by Layer · 后训练工具体系：分层拆解

A narrated explainer video (English + Chinese), synthesized from the two reports in `../reports/`:
`reports/post-training-investigate.md` and `reports/post_training_engineering_review_2026-10-10.pdf`.

## Output (`output/`)

| File | What |
|---|---|
| `post_training_systems_en.mp4` | English narration + English slides; soft subtitles: EN (default) and ZH |
| `post_training_systems_zh.mp4` | 中文解说 + 中文幻灯片；软字幕：中文（默认）与英文 |
| `*.en.srt`, `*.zh.srt` | Subtitle files for each video (own language follows the real audio; the other language is aligned per narration beat) |
| `script_bilingual.md` | Side-by-side EN / 中文 narration script with scene timestamps |

## Structure (24 scenes, 6 parts)

1. **One equation**: every method is ∇J = E_μ[Σ ψ_t ∇log π]; μ fixed → trainer problem, μ = π_θ → systems problem
2. **The stack**: six layers; LoRA/QLoRA arithmetic; ZeRO/FSDP2/Megatron sharding math; generation engines and weight sync
3. **Offline vs. online**: SFT/DPO and the normalizer; the RL loop, PPO/GRPO, the four probability tracks
4. **Where systems differ**: step-time/Amdahl and published numbers; asynchrony mechanisms; five train–inference error sources (top-p math); six algorithmic freedoms (entropy-gradient example); framework entry points; landscape
5. **Anatomy**: frameworks as recipes of shared atoms (backbone × engine × orchestrator + connective layer); four trade-off axes; the objective with 13 numbered positions (data rule ①–⑥, learning rule ⑦–⑫, publish ⑬); the position × framework address book with depth levels L0–L4; three changes traced end to end
6. **Choose & compose**: trajectory contract, minimal architecture, decision procedure, two-stage validation

## Rebuild

Requires: Python 3.12 venv with `kokoro misaki[zh,en] torch soundfile pillow numpy`, a TeX install with `latex` + `dvipng`, `ffmpeg`, Inter and Noto Sans CJK fonts, and system `libespeak-ng`.

```bash
cd src
python preview.py en zh          # still frames of every scene → build/preview/
python build.py both             # narration (cached in build/tts) + both videos
python make_script.py            # bilingual script with timestamps
```

Slide content and narration live in `src/scenes.py`; pronunciation fixes for acronyms in `src/tts.py` (`TERMS`).
Narration is synthesized offline with Kokoro-82M (`af_heart`) and Kokoro-82M-v1.1-zh (`zf_001`).
