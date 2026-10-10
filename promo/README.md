# Trailer · 预告片

1:44, 1920×1080 at 60 fps, English and Chinese cuts. Outputs land in `output/` (the site copies them to `docs/media/`).

| File | Role |
|---|---|
| `src/script.py` | Narration per shot (`say` for the voice, `sub` for captions) |
| `src/timeline.py` | Places sentences on a 120 BPM grid; shot lengths are shared by both languages so one score fits both |
| `src/gfx.py` | GPU (torch) motion-graphics engine: SDF lines/rings/panels with glow, text and LaTeX sprites, bloom, vignette |
| `src/shots.py` | The eleven shot designs (cold open → outro) |
| `src/music.py` | Procedural score and sound design: additive pads, plucked arpeggio, drums, risers, impacts, reverb, ducking |
| `src/render.py` | Frame renderer (`--still t1,t2` for review stills) |
| `src/make.py` | Audio mix + frames + subtitles → `output/promo_<lang>.mp4` (loudness −14 LUFS) |

```bash
python make.py both            # full build
python render.py en --still 22,46,72   # review frames
```
