"""Offline narration with Kokoro-82M (English) and Kokoro-82M-v1.1-zh (Chinese).

Each narration beat is split into sentences; every sentence is synthesized and
cached separately so subtitle timing can follow the real audio.
"""
import hashlib
import os
import re

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "build", "tts")
os.makedirs(CACHE, exist_ok=True)
SR = 24000
VOICE = {"en": "af_heart", "zh": "zf_001"}

# Spoken forms for acronyms / names the G2P would otherwise mangle.
TERMS = [
    ("DeepSeek-V3", "DeepSeek V 3"), ("Qwen3 30B-A3B", "Qwen 3, 30 B A 3 B"),
    ("Qwen3-30B-A3B", "Qwen 3, 30 B A 3 B"), ("GB200s", "G B 200s"), ("GB200", "G B 200"),
    ("H100s", "H 100s"), ("H100", "H 100"), ("OpenRLHF", "Open R L H F"), ("RLHF", "R L H F"),
    ("SkyRL", "Sky R L"), ("NeMo RL", "Nemo R L"), ("NeMo", "Nemo"), ("Prime-RL", "Prime R L"),
    ("prime-rl", "prime R L"), ("RLinf", "R L inf"), ("rLLM", "R L L M"), ("vLLM", "V L L M"),
    ("SGLang", "S G Lang"), ("FSDP2", "F S D P 2"), ("FSDP", "F S D P"), ("ZeRO-3", "Zero 3"),
    ("ZeRO", "Zero"), ("QLoRA", "Q Lora"), ("LoRA", "Lora"), ("GRPO", "G R P O"), ("PPO", "P P O"),
    ("DPO", "D P O"), ("SFT", "S F T"), ("RLVR", "R L V R"), ("TRL", "T R L"), ("MoE", "M O E"),
    ("VLA", "V L A"), ("KV", "K V"), ("KL", "K L"), ("LM", "L M"), ("HF", "Hugging Face"),
    ("ms-swift", "M S Swift"), ("LLaMA-Factory", "Llama Factory"), ("NCCL", "nickel"),
    ("NVMe", "N V M E"), ("AutoTP", "Auto T P"), ("AutoEP", "Auto E P"), ("DTensor", "D Tensor"),
    ("Megatron-Core", "Megatron Core"), ("DeepSpeed-Chat", "DeepSpeed Chat"), ("AReaL", "Areal"),
    ("ROLL", "Roll"), ("ART", "Art"), ("PEFT", "peft"), ("bitsandbytes", "bits and bytes"),
    ("get_peft_model", "get peft model"), ("compute_loss_func", "compute loss func"),
    ("Z_S", "Z S"), ("bf16", "B F 16"), ("fp32", "F P 32"), ("IDs", "I Ds"), ("API", "A P I"),
    ("TPUs", "T P Us"), ("TPU", "T P U"), ("JAX", "Jax"), ("GPU-hours", "GPU hours"),
    ("VLM", "V L M"), ("UTD", "U T D"), ("PipelineRL", "Pipeline R L"), ("ReaLHF", "Real H F"),
    ("K2", "K 2"), ("W0", "W 0"), ("NormalFloat", "Normal Float"), ("HybridFlow", "Hybrid Flow"),
    ("TorchTitan", "Torch Titan"), ("TorchForge", "Torch Forge"), ("PyTorch", "Pie Torch"),
    ("DeepSpeed", "Deep Speed"), ("DeepSeek", "Deep Seek"), ("RL", "R L"), ("ID", "I D"),
]
ZH_EXTRA = [("GPU", "G P U"), ("CPU", "C P U"), ("GB", "G B")]


def spoken(text, lang):
    for a, b in TERMS + (ZH_EXTRA if lang == "zh" else []):
        text = re.sub(r"(?<!\w)" + re.escape(a) + r"(?![\w])", b, text)
    if lang == "zh":
        # Chinese G2P reads ASCII ellipsis/dashes poorly; keep punctuation simple.
        text = text.replace("“", "").replace("”", "")
    return text


def sentences(text, lang):
    if lang == "zh":
        parts = re.split(r"(?<=[。！？；])", text)
    else:
        parts = re.split(r"(?<=[.!?;])\s+", text)
    return [p.strip() for p in parts if p.strip()]


_pipes = {}


def _pipe(lang):
    if lang in _pipes:
        return _pipes[lang]
    import espeak_fix  # noqa: F401  (point misaki at the system espeak-ng)
    import torch
    from kokoro import KModel, KPipeline
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if lang == "en":
        p = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device=dev)
    else:
        repo = "hexgrad/Kokoro-82M-v1.1-zh"
        model = KModel(repo_id=repo).to(dev).eval()
        en = KPipeline(lang_code="a", repo_id=repo, model=False)

        def en_callable(t):
            return next(en(t)).phonemes

        p = KPipeline(lang_code="z", repo_id=repo, model=model, en_callable=en_callable)
    _pipes[lang] = p
    return p


def _zh_speed(len_ps):
    # From the Kokoro v1.1-zh reference script: slow down long phoneme runs.
    speed = 0.8
    if len_ps <= 83:
        speed = 1
    elif len_ps < 183:
        speed = 1 - (len_ps - 83) / 500
    return speed * 1.1


def synth(sentence, lang):
    """Return float32 audio for one sentence (cached on disk)."""
    sp = spoken(sentence, lang)
    key = hashlib.md5(f"{lang}|{VOICE[lang]}|{sp}".encode()).hexdigest()
    path = os.path.join(CACHE, f"{lang}_{key}.wav")
    if os.path.exists(path):
        a, _ = sf.read(path, dtype="float32")
        return a
    p = _pipe(lang)
    speed = 1.0 if lang == "en" else _zh_speed
    chunks = [r.audio.cpu().numpy() for r in p(sp, voice=VOICE[lang], speed=speed)]
    a = np.concatenate(chunks).astype("float32") if chunks else np.zeros(SR // 4, "float32")
    # trim leading/trailing near-silence so gaps are controlled by the assembler
    nz = np.where(np.abs(a) > 0.01)[0]
    if len(nz):
        a = a[max(0, nz[0] - 600): nz[-1] + 1200]
    sf.write(path, a, SR)
    return a


GAP = {"en": 0.28, "zh": 0.26}


def beat_audio(text, lang):
    """Synthesize a beat. Returns (audio, [(start_s, end_s, sentence), ...])."""
    out, segs, t = [], [], 0.0
    gap = np.zeros(int(GAP[lang] * SR), "float32")
    for i, s in enumerate(sentences(text, lang)):
        a = synth(s, lang)
        segs.append((t, t + len(a) / SR, s))
        out.append(a)
        t += len(a) / SR
        out.append(gap)
        t += GAP[lang]
    audio = np.concatenate(out[:-1]) if out else np.zeros(1, "float32")
    return audio, segs
