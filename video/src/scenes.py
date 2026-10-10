"""Scene content: bilingual slide text, formulas, and narration.

Sources: ../post-training-investigate.md and
../post_training_engineering_review_2026-10-10.pdf (source-code review pinned
to framework releases as of 2026-10-09/10).
"""
from engine import (AMBER, BLUE, CORAL, GREEN, LINE, MUTED, PANEL, PANEL2, PINK, TEAL, TEXT,
                    VIOLET, Arrow, Bars, Beat, Box, Bul, Chip, Code, Custom, M, Rect, Scene, T,
                    Tbl, font, draw_rich, L)


def tr(en, zh):
    return {"en": en, "zh": zh}


P1 = tr("PART 1 · ONE EQUATION", "第一部分 · 一个公式")
P2 = tr("PART 2 · THE STACK, LAYER BY LAYER", "第二部分 · 逐层拆解工具栈")
P3 = tr("PART 3 · OFFLINE VS. ONLINE", "第三部分 · 离线与在线")
P4 = tr("PART 4 · WHERE SYSTEMS REALLY DIFFER", "第四部分 · 系统真正的差异")
P5 = tr("PART 5 · CHOOSE & COMPOSE", "第五部分 · 选型与组合")

SCENES = []

# =========================================================== 0. title
SCENES.append(Scene("title", "", "", header=False, beats=[
    Beat(tr(
        "Welcome. In this video we'll build a systematic mental model of the post-training tool "
        "ecosystem, the way a senior post-training engineer sees it. Not a list of frameworks, but a "
        "layered map: what each layer computes, what math it must preserve, and how the pieces compose "
        "into a working system.",
        "欢迎。这期视频从一位资深后训练工程师的视角，为后训练工具生态建立一套系统化的理解。我们不罗列框架，"
        "而是画一张分层地图：每一层在计算什么，必须守住哪些数学语义，以及这些部件怎样组合成一个能真正运行的系统。"),
        add=[
            T(0, 300, tr("Post-Training Systems, Layer by Layer", "后训练工具体系：分层拆解"),
              size=80, w=1920, align="center", bold=True),
            T(0, 430, tr("How the tools compose, through math, algorithms, and engineering logic",
                         "工具如何组合：数学、算法与工程逻辑"),
              size=38, w=1920, align="center", color=MUTED),
            Rect(900, 515, 120, 5, fill=AMBER, radius=2),
            T(0, 556, tr("A senior post-training engineer's map  ·  sources: framework survey + source-code review, as of 2026-10-10",
                         "资深后训练工程师视角  ·  资料：框架调研与源码审查（截至 2026-10-10）"),
              size=26, w=1920, align="center", color=(120, 132, 160)),
        ]),
    Beat(tr(
        "Everything we discuss comes back to three forces that pull against each other: how freely you "
        "can change the algorithm, how much engineering complexity you take on, and how efficiently the "
        "whole loop executes. The material comes from the two research reports in this folder, one of "
        "which reviews the source code of the major frameworks at their October 2026 releases.",
        "贯穿全片的是三股彼此拉扯的力量：算法能改得多自由，要承担多少工程复杂度，以及整个循环跑得多高效。"
        "素材来自这个目录下的两份调研报告，其中一份对主流框架做了源码级审查，版本固定在 2026 年 10 月的发布。"),
        add=[
            Chip(560, 700, tr("Algorithm freedom", "算法自由度"), size=32, fill=(40, 34, 18), color=AMBER, border=AMBER, anchor="ct"),
            Chip(960, 700, tr("Engineering complexity", "工程复杂度"), size=32, fill=(18, 32, 56), color=BLUE, border=BLUE, anchor="ct"),
            Chip(1360, 700, tr("Execution efficiency", "执行效率"), size=32, fill=(14, 44, 40), color=TEAL, border=TEAL, anchor="ct"),
        ]),
]))

# =========================================================== 1. roadmap
_road = [
    (tr("One equation", "一个公式"), tr("μ and ψ: one gradient behind SFT, DPO, PPO, GRPO and distillation",
                                        "μ 与 ψ：SFT、DPO、PPO、GRPO 与蒸馏背后的同一个梯度"), AMBER),
    (tr("The stack", "工具栈"), tr("Parameterization, execution and generation layers, with the memory math",
                                   "参数化、训练执行、推理生成三层，以及显存的算术"), BLUE),
    (tr("Offline vs. online", "离线与在线"), tr("Fixed data is a trainer problem; on-policy data is a systems problem",
                                              "固定数据是训练器问题；在线数据是系统问题"), TEAL),
    (tr("Where systems differ", "系统差异"), tr("Step time, asynchrony, consistency, algorithmic freedom, entry points",
                                              "单步时间、异步、一致性、算法自由度与入口"), VIOLET),
    (tr("Choose & compose", "选型与组合"), tr("Trajectory contract, minimal architecture, decision procedure, validation",
                                             "轨迹契约、最小架构、决策流程与验证"), CORAL),
]
_road_els = []
for i, (t, b, c) in enumerate(_road):
    x = 100 + i * 348
    _road_els += [
        Box(x, 260, 328, 470, fill=PANEL, border=LINE, bar=None),
        T(x + 26, 286, f"0{i + 1}", size=64, bold=True, color=c),
        T(x + 26, 390, t, size=32, w=280, bold=True, color=TEXT),
        T(x + 26, 470, b, size=25, w=280, color=MUTED),
    ]
SCENES.append(Scene("roadmap", tr("ROADMAP", "路线图"), tr("Five parts", "五个部分"), beats=[
    Beat(tr(
        "Here's the roadmap. Part one derives a single equation that covers nearly every post-training "
        "method. Part two walks the stack layer by layer, with the memory math. Part three contrasts offline "
        "and online training. Part four covers where systems really differ: step time, asynchrony, "
        "consistency, and what algorithmic freedom actually means. Part five turns it all into a data "
        "contract, an architecture, a decision procedure, and a validation protocol.",
        "先看路线图。第一部分推导一个几乎涵盖所有后训练方法的统一公式。第二部分逐层走过工具栈，并算清显存账。"
        "第三部分对比离线训练与在线训练。第四部分讨论系统真正拉开差距的地方：单步时间、异步、训练推理一致性，"
        "以及“算法自由度”究竟指什么。第五部分把这些落成数据契约、最小架构、选型流程和验证方法。"),
        add=_road_els),
]))

# =========================================================== 2. unified gradient
_grad_tex = r"\nabla_\theta J=\mathbb{E}_{(x,y)\sim{\color{cyan}\mu}}\Big[\sum_t {\color{orange}\psi_t}\,\nabla_\theta\log\pi_\theta(y_t\mid x,y_{<t})\Big]"
SCENES.append(Scene("unified", P1, tr("One gradient behind every method", "一个梯度统一所有后训练方法"), beats=[
    Beat(tr(
        "Start with the math. Almost every post-training method can be written as one gradient. Sample a "
        "prompt x and a response y from some data distribution mu, then push up the log-probability of each "
        "token, weighted by a coefficient psi. Methods differ in only two places: where mu comes from, and how "
        "psi is computed.",
        "先从数学开始。几乎所有后训练方法都能写成同一个梯度：从某个数据分布 mu 中取出提示 x 和回答 y，"
        "然后提高每个 token 的对数概率，并用系数 psi 加权。不同方法只在两处不同：mu 从哪里来，psi 怎么算。"),
        add=[
            M(960, 190, _grad_tex, size=50, anchor="ct"),
            Chip(640, 318, tr("μ: where the data comes from", "μ：数据从哪里来"), size=26, fill=(14, 40, 52), color=(110, 230, 255), anchor="ct"),
            Chip(1280, 318, tr("ψ: how each token is weighted", "ψ：每个 token 如何加权"), size=26, fill=(52, 36, 14), color=(255, 180, 90), anchor="ct"),
        ]),
    Beat(tr(
        "Supervised fine-tuning uses a fixed dataset, and psi is one on the masked response tokens. DPO also "
        "uses fixed data, preference pairs, and psi is a sigmoid-weighted push up on the chosen answer and "
        "down on the rejected one. PPO and GRPO sample from the policy itself and weight tokens by a clipped "
        "importance ratio times an advantage. On-policy distillation samples from the student and uses the "
        "teacher-minus-student log-probability as the signal.",
        "监督微调用固定数据集，psi 在被 mask 选中的回答 token 上等于 1。DPO 同样用固定数据，也就是偏好对，"
        "psi 是一个 sigmoid 加权：拉高被选中的回答，压低被拒绝的回答。PPO 和 GRPO 从策略本身采样，"
        "用截断后的重要性比率乘以优势来加权。在线蒸馏从学生模型采样，用教师与学生的对数概率之差作为信号。"),
        add=[
            Tbl(100, 410, [330, 560, 830], [
                [tr("Method", "方法"), tr("Data source μ", "数据来源 μ"), tr("Token weight ψ", "token 权重 ψ")],
                ["SFT", tr("fixed dataset D", "固定数据集 D"), ("$", r"m_t\in\{0,1\}\ \ (\text{loss mask})", 34)],
                ["DPO", tr("fixed preference pairs (chosen, rejected)", "固定偏好对（chosen, rejected）"),
                 ("$", r"\pm\,\beta\,\sigma(\hat r_l-\hat r_w),\ \ \hat r=\beta\log\frac{\pi_\theta}{\pi_{\mathrm{ref}}}", 34)],
                ["PPO / GRPO", tr("fresh samples from the old policy (online)", "从旧策略实时采样（在线）"),
                 ("$", r"\rho_t\hat A_t\ \text{(clipped)},\ \ \rho_t=\frac{\pi_\theta}{\pi_{\mathrm{old}}}", 34)],
                [tr("On-policy distillation", "在线蒸馏"), tr("fresh samples from the student", "从学生模型实时采样"),
                 ("$", r"\log\pi_{\mathrm{teacher}}(y_t)-\log\pi_\theta(y_t)", 34)],
            ], size=27, minh=84),
        ]),
    Beat(tr(
        "This one observation organizes the whole tool landscape. If mu is a fixed dataset, post-training is a "
        "trainer problem: load data, compute the loss, step the optimizer. If mu is the policy itself, it becomes "
        "a systems problem: you need a generation engine, reward computation, weight synchronization, and "
        "bookkeeping for which policy produced which token. That is why RL frameworks look so different from "
        "fine-tuning libraries.",
        "这一个观察就能组织起整个工具版图。如果 mu 是固定数据集，后训练就是一个训练器问题：读数据、算损失、走优化器。"
        "如果 mu 就是策略本身，它就变成一个系统问题：需要生成引擎、奖励计算、权重同步，还要记录每个 token "
        "是由哪个版本的策略产生的。这正是强化学习框架和微调库看起来差别如此之大的原因。"),
        add=[
            Box(100, 840, 830, 150, title=tr("μ fixed  →  a trainer problem", "μ 固定  →  训练器问题"),
                body=tr("load data · compute loss · step the optimizer", "读数据 · 算损失 · 走优化器"),
                tcolor=BLUE, bar=BLUE, tsize=32, bsize=27),
            Box(990, 840, 830, 150, title=tr("μ = current policy (moving)  →  a systems problem", "μ = 当前策略（随训练变化） →  系统问题"),
                body=tr("generation engine · rewards & environments · weight sync · version bookkeeping",
                        "生成引擎 · 奖励与环境 · 权重同步 · 版本记账"),
                tcolor=CORAL, bar=CORAL, tsize=32, bsize=27),
        ]),
]))

# =========================================================== 3. the stack
_layers = [
    (tr("Algorithm & experiment", "算法与实验层"),
     tr("reward · advantage · loss · critic · teacher · update rule", "奖励 · 优势 · 损失 · critic · 教师 · 更新规则"),
     tr("**your code** · algorithm parts of TRL, SkyRL, slime, verl", "**你自己的代码** · TRL、SkyRL、slime、verl 的算法实现"), AMBER),
    (tr("Data & runtime", "数据与运行层"),
     tr("rollout · env · multi-turn · buffer · placement · versions", "rollout · 环境 · 多轮 · buffer · 资源布局 · 版本"),
     tr("verl · OpenRLHF · SkyRL · slime · Miles · AReaL · ROLL · NeMo RL", "verl · OpenRLHF · SkyRL · slime · Miles · AReaL · ROLL · NeMo RL"), CORAL),
    (tr("Model & training execution", "模型与训练执行层"),
     tr("fwd/bwd · state sharding · TP/PP/EP/CP · optimizer", "前向/反向 · 状态分片 · TP/PP/EP/CP · 优化器"),
     tr("PyTorch FSDP2 · DeepSpeed · Megatron-Core · TorchTitan · NeMo AutoModel", "PyTorch FSDP2 · DeepSpeed · Megatron-Core · TorchTitan · NeMo AutoModel"), BLUE),
    (tr("Inference generation", "推理生成层"),
     tr("batching · KV cache · prefill/decode · sampling · hot update", "batching · KV cache · prefill/decode · 采样 · 热更新"),
     tr("vLLM · SGLang  (JAX/TPU stacks have their own)", "vLLM · SGLang（JAX/TPU 路线另有对应后端）"), TEAL),
    (tr("Parameterization & kernels", "参数化与 kernel 层"),
     tr("LoRA/QLoRA · attention · fused loss · low precision · comms", "LoRA/QLoRA · attention · 融合损失 · 低精度 · 通信"),
     tr("PEFT · bitsandbytes · Liger · FlashAttention · FlashInfer · Transformer Engine · DeepEP", "PEFT · bitsandbytes · Liger · FlashAttention · FlashInfer · Transformer Engine · DeepEP"), VIOLET),
    (tr("Data quality & observability", "数据质量与观测层"),
     tr("processing · verifiers · eval · logs · profiling · checkpoints", "数据加工 · 验证器 · 评估 · 日志 · profile · checkpoint"),
     tr("DataTrove · distilabel · Math-Verify · lm-eval · W&B · Nsight", "DataTrove · distilabel · Math-Verify · lm-eval · W&B · Nsight"), GREEN),
]


def _band(i, name, duty, tools, c):
    y = 186 + i * 124
    return [
        Box(100, y, 1180, 118, fill=PANEL, border=LINE, bar=c, radius=12),
        T(130, y + 12, name, size=29, bold=True, color=c, w=520),
        T(130, y + 54, duty, size=20, color=MUTED, w=520, lh=1.3),
        T(660, y + 18, tools, size=24, w=600, lh=1.32),
    ]


SCENES.append(Scene("stack", P2, tr("Six layers, one component per role", "六层工具栈：每个角色选一个部件"), beats=[
    Beat(tr(
        "Now the stack. The source-code review organizes the ecosystem into six layers. On top is the algorithm "
        "layer: rewards, advantages, losses, critics, teachers, and update rules. That's where your research lives. "
        "Below it is the data and runtime layer: rollouts, environments, multi-turn trajectories, buffers, resource "
        "placement, and policy versions. This is what verl, OpenRLHF, SkyRL, AReaL, Miles, ROLL and NeMo RL actually "
        "compete on.",
        "接下来看工具栈。源码审查报告把生态分成六层。最上面是算法层：奖励、优势、损失、critic、教师和更新规则，"
        "这是你的研究所在。下面是数据与运行层：rollout、环境、多轮轨迹、buffer、资源布局和策略版本。"
        "verl、OpenRLHF、SkyRL、AReaL、Miles、ROLL 和 NeMo RL 真正竞争的，正是这一层。"),
        add=_band(0, *_layers[0]) + _band(1, *_layers[1])),
    Beat(tr(
        "Next is model execution: forward and backward passes, state sharding, and tensor, pipeline, expert and "
        "context parallelism, provided by FSDP2, DeepSpeed, Megatron-Core or TorchTitan. Beside it, but separate, "
        "is the generation layer. vLLM and SGLang handle batching, the KV cache, sampling, and hot weight updates. "
        "They never run a backward pass.",
        "再往下是模型与训练执行层：前向、反向、状态分片，以及张量、流水线、专家和上下文并行，由 FSDP2、DeepSpeed、"
        "Megatron-Core 或 TorchTitan 提供。和它并列但相互独立的是推理生成层：vLLM 和 SGLang 负责 batching、"
        "KV cache、采样和权重热更新，它们从不做反向传播。"),
        add=_band(2, *_layers[2]) + _band(3, *_layers[3])),
    Beat(tr(
        "At the bottom are parameterization and kernels, such as LoRA, quantization, fused losses and attention "
        "kernels, and the data quality and observability layer: data processing, verifiers, evaluation, logging "
        "and profiling.",
        "最底层是参数化与 kernel 层，比如 LoRA、量化、融合损失和 attention kernel；以及数据质量与观测层："
        "数据加工、验证器、评估、日志和性能分析。"),
        add=_band(4, *_layers[4]) + _band(5, *_layers[5])),
    Beat(tr(
        "Two more layers sit beside this stack: agent and environment adapters, and hosted training services such "
        "as Tinker. Easy-to-use toolchains like LLaMA-Factory, ms-swift, Axolotl and Unsloth cut across several "
        "layers at once. The first rule of composition follows directly. A training backend cannot replace a "
        "rollout loop, and an inference engine cannot replace a trainer. Pick one component per role, then "
        "verify the whole path.",
        "在这个栈旁边还有两层：Agent 与环境接入层，以及像 Tinker 这样的托管训练服务。LLaMA-Factory、ms-swift、"
        "Axolotl、Unsloth 这类易用工具链则同时横跨好几层。由此直接得到第一条组合规则：训练后端替代不了 rollout 闭环，"
        "推理引擎也替代不了训练器。每个角色选一个部件，然后验证整条路径是否贯通。"),
        add=[
            Box(1320, 186, 500, 250, title=tr("Beside the stack", "栈旁的两层"),
                body=tr("Agent & env: rLLM · Agent Lightning · ART · verifiers · NeMo Gym · OpenEnv · Atropos\nHosted: **Tinker**",
                        "Agent 与环境：rLLM · Agent Lightning · ART · verifiers · NeMo Gym · OpenEnv · Atropos\n托管服务：**Tinker**"),
                tcolor=VIOLET, bar=VIOLET, bsize=24),
            Box(1320, 456, 500, 200, title=tr("Cross-layer toolchains", "跨层工具链"),
                body=tr("LLaMA-Factory · ms-swift · Axolotl · Unsloth bundle model, data, PEFT and trainer",
                        "LLaMA-Factory · ms-swift · Axolotl · Unsloth 打包了模型、数据、PEFT 与训练器"),
                tcolor=TEAL, bar=TEAL, bsize=24),
            Box(1320, 676, 500, 236, title=tr("Composition rule #1", "组合规则 #1"),
                body=tr("A trainer can't replace a rollout loop; an engine never runs backward. **One component per role**, then verify the whole path.",
                        "训练器替代不了 rollout 闭环；推理引擎从不反向传播。**每个角色选一个部件**，再验证整条路径。"),
                tcolor=CORAL, bar=CORAL, bsize=24),
        ]),
]))


# =========================================================== 4. LoRA / QLoRA
def _lora_diagram(layer, d, lang):
    x0, y0 = 120, 330
    d.rounded_rectangle([x0, y0, x0 + 230, y0 + 230], 8, fill=(46, 54, 78), outline=(90, 100, 130), width=2)
    d.text((x0 + 115, y0 + 100), "W₀", font=font(52, True), fill=(200, 205, 220), anchor="mm")
    d.text((x0 + 115, y0 + 160), L(tr("frozen  d×k", "冻结  d×k"), lang), font=font(24, False, False, lang),
           fill=MUTED, anchor="mm")
    d.text((x0 + 268, y0 + 115), "+", font=font(56, True), fill=TEXT, anchor="mm")
    bx = x0 + 310
    d.rounded_rectangle([bx, y0, bx + 44, y0 + 230], 6, fill=(30, 110, 100), outline=TEAL, width=2)
    d.text((bx + 22, y0 + 115), "B", font=font(32, True), fill=TEXT, anchor="mm")
    d.text((bx + 22, y0 + 255), "d×r", font=font(22), fill=TEAL, anchor="mm")
    ax = bx + 64
    d.rounded_rectangle([ax, y0 + 93, ax + 230, y0 + 137], 6, fill=(30, 110, 100), outline=TEAL, width=2)
    d.text((ax + 115, y0 + 115), "A", font=font(32, True), fill=TEXT, anchor="mm")
    d.text((ax + 115, y0 + 165), "r×k", font=font(22), fill=TEAL, anchor="mm")
    d.text((ax + 115, y0 + 40), L(tr("trainable", "可训练"), lang), font=font(24, True, False, lang),
           fill=TEAL, anchor="mm")


SCENES.append(Scene("lora", P2, tr("Parameterization: LoRA & QLoRA arithmetic", "参数化层：LoRA 与 QLoRA 的算术"), beats=[
    Beat(tr(
        "Let's walk the layers with math, starting near the bottom. LoRA freezes the pretrained weight W-zero and "
        "learns a low-rank update: B times A, scaled by alpha over r. B starts at zero, so training begins exactly "
        "at the base model. Trainable parameters drop from d times k to r times d-plus-k. For a 4096 by 4096 "
        "projection at rank 8, that's about sixty-five thousand parameters instead of sixteen point eight million: "
        "roughly 0.4 percent.",
        "我们带着数学逐层往上走，先从底部的参数化开始。LoRA 冻结预训练权重 W0，只学习一个低秩更新：B 乘 A，"
        "再乘以 alpha 除以 r。B 初始化为零，所以训练恰好从基座模型出发。可训练参数从 d 乘 k 降到 r 乘以 d 加 k。"
        "对一个 4096 乘 4096 的投影矩阵、秩为 8 时，只有约六万五千个参数，而不是一千六百八十万，大约千分之四。"),
        add=[
            M(100, 196, r"h=W_0x+\frac{\alpha}{r}\,BAx,\qquad B_0=0", size=46),
            Custom(_lora_diagram),
            M(100, 640, r"\#\text{trainable}=r(d+k)\quad\text{vs.}\quad dk", size=40),
            T(100, 720, tr("d = k = 4096, r = 8:  65,536 vs 16,777,216  ≈ **0.39%**",
                           "d = k = 4096，r = 8：65,536 对 16,777,216，约 **0.39%**"), size=30, w=820),
        ]),
    Beat(tr(
        "Why this matters becomes clear from memory accounting. Full fine-tuning with mixed-precision Adam stores "
        "about sixteen bytes per parameter: two for bf16 weights, two for gradients, and twelve for the fp32 master "
        "copy and the two Adam moments. A 7-billion-parameter model needs about 112 gigabytes before activations. "
        "With LoRA, the frozen base costs about two bytes per parameter. QLoRA stores the base in 4-bit NormalFloat, "
        "about half a byte per parameter, which is how the QLoRA paper fine-tuned a 65-billion model on a single "
        "48-gigabyte GPU.",
        "显存账能说明它为什么重要。混合精度 Adam 的全参数微调，每个参数大约占 16 字节：bf16 权重 2 字节，梯度 2 字节，"
        "fp32 主权重和两个 Adam 动量共 12 字节。一个 70 亿参数的模型，在不算激活的情况下就需要约 112 GB。"
        "用 LoRA 时，冻结的基座每个参数约 2 字节。QLoRA 把基座存成 4 位 NormalFloat，每个参数约半个字节，"
        "QLoRA 论文正是这样在单张 48 GB 显卡上微调了 650 亿参数的模型。"),
        add=[
            T(980, 190, tr("Training state per parameter", "每个参数的训练状态"), size=28, bold=True, color=AMBER),
            M(980, 240, r"16\Psi=\underbrace{2\Psi}_{\text{bf16 }W}+\underbrace{2\Psi}_{\text{grad}}+\underbrace{12\Psi}_{\text{fp32 master},\,m,\,v}", size=38),
            T(980, 380, tr("7B model, weights + optimizer state, before activations", "7B 模型：权重与优化器状态，未计激活"), size=24, color=MUTED),
            Bars(980, 430, 840, [
                (tr("Full FT (Adam)", "全参数微调"), 112, CORAL, "≈ 112 GB"),
                (tr("LoRA, bf16 base", "LoRA，bf16 基座"), 14, AMBER, tr("≈ 14 GB + adapters", "≈ 14 GB + adapter")),
                (tr("QLoRA, NF4 base", "QLoRA，NF4 基座"), 3.8, TEAL, tr("≈ 3.5–4 GB + adapters", "≈ 3.5–4 GB + adapter")),
            ], vmax=112, size=24, lw=250),
            T(980, 650, tr("QLoRA paper: a 65B model fine-tuned on one 48 GB GPU (NF4 + double quantization + paged optimizers).",
                           "QLoRA 论文：单张 48 GB 显卡微调 65B 模型（NF4 + 双重量化 + 分页优化器）。"), size=24, w=840, color=MUTED),
        ]),
    Beat(tr(
        "In tools: PEFT injects adapters with get_peft_model, bitsandbytes supplies the 4-bit storage and 8-bit "
        "optimizers, and Unsloth fuses kernels for memory and speed. One engineering caveat: training only LoRA "
        "does not mean syncing only LoRA. In online RL, some integrations merge the adapter into the base, or "
        "gather the full weights, before pushing them to the inference engine.",
        "对应到工具：PEFT 用 get_peft_model 注入 adapter，bitsandbytes 提供 4 位存储和 8 位优化器，Unsloth "
        "用融合 kernel 节省显存、提升速度。有一个工程上的提醒：只训练 LoRA 不等于只同步 LoRA。在在线强化学习里，"
        "有些集成会先把 adapter 合并进基座，或者收集完整权重，再推送给推理引擎。"),
        add=[
            Box(100, 800, 1000, 190, title=tr("Tools", "工具"),
                body=tr("**PEFT** `get_peft_model()` injects adapters · **bitsandbytes** NF4 storage, 8-bit optimizers · **Unsloth** fused kernels · **loralib** minimal reference · **Adapters** multi-adapter fusion",
                        "**PEFT** 用 `get_peft_model()` 注入 adapter · **bitsandbytes** NF4 存储与 8 位优化器 · **Unsloth** 融合 kernel · **loralib** 最小参考实现 · **Adapters** 多 adapter 融合"),
                tcolor=TEAL, bar=TEAL, bsize=24),
            Box(1140, 800, 680, 190, title=tr("Caveat", "注意"),
                body=tr("Training only LoRA ≠ syncing only LoRA. Online RL may merge/unmerge or gather full weights for the engine.",
                        "只训练 LoRA ≠ 只同步 LoRA。在线 RL 可能先 merge/unmerge，或收集完整权重再发给推理引擎。"),
                tcolor=CORAL, bar=CORAL, bsize=24),
        ]),
]))

# =========================================================== 5. ZeRO / FSDP2 / Megatron
SCENES.append(Scene("zero", P2, tr("Execution: the sharding math", "训练执行层：分片的数学"), beats=[
    Beat(tr(
        "One layer up is execution, and ZeRO's analysis is the cleanest way to reason about it. Let psi be the "
        "parameter count, K the optimizer bytes per parameter, twelve for mixed-precision Adam, and N the number "
        "of GPUs. Plain data parallelism replicates everything: sixteen psi per GPU. ZeRO stage one shards the "
        "optimizer states, stage two also shards gradients, and stage three, which is also what FSDP2 does, shards "
        "the parameters too. For a 7.5-billion model on 64 GPUs, that's 120 gigabytes per GPU, then 31, then 17, "
        "and finally under 2.",
        "再往上一层是训练执行，ZeRO 的分析是最清晰的推理方式。设 psi 为参数量，K 为每个参数的优化器字节数，"
        "混合精度 Adam 下 K 等于 12，N 为 GPU 数。普通数据并行把一切都复制一份：每张卡 16 psi。ZeRO 第一阶段分片优化器状态，"
        "第二阶段再分片梯度，第三阶段连参数也分片，FSDP2 做的也是这一件事。对 75 亿参数、64 张卡，每卡显存从 120 GB "
        "依次降到 31、17，最后不到 2 GB。"),
        add=[
            Tbl(100, 186, [250, 300, 380], [
                [tr("Stage", "阶段"), tr("Sharded", "分片内容"), tr("Memory per GPU", "每卡显存")],
                ["DDP", tr("nothing", "不分片"), ("$", r"(2+2+K)\,\Psi", 28)],
                ["ZeRO-1", tr("optimizer states", "优化器状态"), ("$", r"4\Psi+\frac{K\Psi}{N}", 28)],
                ["ZeRO-2", tr("+ gradients", "+ 梯度"), ("$", r"2\Psi+\frac{(2+K)\Psi}{N}", 28)],
                [tr("ZeRO-3 / FSDP2", "ZeRO-3 / FSDP2"), tr("+ parameters", "+ 参数"), ("$", r"\frac{(4+K)\Psi}{N}", 28)],
            ], size=25, minh=60, pad=10),
            T(1100, 196, tr("Ψ = 7.5B, K = 12, N = 64  (ZeRO paper)", "Ψ = 7.5B，K = 12，N = 64（ZeRO 论文）"), size=26, color=MUTED),
            Bars(1100, 250, 720, [
                ("DDP", 120, CORAL, "120 GB"),
                ("ZeRO-1", 31.4, AMBER, "31.4 GB"),
                ("ZeRO-2", 16.6, BLUE, "16.6 GB"),
                ("ZeRO-3", 1.9, TEAL, "1.9 GB"),
            ], vmax=120, size=24, lw=130, bh=34, gap=30),
        ]),
    Beat(tr(
        "Three backends implement this layer. FSDP2 shards each parameter as a DTensor and keeps ordinary PyTorch "
        "control flow. DeepSpeed adds offload to CPU and NVMe, plus AutoTP, AutoEP and its hybrid engine. "
        "Megatron-Core brings tensor, pipeline, expert and context parallelism for very large dense and MoE models. "
        "Each has a contract you must respect. In FSDP2, the local shard is not the full tensor. In DeepSpeed, the "
        "engine owns gradient-accumulation boundaries. In Megatron, the pipeline schedule owns the backward pass.",
        "三个后端实现了这一层。FSDP2 把每个参数切成 DTensor，同时保留普通的 PyTorch 控制流。DeepSpeed 增加了 CPU 与 "
        "NVMe offload，还有 AutoTP、AutoEP 和 Hybrid Engine。Megatron-Core 为超大的稠密模型和 MoE 提供张量、流水线、"
        "专家与上下文并行。每个后端都有必须遵守的契约：在 FSDP2 里，本地分片不是完整张量；在 DeepSpeed 里，"
        "梯度累积的边界归引擎管理；在 Megatron 里，反向传播由流水线调度器掌控。"),
        add=[
            Box(100, 632, 555, 186, title="FSDP2",
                body=tr("Per-parameter DTensor sharding via `fully_shard`; plain PyTorch control flow.\n**Contract:** local shard ≠ `full_tensor()` (a collective).",
                        "通过 `fully_shard` 逐参数 DTensor 分片；保留普通 PyTorch 控制流。\n**契约：**本地分片 ≠ `full_tensor()`（后者是集合通信）。"),
                tcolor=TEAL, bar=TEAL, bsize=22, tsize=28, pad=18),
            Box(682, 632, 555, 186, title="DeepSpeed",
                body=tr("ZeRO-1/2/3 + CPU/NVMe offload; AutoTP, AutoEP, Hybrid Engine.\n**Contract:** the engine owns accumulation & step boundaries.",
                        "ZeRO-1/2/3 + CPU/NVMe offload；AutoTP、AutoEP、Hybrid Engine。\n**契约：**梯度累积与 step 边界归引擎管理。"),
                tcolor=BLUE, bar=BLUE, bsize=22, tsize=28, pad=18),
            Box(1264, 632, 556, 186, title="Megatron-Core",
                body=tr("TP · PP · EP · CP for large dense & MoE models; Bridge recipes.\n**Contract:** the pipeline schedule owns micro-batches & backward.",
                        "为大稠密模型与 MoE 提供 TP · PP · EP · CP；Bridge 提供配方。\n**契约：**微批与反向传播由流水线调度器掌控。"),
                tcolor=VIOLET, bar=VIOLET, bsize=22, tsize=28, pad=18),
        ]),
    Beat(tr(
        "Composition rules. FSDP2 and ZeRO-3 are alternative ways to manage the same model's state, so don't wrap "
        "one inside the other. Ray orchestrates processes and resources, while NCCL moves tensors; they live at "
        "different layers. And a classic bug: dividing the loss by the accumulation steps yourself while the engine "
        "scales it again, which shrinks your gradients twice.",
        "几条组合规则。FSDP2 和 ZeRO-3 是管理同一个模型状态的两种替代方案，不要互相嵌套。Ray 负责进程与资源编排，"
        "NCCL 负责张量通信，它们属于不同层次。还有一个经典 bug：你自己已经把 loss 除以累积步数，引擎又缩放一次，"
        "梯度就被缩小了两次。"),
        add=[
            Box(100, 834, 960, 166, title=tr("Composition rules", "组合规则"),
                body=tr("FSDP2 vs ZeRO-3: alternatives, never double-wrap\nRay = processes & resources; NCCL = tensors\nBug: loss ÷ GA steps by you **and** by the engine → gradients shrunk twice",
                        "FSDP2 与 ZeRO-3 二选一，不要双重包裹\nRay 管进程与资源；NCCL 管张量\nBug：你和引擎**各除一次**累积步数 → 梯度被缩小两次"),
                tcolor=AMBER, bar=AMBER, bsize=22, tsize=26, pad=16, lh=1.3),
        ]),
    Beat(tr(
        "And a correction worth stating plainly. DeepSpeed, DeepSpeed-Chat and OpenRLHF are three different things. "
        "DeepSpeed is execution infrastructure. DeepSpeed-Chat is one classic three-stage RLHF application built on "
        "it. OpenRLHF is a Ray pipeline that still trains with DeepSpeed. With identical math, sharding changes the "
        "cost of training, not the objective being learned.",
        "还有一处值得明确的纠正：DeepSpeed、DeepSpeed-Chat 和 OpenRLHF 是三样不同的东西。DeepSpeed 是执行基础设施；"
        "DeepSpeed-Chat 是构建在它之上的一个经典三阶段 RLHF 应用；OpenRLHF 是一条 Ray 流水线，训练仍然由 DeepSpeed 执行。"
        "在数学完全一致的前提下，分片改变的是训练成本，而不是学习目标。"),
        add=[
            Box(1080, 834, 740, 166, title=tr("DeepSpeed ≠ DeepSpeed-Chat ≠ OpenRLHF", "DeepSpeed ≠ DeepSpeed-Chat ≠ OpenRLHF"),
                body=tr("infra · a classic SFT→RM→PPO app · a Ray pipeline on DeepSpeed. Sharding changes **cost**, not the objective.",
                        "执行底座 · 经典 SFT→RM→PPO 应用 · 基于 DeepSpeed 的 Ray 流水线。分片改变**成本**，不改变目标。"),
                tcolor=CORAL, bar=CORAL, tsize=26, bsize=22, pad=16, lh=1.3),
        ]),
]))


# =========================================================== 6. generation layer
def _sync_diagram(layer, d, lang):
    d.rounded_rectangle([990, 200, 1290, 330], 14, fill=(20, 40, 70), outline=BLUE, width=2)
    d.text((1140, 236), L(tr("Trainer", "训练端"), lang), font=font(30, True, False, lang), fill=BLUE, anchor="mm")
    d.text((1140, 286), "FSDP shards / TP×PP", font=font(22), fill=MUTED, anchor="mm")
    d.rounded_rectangle([1520, 200, 1820, 330], 14, fill=(16, 50, 46), outline=TEAL, width=2)
    d.text((1670, 236), L(tr("Engine", "推理引擎"), lang), font=font(30, True, False, lang), fill=TEAL, anchor="mm")
    d.text((1670, 286), "inference TP · KV cache", font=font(22), fill=MUTED, anchor="mm")
    d.line([1295, 265, 1505, 265], fill=AMBER, width=5)
    d.polygon([(1515, 265), (1497, 254), (1497, 276)], fill=AMBER)
    d.text((1405, 228), L(tr("reshard · cast", "重分片 · 转换"), lang), font=font(20, False, False, lang), fill=AMBER, anchor="mm")
    d.text((1405, 302), L(tr("transfer · load", "传输 · 加载"), lang), font=font(20, False, False, lang), fill=AMBER, anchor="mm")


SCENES.append(Scene("generation", P2, tr("Generation engines & the weight-sync boundary", "推理生成层与权重同步边界"), beats=[
    Beat(tr(
        "The generation layer has different physics. Prefill processes the prompt in parallel and is compute-bound. "
        "Decode emits one token per step, and every step must re-read the weights and the KV cache from memory, "
        "so it's bandwidth-bound. That's why engines like vLLM and SGLang exist: paged KV cache, continuous batching "
        "and prefix sharing let many sequences share each weight read. With long reasoning traces, this "
        "autoregressive phase is frequently the largest part of an RL step.",
        "推理生成层遵循另一套物理规律。Prefill 并行处理整段提示，是计算受限的。Decode 每一步只产生一个 token，"
        "而每一步都要把权重和 KV cache 从显存重新读一遍，所以是带宽受限的。这就是 vLLM、SGLang 这类引擎存在的理由："
        "分页 KV cache、连续批处理和前缀共享，让许多序列分摊每一次权重读取。在长推理轨迹下，这个自回归阶段常常是 RL 单步里最大的开销。"),
        add=[
            Box(100, 190, 410, 190, title=tr("Prefill", "Prefill 预填充"),
                body=tr("all prompt tokens in parallel → **compute-bound**", "并行处理全部提示 token → **计算受限**"),
                tcolor=BLUE, bar=BLUE, bsize=25),
            Box(530, 190, 410, 190, title=tr("Decode", "Decode 解码"),
                body=tr("one token per step; re-reads weights + KV → **bandwidth-bound**", "每步一个 token；反复读权重与 KV → **带宽受限**"),
                tcolor=CORAL, bar=CORAL, bsize=25),
            M(100, 420, r"t_{\text{step}}\approx\frac{|W|+\sum_{i\in\text{batch}}|\mathrm{KV}_i|}{\text{HBM bandwidth}}", size=40),
            T(100, 540, tr("Engines amortize each weight read across many sequences: paged KV cache, continuous batching, prefix sharing (**vLLM**, **SGLang**).",
                           "引擎让许多序列分摊每一次权重读取：分页 KV cache、连续批处理、前缀共享（**vLLM**、**SGLang**）。"), size=26, w=840, color=MUTED),
        ]),
    Beat(tr(
        "The trainer and the engine also lay out the same weights differently: FSDP shards or Megatron partitions on "
        "one side, inference tensor parallelism on the other. So every policy update means resharding, possibly "
        "converting precision, transferring and reloading. vLLM now separates the weight source, the trainer-side "
        "transfer engine and the inference-side receiver. SGLang supports updates from disk, tensors or a "
        "distributed group, with pause and resume. Weight sync is not plumbing; it's a protocol boundary.",
        "训练端和推理引擎对同一份权重的布局也不同：一边是 FSDP 分片或 Megatron 切分，另一边是推理侧的张量并行。"
        "因此每次策略更新都意味着重新分片、可能的精度转换、传输和重新加载。vLLM 现在把权重来源、训练侧传输引擎和推理侧接收端分开；"
        "SGLang 支持从磁盘、张量或分布式通信组更新，并支持暂停与恢复。权重同步不是管道杂活，而是一条协议边界。"),
        add=[
            Custom(_sync_diagram),
            Bul(990, 370, 830, [
                tr("**vLLM**: WeightSource · trainer-side transfer engine · receiver (NCCL, IPC, resharding)",
                   "**vLLM**：WeightSource · 训练侧传输引擎 · 推理侧接收端（NCCL、IPC、重分片）"),
                tr("**SGLang**: from disk / tensors / dist. group; pause · resume · release memory; quantization post-processing once per update session",
                   "**SGLang**：从磁盘 / 张量 / 分布式组更新；暂停 · 恢复 · 释放显存；每个更新 session 只做一次量化后处理"),
                tr("In-flight requests: keep or clear old KV? Tokens already generated stay **old-policy** tokens.",
                   "在途请求：保留还是清空旧 KV？已生成的 token 仍然来自**旧策略**。"),
            ], size=24, gap=16),
        ]),
    Beat(tr(
        "A practical rule: anything the policy needs in order to generate must be representable in the engine and "
        "synced correctly. Learner-only parts like critics, Q-heads or target networks can stay on the training side. "
        "DeepSpeed's hybrid engine was an early answer to this tension: the same actor, switching between a training "
        "layout and a generation layout.",
        "一条实用规则：凡是策略在生成时需要用到的参数，都必须能在推理引擎中表达，并被正确同步；只用于学习的部件，"
        "比如 critic、Q head 或 target 网络，可以留在训练侧。DeepSpeed 的 Hybrid Engine 是对这一矛盾的早期回答："
        "同一个 actor，在训练布局和生成布局之间切换。"),
        add=[
            Box(100, 760, 1720, 200, title=tr("Rule", "规则"),
                body=tr("Whatever the policy needs to **generate** must be expressible in the engine and synced; learner-only parts (critic, Q-head, target) can stay in the learner.\nPrecedent: DeepSpeed Hybrid Engine (2023), one actor, two layouts.",
                        "策略在**生成**时需要的一切参数都必须能在引擎中表达并同步；只用于学习的部件（critic、Q head、target）可以留在训练端。\n先例：DeepSpeed Hybrid Engine（2023），同一个 actor，两种布局。"),
                tcolor=AMBER, bar=AMBER, bsize=26),
        ]),
]))

# =========================================================== 7. offline SFT / DPO
_chain = [("Transformers", BLUE), ("PEFT", TEAL), ("bitsandbytes", VIOLET),
          ("Accelerate · DeepSpeed · FSDP2", AMBER), ("TRL SFT / DPO Trainer", CORAL)]


def _chain_draw(layer, d, lang):
    x, y = 100, 676
    for i, (name, c) in enumerate(_chain):
        f = font(25, True)
        w = int(f.getlength(name)) + 44
        d.rounded_rectangle([x, y, x + w, y + 58], 29, fill=PANEL2, outline=c, width=2)
        d.text((x + 22, y + 13), name, font=f, fill=c)
        x += w
        if i < len(_chain) - 1:
            d.text((x + 12, y + 8), "+", font=font(34, True), fill=MUTED)
            x += 48


SCENES.append(Scene("offline", P3, tr("Offline post-training: SFT & DPO", "离线后训练：SFT 与 DPO"), beats=[
    Beat(tr(
        "Part three: offline versus online. Offline methods train on a fixed dataset. SFT minimizes masked negative "
        "log-likelihood. DPO turns a preference pair into a logistic loss on the difference of policy-to-reference "
        "log-ratios, scaled by beta. Neither needs a generation engine, and the frozen reference model's "
        "log-probabilities can even be precomputed.",
        "第三部分：离线与在线。离线方法在固定数据集上训练。SFT 最小化带 mask 的负对数似然。DPO 把一个偏好对变成逻辑回归损失，"
        "作用在策略相对参考模型的对数比之差上，再乘以 beta。两者都不需要生成引擎，冻结参考模型的对数概率甚至可以提前算好。"),
        add=[
            M(100, 192, r"\mathcal{L}_{\text{SFT}}=-\frac{1}{Z}\sum_i\sum_t m_{i,t}\,\log\pi_\theta(y_{i,t}\mid x_i,y_{i,<t})", size=40),
            M(100, 290, r"\mathcal{L}_{\text{DPO}}=-\mathbb{E}\Big[\log\sigma\Big(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\Big)\Big]", size=40),
            Chip(1340, 210, tr("no generation engine needed", "不需要生成引擎"), size=24, fill=(16, 44, 40), color=TEAL),
            Chip(1340, 280, tr("π_ref log-probs can be precomputed", "π_ref 对数概率可预先计算"), size=24, fill=(16, 44, 40), color=TEAL),
        ]),
    Beat(tr(
        "The subtle part is the normalizer Z. It looks like a detail, but it's part of the objective. Token-mean "
        "gives every token the same weight. Sequence-mean averages within each response first, so a token in a "
        "1000-token answer weighs ten times less than a token in a 100-token answer. And with gradient accumulation "
        "or multiple ranks, an average of local means is not the global mean. These aren't system knobs; they "
        "change what you optimize.",
        "微妙之处在于归一化因子 Z。它看起来像细节，实际上是目标函数的一部分。token 平均让每个 token 权重相同；"
        "序列平均先在每个回答内部求平均，于是 1000 个 token 的回答中的一个 token，权重只有 100 个 token 回答中的十分之一。"
        "而在梯度累积或多卡情况下，局部均值的平均并不等于全局均值。这些不是系统旋钮，它们改变的是你在优化什么。"),
        add=[
            T(100, 418, tr("The normalizer is part of the objective", "归一化因子是目标函数的一部分"), size=28, bold=True, color=AMBER),
            M(100, 470, r"\text{token-mean: }\ Z=\sum_i|y_i|\ \Rightarrow\ w_{\text{tok}}=\frac{1}{N_{\text{tok}}}", size=34),
            M(100, 562, r"\text{sequence-mean: }\ \frac{1}{G}\sum_i\frac{1}{|y_i|}\sum_t\ \Rightarrow\ w_{\text{tok}}=\frac{1}{G\,|y_i|}", size=34),
            Box(1060, 418, 760, 220, title=tr("Example: G = 2, |y₁| = 100, |y₂| = 1000", "例：G = 2，|y₁| = 100，|y₂| = 1000"),
                body=tr("sequence-mean: 1/200 vs 1/2000 per token (**10×** apart)\ntoken-mean: 1/1100 for every token\nmean of micro-batch means ≠ global mean",
                        "序列平均：每 token 1/200 对 1/2000（相差 **10 倍**）\ntoken 平均：每个 token 都是 1/1100\n微批均值的平均 ≠ 全局均值"),
                tcolor=AMBER, bar=AMBER, tsize=26, bsize=25),
        ]),
    Beat(tr(
        "The standard offline composition is: Transformers for the model, PEFT for adapters, bitsandbytes when memory "
        "is tight, Accelerate, DeepSpeed or FSDP2 for scale, and TRL's SFT or DPO trainer for the objective and "
        "lifecycle. LLaMA-Factory, ms-swift and Axolotl package the same pieces as config-driven recipes, and "
        "Unsloth focuses on memory and speed.",
        "标准的离线组合是：Transformers 提供模型，PEFT 提供 adapter，显存紧张时加上 bitsandbytes，用 Accelerate、"
        "DeepSpeed 或 FSDP2 扩展规模，再用 TRL 的 SFT 或 DPO 训练器承载目标函数和训练生命周期。LLaMA-Factory、"
        "ms-swift 和 Axolotl 把同样的部件封装成配置驱动的配方，Unsloth 则专注于显存和速度。"),
        add=[
            Custom(_chain_draw),
            T(100, 752, tr("Recipes: **LLaMA-Factory** · **ms-swift** · **Axolotl**     Memory/speed: **Unsloth**     JAX/TPU: **Tunix**",
                           "配方平台：**LLaMA-Factory** · **ms-swift** · **Axolotl**     显存/速度：**Unsloth**     JAX/TPU：**Tunix**"), size=25, color=MUTED, w=1720),
        ]),
    Beat(tr(
        "One efficiency boundary from the code review: in the TRL snapshot reviewed, a custom compute-loss function "
        "runs a full-logits forward pass, plus an extra no-gradient fused pass for metrics, while the default path "
        "can use a fused LM head. So when you change the loss, re-check memory and throughput instead of inheriting "
        "the defaults.",
        "源码审查发现了一个效率边界：在审查的 TRL 快照中，自定义 compute_loss_func 会走完整 logits 的前向，"
        "还要额外做一次无梯度的融合前向来计算指标；而默认路径可以直接使用融合的 LM head。所以换了损失函数后，"
        "要重新核对显存和吞吐，不能直接沿用默认路径的预期。"),
        add=[
            Box(100, 830, 1720, 160, title=tr("Code-review finding · TRL @72cf37d", "源码审查发现 · TRL @72cf37d"),
                body=tr("custom `compute_loss_func` → full-logits forward + extra no-grad fused forward for metrics;  default → fused LM head.  **Changing the loss can change memory & throughput.**",
                        "自定义 `compute_loss_func` → 完整 logits 前向 + 额外一次无梯度融合前向算指标；默认路径 → 融合 LM head。**换损失可能改变显存与吞吐。**"),
                tcolor=CORAL, bar=CORAL, bsize=25),
        ]),
]))

# =========================================================== 8. online RL loop
_nodes = [
    (tr("Prompts / tasks", "提示 / 任务"), "", (505, 225), MUTED),
    (tr("Rollout engine", "Rollout 引擎"), "vLLM · SGLang", (810, 345), TEAL),
    (tr("Reward · env · verifier", "奖励 · 环境 · 验证器"), "", (810, 545), AMBER),
    (tr("Advantage", "优势估计"), "GAE · group norm", (505, 665), VIOLET),
    (tr("Learner update", "学习器更新"), "FSDP2 · DeepSpeed · Megatron", (200, 545), BLUE),
    (tr("Publish weights", "发布权重"), tr("version v+1", "版本 v+1"), (200, 345), CORAL),
]


def _loop_draw(layer, d, lang):
    import math
    cx, cy = 505, 445
    hw, hh = 150, 46

    def edge(dx, dy):
        tx = hw / abs(dx) if dx else 1e9
        ty = hh / abs(dy) if dy else 1e9
        return min(tx, ty)

    for i in range(len(_nodes)):
        a = _nodes[i][2]
        b = _nodes[(i + 1) % len(_nodes)][2]
        dist = math.dist(a, b)
        dx, dy = (b[0] - a[0]) / dist, (b[1] - a[1]) / dist
        t0, t1 = edge(dx, dy) + 6, edge(dx, dy) + 10
        sx, sy = a[0] + t0 * dx, a[1] + t0 * dy
        ex, ey = b[0] - t1 * dx, b[1] - t1 * dy
        d.line([(sx, sy), (ex, ey)], fill=(110, 122, 150), width=4)
        Arrow._head(d, (sx, sy), (ex, ey), (110, 122, 150), 16)
    for name, sub, (x, y), c in _nodes:
        d.rounded_rectangle([x - hw, y - hh, x + hw, y + hh], 14, fill=PANEL, outline=c, width=2)
        nm = L(name, lang)
        sb = L(sub, lang)
        f = font(24, True, False, lang)
        if sb:
            d.text((x, y - 13), nm, font=f, fill=c, anchor="mm")
            d.text((x, y + 19), sb, font=font(17, False, False, lang), fill=MUTED, anchor="mm")
        else:
            d.text((x, y), nm, font=f, fill=c, anchor="mm")
    d.text((cx, cy), L(tr("one RL step", "一个 RL 步"), lang), font=font(26, True, False, lang), fill=(90, 102, 132), anchor="mm")


def _track(x, sym, title, body, c):
    return [
        Box(x, 760, 415, 230, fill=PANEL, border=c, bar=None, radius=14),
        M(x + 24, 778, sym, size=40, color=c),
        T(x + 24, 840, title, size=25, bold=True, color=c, w=370),
        T(x + 24, 880, body, size=22, color=MUTED, w=370),
    ]


SCENES.append(Scene("online", P3, tr("Online RL: the loop and its objectives", "在线 RL：闭环与目标函数"), beats=[
    Beat(tr(
        "Online RL closes a loop. Sample prompts, generate rollouts with the inference engine, score them with a "
        "reward model, verifier or environment, estimate advantages, update the learner, publish new weights to the "
        "engine, and repeat. Every arrow in this loop is a place where a framework makes design decisions.",
        "在线强化学习是一个闭环：取提示，用推理引擎生成 rollout，用奖励模型、验证器或环境打分，估计优势，更新学习器，"
        "把新权重发布给推理引擎，然后循环。这个闭环里的每一条箭头，都是框架做设计决策的地方。"),
        add=[Custom(_loop_draw)]),
    Beat(tr(
        "The objectives are compact. PPO maximizes a clipped surrogate: the probability ratio between the current "
        "and old policy, times the advantage, clipped to one plus or minus epsilon. GRPO removes the critic. It "
        "samples a group of G responses per prompt and standardizes each reward against the group mean and standard "
        "deviation, with an optional KL penalty toward a reference model.",
        "目标函数本身很紧凑。PPO 最大化一个截断代理目标：当前策略与旧策略的概率比乘以优势，并把比率截断在一加减 epsilon 之内。"
        "GRPO 去掉了 critic：对每个提示采样一组 G 个回答，用组内均值和标准差标准化每个奖励，并可选地加上对参考模型的 KL 惩罚。"),
        add=[
            T(1010, 196, "PPO", size=30, bold=True, color=AMBER),
            M(1010, 246, r"J=\mathbb{E}_t\Big[\min\big(\rho_t\hat A_t,\ \mathrm{clip}(\rho_t,1\!-\!\epsilon,1\!+\!\epsilon)\,\hat A_t\big)\Big]", size=34, maxw=810),
            M(1010, 330, r"\rho_t=\frac{\pi_\theta(a_t\mid h_t)}{\pi_{\text{old}}(a_t\mid h_t)}", size=34),
            T(1010, 440, "GRPO", size=30, bold=True, color=AMBER),
            M(1010, 490, r"\hat A_i=\frac{r_i-\mathrm{mean}(r_{1:G})}{\mathrm{std}(r_{1:G})+\delta}", size=34),
            M(1010, 590, r"\text{optional: }-\beta\,\mathrm{KL}\big(\pi_\theta\,\|\,\pi_{\text{ref}}\big)", size=32),
        ]),
    Beat(tr(
        "Here's what separates careful implementations. There are four probability tracks. Mu is the behavior "
        "distribution the engine actually sampled from. Pi-old is the trainer's recomputation at the start of the "
        "step. Pi-theta is the current, differentiable policy. And pi-ref is the KL anchor, or a teacher in "
        "distillation. The true importance weight factorizes into the PPO ratio times a train-inference mismatch "
        "term. OpenRLHF, AReaL and others keep these tracks separate, and truncated importance sampling caps the "
        "mismatch term.",
        "真正区分细致实现的，是下面这一点：一共有四条概率轨道。mu 是推理引擎实际采样时的行为分布；pi old 是训练器在这一步开始时的重算；"
        "pi theta 是当前可微的策略；pi ref 是 KL 的锚点，在蒸馏里则是教师。真实的重要性权重可以分解为 PPO 比率乘以一个训练推理失配项。"
        "OpenRLHF、AReaL 等框架会把这几条轨道分开保存，截断重要性采样则给失配项设一个上限。"),
        add=_track(100, r"\mu", tr("behavior", "行为策略"), tr("what the engine actually sampled (rollout logprobs)", "引擎实际采样的分布（rollout logprob）"), BLUE)
        + _track(535, r"\pi_{\text{old}}", tr("old / proximal", "旧策略"), tr("trainer recompute at step start, frozen", "步首由训练端重算，固定不变"), TEAL)
        + _track(970, r"\pi_\theta", tr("current", "当前策略"), tr("differentiable, being updated", "可微，正在被更新"), AMBER)
        + _track(1405, r"\pi_{\text{ref}},\ \pi_{\text{T}}", tr("anchor / teacher", "参考 / 教师"), tr("KL anchor or distillation signal", "KL 锚点或蒸馏信号"), VIOLET)
        + [M(1010, 650, r"\frac{\pi_\theta}{\mu}=\underbrace{\frac{\pi_\theta}{\pi_{\text{old}}}}_{\text{PPO ratio}}\cdot\underbrace{\frac{\pi_{\text{old}}}{\mu}}_{\text{mismatch}}\ \ \to\ \ \text{TIS: }\min\Big(\frac{\pi_{\text{old}}}{\mu},C\Big)", size=28, maxw=810)]),
]))


# =========================================================== 9. step time
def _gantt(layer, d, lang):
    x0, y0, s = 260, 230, 0.78
    f = font(22, True, False, lang)
    d.text((100, y0 + 10), L(tr("sequential", "串行"), lang), font=f, fill=MUTED)
    segs = [("gen", 640, BLUE), ("rew", 60, AMBER), ("train", 160, TEAL), ("sync", 50, CORAL)]
    x = x0
    for nm, w, c in segs:
        d.rounded_rectangle([x, y0, x + w * s, y0 + 44], 6, fill=c)
        if w > 100:
            d.text((x + 10, y0 + 9), nm, font=font(22, True), fill=(10, 15, 30))
        x += w * s + 3
    y1 = y0 + 90
    d.text((100, y1 + 30), L(tr("pipelined", "流水化"), lang), font=f, fill=MUTED)
    d.rounded_rectangle([x0, y1, x0 + 700 * s, y1 + 40], 6, fill=BLUE)
    d.text((x0 + 10, y1 + 7), "gen + rew  (k+1)", font=font(22, True), fill=(10, 15, 30))
    d.rounded_rectangle([x0, y1 + 48, x0 + 160 * s, y1 + 88], 6, fill=TEAL)
    d.text((x0 + 10, y1 + 55), "train k", font=font(22, True), fill=(10, 15, 30))
    d.rounded_rectangle([x0 + 700 * s + 3, y1, x0 + 750 * s, y1 + 88], 6, fill=CORAL)


SCENES.append(Scene("steptime", P4, tr("Step time: find the bottleneck first", "单步时间：先找到瓶颈"), beats=[
    Beat(tr(
        "Part four is where systems really differ. Write down the time of one synchronous step: generation, reward, "
        "training and weight sync, all in sequence. With pipelining on independent resources, the ideal step time "
        "approaches the slower of generation and training, plus sync. Amdahl's law sets the priority. If generation "
        "is 80 percent of the step, making training twice as fast saves only ten percent.",
        "第四部分讲系统真正拉开差距的地方。先写出一次同步单步的时间：生成、奖励、训练和权重同步依次进行。"
        "在资源相互独立的理想流水化下，单步时间趋近于生成与训练中较慢的那一个，再加上同步。Amdahl 定律决定了优先级："
        "如果生成占单步的 80%，把训练加速一倍也只能省下 10%。"),
        add=[
            Custom(_gantt),
            M(100, 450, r"T^{\text{seq}}=T_{\text{gen}}+T_{\text{rew}}+T_{\text{train}}+T_{w}", size=36),
            M(100, 515, r"T^{\text{pipe}}\approx\max\big(T_{\text{gen}}+T_{\text{rew}},\ T_{\text{train}}\big)+T_{w}", size=36),
            Box(1060, 200, 760, 210, title=tr("Amdahl check", "Amdahl 检验"),
                body=tr("generation 80%, training 20%, train 2× faster:", "生成 80%，训练 20%，训练加速 2 倍："),
                tcolor=AMBER, bar=AMBER, bsize=25),
            M(1090, 320, r"0.8+\frac{0.2}{2}=0.90\ \Rightarrow\ \text{only }10\%\text{ saved}", size=36, color=AMBER),
        ]),
    Beat(tr(
        "Published numbers confirm that async is not automatically faster. In NeMo RL's performance table, "
        "DeepSeek-V3 on 512 H100s dropped from 111 to about 64 seconds per step with one-step-off async. But Qwen3 "
        "30B-A3B on 16 GB200s went from 221 to 301 seconds: splitting resources made it slower. And Miles cut a "
        "trillion-parameter Kimi K2 weight update from 53 seconds with NCCL broadcast to about 7 seconds with "
        "peer-to-peer transfer, which shows weight sync can be a first-class bottleneck.",
        "公开数据也说明异步并不自动更快。在 NeMo RL 的性能表里，DeepSeek-V3 在 512 张 H100 上，one-step-off 异步把单步从 111 秒降到约 64 秒；"
        "但 Qwen3-30B-A3B 在 16 张 GB200 上，却从 221 秒变成 301 秒，资源拆分反而让它更慢。Miles 则把万亿参数 Kimi K2 的一次权重更新，"
        "从 NCCL 广播的 53 秒降到点对点传输的约 7 秒，说明权重同步本身就可能是头等瓶颈。"),
        add=[
            Tbl(100, 600, [190, 660, 520, 350], [
                [tr("Case", "案例"), tr("Setup", "配置"), tr("Result", "结果"), tr("Lesson", "说明")],
                ["NeMo RL", "DeepSeek-V3 · 512×H100 · BF16 · seq 1536", tr("sync 111 s → one-step-off **64.1 s**/step", "同步 111 秒 → one-step-off **64.1 秒**/步"), tr("overlap helps here", "此处重叠有效")],
                ["NeMo RL", "Qwen3-30B-A3B · 16×GB200 · 4K · gen batch 2048", tr("sync 221 s → async **301 s**/step", "同步 221 秒 → 异步 **301 秒**/步"), tr("async can be slower", "异步可能更慢")],
                ["Miles", "Kimi K2 1T-A32B · 32+32 H100 nodes · 1 GB buckets", tr("NCCL broadcast 53.28 s → P2P **7.23 s**", "NCCL 广播 53.28 秒 → P2P **7.23 秒**"), tr("weight-update phase only", "仅权重更新阶段")],
            ], size=21, minh=50, pad=10),
            T(100, 812, tr("Official numbers: per-step (≈5-step average) or per-phase. Not convergence speed, not end-to-end speedups.",
                           "官方数据：单步（约 5 步平均）或单阶段耗时，不代表收敛速度，也不是端到端加速比。"), size=21, color=MUTED, w=1720),
        ]),
    Beat(tr(
        "So the metric that matters is cost to target: total GPU-hours, across all roles, to reach the same "
        "validation score. Tokens per second can reward long, useless generations; steps per second can hide "
        "smaller batches; wall time can hide extra GPUs. Profile each phase, and fix the biggest one first.",
        "所以真正重要的指标是达到目标的成本：所有角色加起来，达到同一验证分数所消耗的 GPU 小时。只看每秒 token 数，"
        "可能奖励了又长又无效的生成；只看每秒步数，可能掩盖了更小的 batch；只看墙钟时间，可能掩盖了用了更多 GPU。"
        "把每个阶段分别 profile，先解决最大的那个。"),
        add=[
            M(100, 868, r"\text{Cost-to-target}=\textstyle\sum_{r\,\in\,\text{roles}}\text{GPUs}_r\times\text{hours}_r\ \ \text{(until the validation target is reached)}", size=32, color=TEAL),
            T(100, 940, tr("Log separately: rollout · reward/env · policy/ref/value fwd · bwd/optimizer · weight sync · idle · peak memory",
                           "分别记录：rollout · 奖励/环境 · policy/ref/value 前向 · 反向/优化器 · 权重同步 · 空闲 · 峰值显存"), size=22, color=MUTED, w=1720),
        ]),
]))


# =========================================================== 10. async
def _span_draw(layer, d, lang):
    x, y = 1180, 210
    d.text((x, y), L(tr("one trajectory, mid-flight update", "一条轨迹，生成中途更新权重"), lang), font=font(24, True, False, lang), fill=MUTED)
    d.rounded_rectangle([x, y + 44, x + 330, y + 94], 6, fill=(40, 80, 130))
    d.text((x + 165, y + 69), "tokens @ v3", font=font(24, True), fill=TEXT, anchor="mm")
    d.rounded_rectangle([x + 334, y + 44, x + 630, y + 94], 6, fill=(30, 120, 105))
    d.text((x + 482, y + 69), "tokens @ v4", font=font(24, True), fill=TEXT, anchor="mm")
    d.text((x, y + 108), L(tr("→ version belongs to token spans", "→ 版本应绑定到 token 片段"), lang), font=font(24, False, False, lang), fill=AMBER)


SCENES.append(Scene("async", P4, tr("Asynchrony is not a boolean", "异步不是一个布尔开关"), beats=[
    Beat(tr(
        "Asynchrony introduces staleness. Define a sample's age as the learner's current version minus the version "
        "that generated it. With mid-flight weight updates, one trajectory can even span several versions, so the "
        "version belongs to token spans, not just to samples. A strict gate only consumes data with age at most k, "
        "and importance weights, usually truncated, correct the remaining shift.",
        "异步带来了数据陈旧。把样本的年龄定义为学习器当前版本减去生成它的版本。如果生成中途更新了权重，一条轨迹甚至会跨越多个版本，"
        "所以版本号应该绑定到 token 片段，而不只是绑定到样本。严格的门限只消费年龄不超过 k 的数据，剩下的分布偏移再用重要性权重修正，"
        "通常还会做截断。"),
        add=[
            M(100, 200, r"\Delta v=v_{\text{learner}}-v_{\text{behavior}}(\text{token span}),\qquad\text{consume iff }\Delta v\le k", size=34, maxw=1040),
            M(100, 280, r"w_t=\min\Big(\frac{\pi_\theta(a_t\mid h_t)}{\mu_t(a_t\mid h_t)},\,C\Big)", size=36),
            Custom(_span_draw),
        ]),
    Beat(tr(
        "The key insight from the code review is that async is not a boolean. A bounded queue limits how far producers "
        "run ahead, but not the age of each sample when it's trained on. An admission window decides whether new tasks "
        "start, but a slow in-flight task can still finish stale. A consume-time gate filters by version, but dropping "
        "samples changes the data distribution. Importance correction fixes known probability shifts, but not for free "
        "in variance. And mid-flight updates buy overlap at the cost of mixed-policy trajectories.",
        "源码审查最重要的结论之一是：异步不是一个布尔开关。有界队列限制生产者领先多少，但不限制每条样本被训练时的年龄。"
        "准入窗口决定新任务能否启动，但一个很慢的在途任务完成时仍可能已经过时。消费时的年龄门限按版本过滤，但丢弃样本会改变数据分布。"
        "重要性修正能处理已知概率的偏移，但要付出方差代价。生成中途更新权重换来了重叠，代价是轨迹混入了多个策略。"),
        add=[
            Tbl(100, 380, [420, 600, 700], [
                [tr("Mechanism", "机制"), tr("What it constrains", "它实际约束什么"), tr("What it does NOT guarantee", "它不保证什么")],
                [tr("Bounded queue / back-pressure", "有界队列 / 背压"), tr("pending count, producer lead", "待消费数量、生产者领先程度"), tr("per-sample age at training time", "每条样本训练时的版本年龄")],
                [tr("Admission window", "准入窗口"), tr("whether new tasks may start", "新任务能否启动"), tr("slow in-flight tasks finish fresh", "慢的在途任务完成时仍足够新")],
                [tr("Consume-time age gate", "消费时年龄门限"), tr("filter / wait by version before training", "训练前按版本过滤或等待"), tr("unchanged data distribution after drops", "丢弃后数据分布不变")],
                [tr("Importance correction", "重要性修正"), tr("known-probability distribution shift", "已知概率的分布偏移"), tr("unbiased, low-variance reuse of any old data", "任意旧数据复用都无偏且低方差")],
                [tr("Mid-flight weight update", "生成中途更新权重"), tr("generation / training overlap", "生成与训练的重叠程度"), tr("one trajectory = one fixed policy", "一条轨迹来自同一个固定策略")],
            ], size=23, minh=58),
        ]),
    Beat(tr(
        "Real frameworks implement different subsets. OpenRLHF mainly uses queue back-pressure. SkyRL's staleness "
        "setting governs admission and capacity. Miles and prime-rl can filter by age at consumption, and verl offers "
        "drop or wait. A staleness value of two may not mean the same age in two frameworks. And if long trajectories "
        "are discarded, hard problems quietly disappear from training. Measure age, length and difficulty explicitly.",
        "真实框架实现的是不同的子集：OpenRLHF 主要靠队列背压；SkyRL 的陈旧度配置作用于准入和总容量；Miles 和 prime-rl "
        "可以在消费时按年龄过滤；verl 提供 drop 和 wait 两种策略。同样配置成 2，在两个框架里不一定代表同样的年龄。"
        "而如果长轨迹被丢弃，难题就会悄悄从训练中消失。要按年龄、长度和难度分层，显式地度量。"),
        add=[
            Chip(100, 800, tr("OpenRLHF: queue/slot back-pressure", "OpenRLHF：队列/slot 背压"), size=22, fill=PANEL2),
            Chip(540, 800, tr("SkyRL: admission + capacity", "SkyRL：准入 + 容量"), size=22, fill=PANEL2),
            Chip(905, 800, tr("Miles: consume-time age filter", "Miles：消费时年龄过滤"), size=22, fill=PANEL2),
            Chip(1300, 800, tr("prime-rl: hard drop-stale", "prime-rl：出队前硬过滤"), size=22, fill=PANEL2),
            Chip(100, 862, tr("verl V1: drop | wait", "verl V1：drop | wait"), size=22, fill=PANEL2),
            Chip(400, 862, tr("NeMo RL: age-filtered replay path", "NeMo RL：按年龄筛选的 replay 路径"), size=22, fill=PANEL2),
            T(100, 935, tr("⚠ Same 'staleness = 2' ≠ same age across frameworks  ·  dropping slow trajectories quietly removes hard problems",
                           "⚠ 同样配置为 2 ≠ 同样的年龄  ·  丢弃慢轨迹会悄悄移除难题"), size=25, color=CORAL, w=1720),
        ]),
]))

# =========================================================== 11. consistency
SCENES.append(Scene("consistency", P4, tr("Train–inference consistency: five error sources", "训练—推理一致性：五个误差来源"), beats=[
    Beat(tr(
        "Consistency between training and inference has at least five independent sources of error. Re-tokenizing "
        "text can change the token sequence, so store the exact generated IDs. The behavior policy may be stale, so "
        "record versions and correct for it. MoE routers may pick different experts in the trainer, so replay the "
        "rollout-time routes. Top-p or top-k sampling truncates the distribution. And different kernels, precision or "
        "batch shapes produce different log-probs for the same weights. Each mechanism fixes one source; none "
        "substitutes for another.",
        "训练与推理之间的一致性，至少有五个相互独立的误差来源。把文本重新分词可能改变 token 序列，所以要保存实际生成的 token ID。"
        "行为策略可能已经陈旧，所以要记录版本并做修正。MoE 路由器在训练端可能选中不同的专家，所以要重放 rollout 时的路由。"
        "top-p 或 top-k 采样截断了分布。不同的 kernel、精度或 batch 形状，也会让同一份权重算出不同的对数概率。"
        "每种机制只修复一个来源，彼此不能替代。"),
        add=[
            Tbl(100, 190, [430, 690, 600], [
                [tr("Error source", "误差来源"), tr("What to do", "需要解决什么"), tr("Representative mechanism", "代表性机制")],
                [tr("Re-tokenization drift", "重新分词导致 token 变化"), tr("keep exact generated token IDs, masks, logprobs", "保存真实生成的 token、动作掩码与 logprob"), "Miles TITO · rLLM trace capture"],
                [tr("Stale behavior policy", "采样策略比训练策略旧"), tr("record versions, bound staleness, importance correction", "记录版本、限制陈旧度、重要性修正"), tr("AReaL · NeMo RL async", "AReaL · NeMo RL 异步")],
                [tr("MoE routing differs", "MoE 专家路由不同"), tr("replay rollout-time expert routes", "训练时重放采样时的路由"), tr("Miles R3 · SkyRL · NeMo RL routing replay", "Miles R3 · SkyRL · NeMo RL 路由重放")],
                [tr("top-p / top-k truncation", "top-p / top-k 截断"), tr("normalize on the actual sampling support", "在实际采样支持集上归一化"), tr("SkyRL sampler-support replay", "SkyRL 采样支持集重放")],
                [tr("Kernels / precision / batch shape", "kernel / 精度 / batch 形状"), tr("share model definitions, batch-invariant kernels", "共享模型定义与 batch 无关的 kernel"), "TitanRL · vexact (verl ecosystem)"],
            ], size=24, minh=72),
        ]),
    Beat(tr(
        "The top-p case is a clean piece of math. Nucleus sampling draws from the smallest set S whose probability mass "
        "reaches p, renormalized by Z_S. So the true behavior log-prob is log pi minus log Z_S. If the trainer "
        "recomputes a full-vocabulary softmax, every ratio is off by a factor Z_S that varies from token to token. "
        "Sampler-support replay fixes this by normalizing over the same candidate set during training.",
        "top-p 是一个很干净的数学例子。核采样从概率质量达到 p 的最小集合 S 中抽取，并用 Z_S 重新归一化。所以真实的行为对数概率是 "
        "log pi 减去 log Z_S。如果训练端用全词表 softmax 重算，每一个比率都会偏差一个随 token 变化的因子 Z_S。"
        "采样支持集重放的做法，是在训练时也在同一个候选集合上归一化。"),
        add=[
            Rect(100, 690, 1720, 300, fill=PANEL, border=LINE, radius=16),
            M(130, 712, r"S_p=\text{smallest set with }\sum_{a\in S_p}\pi(a)\ge p,\qquad \mu(a)=\frac{\pi(a)}{Z_S}\,\mathbf{1}[a\in S_p],\qquad Z_S=\sum_{b\in S_p}\pi(b)\le 1", size=32, maxw=1660),
            M(130, 820, r"\log\mu(a)=\log\pi(a)-\log Z_S", size=40, color=AMBER),
            T(740, 830, tr("→ a full-softmax recompute is off by the factor Z(S), differently for every token",
                           "→ 全词表 softmax 重算会偏差因子 Z(S)，而且每个 token 都不同"), size=26, w=1060),
            T(130, 920, tr("Fix: normalize on the same candidate set during training (sampler-support replay).",
                           "修复：训练时在同一候选集合上归一化（采样支持集重放）。"), size=26, color=MUTED, w=1660),
        ]),
]))

# =========================================================== 12. six freedoms
SCENES.append(Scene("freedoms", P4, tr("What \"customize every detail\" really means", "“定制每个算法细节”到底意味着什么"), beats=[
    Beat(tr(
        "Now, what does it mean to customize every detail of an algorithm? There are six distinct freedoms: the "
        "objective, the estimator, the data lifetime, the model state, the update rule, and execution scheduling. "
        "Each needs a different kind of access. A custom-loss hook usually covers only part of the first row. A new "
        "critic, a target network or replay touches rows three to five, which live in the learner loop, not in a "
        "loss callback.",
        "那么，“能定制算法的每个细节”到底是什么意思？一共有六种不同的自由度：目标函数、估计器、数据寿命、模型状态、更新规则和执行调度。"
        "每一种需要的访问权限都不一样。框架提供的自定义 loss 接口，通常只覆盖第一行的一部分。新增 critic、target 网络或经验回放，"
        "会触及第三到第五行，它们位于学习器主循环里，而不在损失回调里。"),
        add=[
            Tbl(100, 186, [290, 620, 810], [
                [tr("Freedom", "自由度"), tr("Typical research change", "典型研究改动"), tr("Control you actually need", "实际需要的控制")],
                [tr("1 Objective", "1 目标函数"), tr("new surrogate, entropy, divergence, distillation", "新 surrogate、熵、散度、蒸馏目标"), tr("current logprob / full logits / hidden, mask, differentiable graph, reducer", "当前 logprob / 全 logits / hidden、mask、可微图、reducer")],
                [tr("2 Estimator", "2 估计器"), tr("advantage, baseline, return, credit", "优势、baseline、回报、credit"), tr("full group / trajectory, value, cross-rank statistics", "完整组 / 轨迹、value、跨 rank 统计")],
                [tr("3 Data lifetime", "3 数据寿命"), tr("replay, priority, dynamic sampling, curriculum", "回放、优先级、动态采样、课程"), tr("buffer write / sample / delete, group IDs, versions, weights", "buffer 写入/采样/删除、group ID、版本、权重")],
                [tr("4 Model state", "4 模型状态"), tr("Q-head, multiple critics, target, teacher", "Q head、多个 critic、target、教师"), tr("model creation, own optimizers, checkpoint, sync policy", "模型创建、独立优化器、checkpoint、同步策略")],
                [tr("5 Update rule", "5 更新规则"), tr("actor/critic ratio, UTD, inner loops, meta-gradients", "actor/critic 比例、UTD、内层优化、元梯度"), tr("learner loop, backward order, step / skip / accumulation", "学习器主循环、反向顺序、step/skip/累积")],
                [tr("6 Execution", "6 执行调度"), tr("async, dynamic resources, partial rollout", "异步、动态资源、partial rollout"), tr("admission, back-pressure, weight publishing, recovery", "准入、背压、权重发布、恢复")],
            ], size=22, minh=62),
            T(100, 640, tr("A framework's custom-loss hook usually covers only **part of row 1**.", "框架的自定义 loss 接口通常只覆盖**第 1 行的一部分**。"), size=26, color=MUTED, w=1720),
        ]),
    Beat(tr(
        "Here's a concrete example of why the inputs matter. Exact entropy needs the full vocabulary distribution. If "
        "your hook only receives the log-prob of the sampled token, the naive move, adding minus log pi of that token "
        "as a bonus, has zero expected gradient, because the score function has mean zero under on-policy sampling. "
        "The correct estimator weights the score by the log-prob itself. And exact entropy needs full logits, which "
        "under tensor parallelism are sharded across the vocabulary.",
        "举一个具体例子，说明输入为什么重要。精确的熵需要完整的词表分布。如果你的接口只拿到被采样 token 的对数概率，"
        "那么朴素做法，也就是把这个 token 的负对数概率当作奖励项，期望梯度为零，因为在同策略采样下，得分函数的期望是零。"
        "正确的估计器要用对数概率本身给得分函数加权。而精确熵需要完整 logits，在张量并行下，这些 logits 是按词表切分在不同卡上的。"),
        add=[
            Rect(100, 700, 1720, 295, fill=PANEL, border=LINE, radius=16),
            M(130, 718, r"H(\pi_\theta)=-\sum_{a\in V}\pi_\theta(a)\log\pi_\theta(a)", size=34),
            M(130, 800, r"\mathbb{E}_{a\sim\pi_\theta}\big[\nabla_\theta\log\pi_\theta(a)\big]=0\ \Rightarrow\ \text{naive bonus }-\log\pi_\theta(a_t)\text{ has zero expected gradient}", size=31, color=CORAL, maxw=1660),
            M(130, 880, r"\nabla_\theta H=-\,\mathbb{E}_{a\sim\pi_\theta}\big[\log\pi_\theta(a)\,\nabla_\theta\log\pi_\theta(a)\big]", size=34, color=TEAL),
            T(1040, 730, tr("Exact entropy needs **full logits**, vocab-sharded under TP",
                            "精确熵需要**完整 logits**，TP 下按词表分片"), size=26, w=760, color=MUTED),
        ]),
]))

# =========================================================== 13. entry points
_entry_rows = [
    ["TRL", tr("Trainer subclass · `compute_loss_func` · rollout/reward hooks", "Trainer 子类 · `compute_loss_func` · rollout/奖励 hook"), tr("upstream Trainer lifecycle, data refresh", "上游 Trainer 生命周期、数据刷新")],
    ["SkyRL", tr("policy-loss & advantage registries · generator · trainer stages", "policy loss 与 advantage 注册表 · generator · trainer 阶段"), tr("model worker, data fields, main loop", "model worker、数据字段、主循环")],
    ["slime", tr("custom loss gets **logits + reducer** · custom advantage / generation / TIS", "自定义 loss 拿到 **logits + reducer** · 自定义 advantage / generation / TIS"), tr("Megatron actor & schedule, Sample", "Megatron actor 与调度、Sample")],
    ["Miles", tr("shared loss / advantage · custom async DataBuffer", "公共 loss / advantage · 自定义异步 DataBuffer"), tr("backend workers, update protocol", "后端 worker、更新协议")],
    ["OpenRLHF", tr("edit `loss.py` · `experience_maker.py` · `ppo_actor.py`", "修改 `loss.py` · `experience_maker.py` · `ppo_actor.py`"), tr("DeepSpeed wiring, roles", "DeepSpeed 接线、角色")],
    ["verl V1", tr("estimator / loss registries (selected-token logprob) · buffer / sampler", "estimator / loss 注册表（选中 token 的 logprob）· buffer / sampler"), tr("TransferQueue lifecycle, worker outputs", "TransferQueue 生命周期、worker 输出")],
    ["ROLL", tr("custom `Worker.loss_func` on logits · rollout-loop class", "自定义 `Worker.loss_func`（输入 logits）· rollout loop 类"), tr("advantage pipeline, Strategy", "advantage pipeline、Strategy")],
    ["AReaL", tr("workflow · PPOActor estimator/update · TrainEngine loss", "workflow · PPOActor 估计与更新 · TrainEngine loss"), tr("service roles, learner protocol", "服务角色、learner 协议")],
    ["NeMo RL", tr("LossFunction with `global_valid_toks` / `global_valid_seqs`", "带 `global_valid_toks` / `global_valid_seqs` 的 LossFunction"), tr("backend × controller paths", "后端 × controller 路径组合")],
    [tr("Own learner", "自有 learner"), tr("everything: model, optimizers, data, step", "一切：模型、优化器、数据、step"), tr("adapters to engines & runtime", "与推理引擎和运行时的适配")],
]
_ehead = [tr("Route", "路线"), tr("Built-in entry points", "内置入口"), tr("Beyond that, you modify", "超出接口后要改哪里")]
SCENES.append(Scene("entry", P4, tr("Where each framework lets you in", "各框架的算法入口"), beats=[
    Beat(tr(
        "With those freedoms in mind, here's where each framework lets you in, based on the pinned source code. TRL "
        "gives you trainer subclasses and loss functions, but you inherit the upstream Transformers lifecycle. SkyRL "
        "has separate registries for policy losses and advantage estimators, plus a clean generator interface. slime's "
        "custom loss receives differentiable logits and a reducer, which is unusually direct. verl's registries pass "
        "selected-token log-probs, so full-logit objectives need deeper worker changes. ROLL lets you override the "
        "worker's loss on logits. And OpenRLHF is a few explicit files that are easy to fork.",
        "带着这些自由度，再看各框架基于固定版本源码的算法入口。TRL 提供 Trainer 子类和损失函数入口，但你继承的是上游 Transformers 的生命周期。"
        "SkyRL 为 policy loss 和优势估计器分别提供注册表，还有清晰的 generator 接口。slime 的自定义 loss 直接拿到可微的 logits 和 reducer，"
        "这一点非常直接。verl 的注册表传入的是选中 token 的对数概率，因此需要全 logits 的目标得深入修改 worker。ROLL 允许在 logits 上覆写 worker 的损失。"
        "OpenRLHF 则是几个明确的文件，很容易维护一个小 fork。"),
        add=[Tbl(100, 186, [220, 860, 640], [_ehead] + _entry_rows, size=22, minh=58)]),
    Beat(tr(
        "NeMo RL has a design worth copying everywhere: its loss interface receives the global count of valid tokens "
        "and sequences, so distributed normalization is part of the contract instead of an accident of micro-batching. "
        "The right comparison is not which framework lists more algorithms. It's how many boundaries your change must "
        "cross, which fast paths survive it, and what it costs to maintain.",
        "NeMo RL 有一个值得到处借鉴的设计：它的损失接口直接接收全局有效 token 数和序列数，于是分布式归一化成了契约的一部分，"
        "而不是微批划分的偶然结果。正确的比较方式不是看谁列出的算法更多，而是看你的改动要跨越多少边界、哪些高性能路径还能保留，"
        "以及长期维护要付出多少成本。"),
        add=[
            Box(100, 850, 1720, 140, title=tr("Compare: boundaries crossed · fast paths kept · maintenance cost", "比较：跨越多少边界 · 保留哪些快路径 · 维护成本"),
                body=tr("Design to copy (NeMo RL): the loss receives global valid-token / valid-sequence counts → distributed normalization is part of the contract.",
                        "值得借鉴（NeMo RL）：loss 接收全局有效 token / 序列数 → 分布式归一化写进契约。"),
                tcolor=AMBER, bar=AMBER, bsize=24),
        ]),
]))

# =========================================================== 14. landscape
_cards = [
    ("verl", "FSDP/FSDP2 · Megatron · vLLM/SGLang", tr("HybridFlow separates control flow from distributed compute; broadest platform", "HybridFlow：控制流与分布式计算分离；覆盖最广的平台"), AMBER),
    ("OpenRLHF", "Ray · DeepSpeed · vLLM", tr("clear roles; natural for HF + DeepSpeed teams; easy small forks", "角色清晰；HF + DeepSpeed 团队上手自然；易于小 fork"), BLUE),
    ("slime", "Megatron · SGLang", tr("thin algorithm layer; big MoE; straw rollout persistence", "算法层很薄；大 MoE；straw 持久化 rollout"), TEAL),
    ("Miles", "slime fork · Megatron/FSDP/TorchTitan", tr("weight transfer, TITO/R3, low precision, async buffers, recovery", "权重传输、TITO/R3、低精度、异步 buffer、恢复"), VIOLET),
    ("SkyRL", "FSDP2/Megatron · vLLM", tr("modular generator/env; sampler-support & routing replay", "模块化 generator/环境；采样支持集与路由重放"), CORAL),
    ("AReaL", "FSDP2 · Megatron · services", tr("async-native; staleness control; decoupled PPO; agents", "原生异步；陈旧度控制；解耦 PPO；Agent"), GREEN),
    ("NeMo RL", "Megatron-Core · DTensor", tr("MoE, multimodal, multi-teacher distillation; NVIDIA stack", "MoE、多模态、多教师蒸馏；NVIDIA 技术栈"), PINK),
    ("ROLL", "Ray roles · FSDP2/Megatron", tr("multi-domain sampling, multi-reward, agentic pipelines", "多领域混采、多奖励、Agent 流水线"), AMBER),
    ("RLinf", "env–rollout–train co-design", tr("embodied / VLA, simulation & real robots", "具身 / VLA，仿真与真机闭环"), BLUE),
    ("Prime-RL", "FSDP2 · vLLM · async", tr("environment-driven agent RL with verifiers", "以 verifiers 环境驱动的 Agent RL"), TEAL),
]
_card_els = []
for i, (nm, stack, s, c) in enumerate(_cards):
    x = 100 + (i % 5) * 348
    y = 186 + (i // 5) * 290
    _card_els += [
        Box(x, y, 328, 270, fill=PANEL, border=LINE, bar=c, radius=14),
        T(x + 26, y + 18, nm, size=32, bold=True, color=c),
        T(x + 26, y + 66, stack, size=19, color=MUTED, w=285, lh=1.25),
        T(x + 26, y + 130, s, size=23, w=285, lh=1.3),
    ]
SCENES.append(Scene("landscape", P4, tr("The framework landscape at a glance", "主流框架定位速览"), beats=[
    Beat(tr(
        "Here is the landscape at a glance. These are judgments about architecture fit, not a speed ranking. verl is the "
        "broadest platform, with HybridFlow separating control flow from distributed compute. OpenRLHF composes Ray, "
        "DeepSpeed and vLLM in a way HF teams find natural. slime is a thin algorithm layer on Megatron and SGLang, "
        "strong for large MoE. Miles, forked from slime, invests in weight transfer, consistency and recovery. SkyRL "
        "offers modular generator and environment interfaces. AReaL co-designs asynchronous sampling with the learning "
        "rule. NeMo RL integrates NVIDIA's large-model stack and distillation. ROLL handles multi-domain, multi-reward "
        "pipelines. RLinf targets embodied and VLA training, and Prime-RL couples async training with verifier "
        "environments.",
        "这是主流框架的速览。这里是架构适配上的判断，不是速度排名。verl 是覆盖面最广的平台，HybridFlow 把控制流与分布式计算分开。"
        "OpenRLHF 以 HF 团队熟悉的方式组合了 Ray、DeepSpeed 和 vLLM。slime 是架在 Megatron 和 SGLang 上的薄算法层，擅长大 MoE。"
        "从 slime 分叉出来的 Miles，重点投入权重传输、一致性和故障恢复。SkyRL 提供模块化的 generator 与环境接口。AReaL 把异步采样和学习规则一起设计。"
        "NeMo RL 整合了 NVIDIA 的大模型技术栈和蒸馏。ROLL 擅长多领域、多奖励的流水线。RLinf 面向具身与 VLA 训练，Prime-RL 则把异步训练和验证器环境结合在一起。"),
        add=_card_els),
    Beat(tr(
        "Around them: fine-tuning toolchains for SFT and preference work; agent adapters that capture real agent "
        "traces; environment hubs; and Tinker, a hosted API where you write the loop and the service runs distributed "
        "training. Also check maintenance status. torchtune is no longer actively developed, ReaLHF is archived in favor "
        "of AReaL, and TorchForge's direction moved into TorchTitan.",
        "围绕它们的还有：做 SFT 和偏好优化的微调工具链；捕获真实 Agent 轨迹的接入层；环境平台；以及 Tinker，一个托管 API，"
        "你来写训练循环，服务端负责分布式训练。另外要核对维护状态：torchtune 已不再积极开发，ReaLHF 已归档、转向 AReaL，"
        "TorchForge 的方向则并入了 TorchTitan。"),
        add=[
            Box(100, 780, 420, 210, title=tr("Fine-tuning", "微调工具链"), body="TRL · LLaMA-Factory · ms-swift · Axolotl · Unsloth · Tunix · Open Instruct",
                tcolor=TEAL, bar=TEAL, bsize=22, tsize=26),
            Box(540, 780, 420, 210, title=tr("Agents & envs", "Agent 与环境"), body="rLLM · Agent Lightning · ART · verifiers · NeMo Gym · OpenEnv · Atropos",
                tcolor=VIOLET, bar=VIOLET, bsize=22, tsize=26),
            Box(980, 780, 380, 210, title=tr("Hosted", "托管"), body=tr("Tinker: forward_backward · optim_step · sample (LoRA-centric)", "Tinker：forward_backward · optim_step · sample（以 LoRA 为主）"),
                tcolor=BLUE, bar=BLUE, bsize=22, tsize=26),
            Box(1380, 780, 440, 210, title=tr("Maintenance flags", "维护状态"), body=tr("torchtune: not actively developed · ReaLHF: archived → AReaL · TorchForge → TorchTitan · Cosmos-RL: limited", "torchtune 停止积极开发 · ReaLHF 归档 → AReaL · TorchForge → TorchTitan · Cosmos-RL 有限维护"),
                tcolor=CORAL, bar=CORAL, bsize=22, tsize=26),
        ]),
]))

# =========================================================== 15. contract & architecture
_pseudo = """while not budget.exhausted():
    traces  = rollout.collect(prompts, policy_version)
    records = trajectory.encode(traces)
    replay.add(records)
    for spec in algorithm.plan_updates(replay):
        batch = replay.sample(spec)
        out   = learner.forward(batch, spec)
        loss, metrics = algorithm.objective(out, batch)
        learner.backward_and_step(loss, spec)
        algorithm.update_targets_if_needed(learner)
    if algorithm.should_publish():
        policy_version = runtime.publish_weights(learner)"""

_mods = [("trajectory", AMBER, tr("own", "自持")), ("algorithm", AMBER, tr("own", "自持")), ("learner", AMBER, tr("own/reuse", "自持/复用")),
         ("rollout", TEAL, tr("reuse", "复用")), ("runtime", TEAL, tr("reuse", "复用")), ("evaluation", BLUE, tr("define", "定义"))]


def _mods_draw(layer, d, lang):
    for i, (nm, c, tag) in enumerate(_mods):
        x = 940 + (i % 3) * 296
        y = 190 + (i // 3) * 74
        d.rounded_rectangle([x, y, x + 280, y + 62], 12, fill=PANEL2, outline=c, width=2)
        d.text((x + 18, y + 31), nm, font=font(23, True, True), fill=c, anchor="lm")
        d.text((x + 266, y + 31), L(tag, lang), font=font(19, False, False, lang), fill=MUTED, anchor="rm")


_clocks = [(tr("optimizer step", "优化器 step"), AMBER), (tr("published policy version", "发布的策略版本"), TEAL),
           (tr("recoverable checkpoint", "可恢复的 checkpoint"), BLUE)]


def _clocks_draw(layer, d, lang):
    x, y = 100, 845
    for i, (t, c) in enumerate(_clocks):
        ch = Chip(x, y, t, size=26, fill=PANEL2, color=c, border=c)
        w, h = ch.size_of(lang)
        ch.draw(layer, d, lang)
        x += w
        if i < len(_clocks) - 1:
            d.text((x + 22, y + 6), "≠", font=font(34, True), fill=MUTED)
            x += 80


SCENES.append(Scene("contract", P5, tr("Own the trajectory contract; reuse the runtime", "自己掌握轨迹契约，复用运行时"), beats=[
    Beat(tr(
        "So what should you own? The review's answer is: own the trajectory contract. Keep exact token IDs per turn. "
        "Separate attention, loss and packing masks. Store behavior log-probs together with the sampling parameters. "
        "Keep old, reference and teacher log-probs distinct. Carry group, episode and turn IDs. Record versions per token "
        "span. And distinguish terminated from truncated, because they bootstrap differently. When you switch backends, "
        "this contract tells you exactly what information was lost.",
        "那么你应该自己掌握什么？报告的回答是：掌握轨迹契约。保存每一轮真实的 token ID；把 attention、loss 和 packing 的掩码分开；"
        "把行为对数概率和采样参数一起保存；旧策略、参考模型和教师的对数概率分别保存；携带 group、episode 和 turn 的 ID；"
        "按 token 片段记录版本；区分正常终止和截断，因为两者的 bootstrap 方式不同。切换后端时，这份契约能准确告诉你丢失了哪些信息。"),
        add=[
            T(100, 186, tr("Trajectory contract (define it yourself)", "轨迹契约（由你自己定义）"), size=28, bold=True, color=AMBER),
            Bul(100, 236, 790, [
                tr("raw `token_ids` per turn (no re-tokenize)", "每轮原始 `token_ids`（不重新分词）"),
                tr("attention / loss / segment masks", "attention / loss / segment 掩码"),
                tr("`behavior_logprobs` + temperature, top-p, top-k", "`behavior_logprobs` + temperature、top-p、top-k"),
                tr("old / current / ref / teacher logprobs", "old / current / ref / teacher logprob"),
                tr("group / episode / turn / sample IDs", "group / episode / turn / sample ID"),
                tr("reward · advantage · return · value (shape, dtype, stop-grad)", "reward · advantage · return · value（形状、dtype、stop-grad）"),
                tr("policy version per token span", "按 token 片段记录策略版本"),
                tr("terminated vs truncated", "正常终止 vs 截断"),
                tr("sampler support · MoE routes (when needed)", "采样支持集 · MoE 路由（需要时）"),
                tr("sample weight · reuse count · source · filter reason", "样本权重 · 使用次数 · 来源 · 过滤原因"),
            ], size=23, gap=7, lh=1.3),
        ]),
    Beat(tr(
        "Then keep a minimal architecture of six modules: trajectory, algorithm, learner, rollout, runtime and evaluation. "
        "Write the algorithm in plain PyTorch, and reuse engines and runtimes for the rest. This loop makes the control "
        "flow explicit: collect, encode, plan updates, sample, forward, compute the objective, step, update targets, and "
        "publish. If your method is a single-actor policy gradient, hand most of these steps to an existing trainer. If "
        "it has multiple networks, own the learner loop.",
        "然后保持一个由六个模块组成的最小架构：trajectory、algorithm、learner、rollout、runtime 和 evaluation。"
        "算法用纯 PyTorch 写，其余部分复用现有的推理引擎和运行时。这个循环让控制流一目了然：采集、编码、规划更新、采样、前向、"
        "计算目标、更新参数、更新 target、发布权重。如果你的方法只是单 actor 的策略梯度，把大部分步骤交给现成的 trainer；"
        "如果涉及多个网络，就自己持有学习器主循环。"),
        add=[
            Custom(_mods_draw),
            Code(940, 350, 880, _pseudo, size=20, title=tr("who controls learning (concept, not a real API)", "谁控制学习过程（概念伪代码，非真实 API）"), hl=[4, 7, 8]),
        ]),
    Beat(tr(
        "Finally, name your clocks. An optimizer step, a published policy version and a recoverable checkpoint are three "
        "different events. And a real checkpoint for online training also needs the rollout queue, trajectory versions, "
        "replay position, environment state and sampler state, not just model weights.",
        "最后，给你的时钟命名。一次优化器 step、一个发布的策略版本、一个可恢复的 checkpoint，是三种不同的事件。"
        "而在线训练真正的 checkpoint，除了模型权重，还需要 rollout 队列、轨迹版本、replay 消费位置、环境状态和采样器状态。"),
        add=[
            Custom(_clocks_draw),
            T(100, 930, tr("Online checkpoint = weights + optimizer + rollout queue + versions + replay position + env state + RNG / sampler state",
                           "在线训练的 checkpoint = 权重 + 优化器 + rollout 队列 + 版本 + replay 位置 + 环境状态 + 随机数/采样器状态"), size=24, color=MUTED, w=1720),
        ]),
]))


# =========================================================== 16. decision procedure
def _arrow_label(x, y, text, c):
    return T(x, y, text, size=24, bold=True, color=c)


SCENES.append(Scene("decide", P5, tr("A decision procedure", "选型决策流程"), beats=[
    Beat(tr(
        "Now we can turn all of this into a decision procedure. First question: does your training data come from the "
        "current policy? If not, you're offline. Start with TRL plus PEFT; add DeepSpeed or FSDP2 for scale; use "
        "LLaMA-Factory, ms-swift or Axolotl for config-driven recipes; QLoRA or Unsloth when memory is the constraint; "
        "and Tunix on TPUs.",
        "现在可以把这些整理成一个决策流程。第一个问题：训练数据是否由当前策略产生？如果不是，你处在离线场景：从 TRL 加 PEFT 起步；"
        "需要扩展规模时加上 DeepSpeed 或 FSDP2；想要配置驱动的配方，用 LLaMA-Factory、ms-swift 或 Axolotl；显存是瓶颈时用 QLoRA 或 Unsloth；"
        "在 TPU 上则选 Tunix。"),
        add=[
            Box(100, 190, 540, 120, title=tr("Q1  Is training data produced by the current policy?", "Q1  训练数据是否由当前策略产生？"),
                tcolor=TEXT, fill=PANEL2, border=AMBER, tsize=27),
            Arrow([(370, 312), (370, 392)], color=MUTED),
            _arrow_label(388, 330, tr("no", "否"), MUTED),
            Box(100, 395, 540, 330, title=tr("Offline", "离线"),
                body=tr("**TRL** (+PEFT) for SFT / DPO\nscale: **DeepSpeed** or **FSDP2**\nrecipes: LLaMA-Factory · ms-swift · Axolotl\nmemory-bound: **QLoRA** · **Unsloth**\nTPU / JAX: **Tunix**",
                        "**TRL**（+PEFT）做 SFT / DPO\n扩展：**DeepSpeed** 或 **FSDP2**\n配方：LLaMA-Factory · ms-swift · Axolotl\n显存受限：**QLoRA** · **Unsloth**\nTPU / JAX：**Tunix**"),
                tcolor=BLUE, bar=BLUE, bsize=25, lh=1.45),
        ]),
    Beat(tr(
        "If you're online, ask: does your idea fit inside the generator, advantage and policy-loss interfaces? If yes, "
        "reuse a complete runtime. The review's default first prototype is SkyRL on FSDP2 with vLLM; verl for a shared, "
        "multi-backend team platform; OpenRLHF if you already have HF and DeepSpeed assets; slime, Miles or NeMo RL once "
        "you've committed to Megatron and large MoE. If not, because you need new networks, replay or a different update "
        "rule, own the learner, and reuse FSDP2 or DeepSpeed for execution and vLLM or SGLang for generation.",
        "如果是在线场景，再问：你的想法能否完全落在 generator、优势估计和 policy loss 这三个接口里？如果能，就复用一个完整运行时。"
        "报告推荐的第一个原型是 SkyRL 的 FSDP2 加 vLLM 路径；团队共享、多后端的平台选 verl；已有 HF 和 DeepSpeed 积累选 OpenRLHF；"
        "确定走 Megatron 和大 MoE 路线后，选 slime、Miles 或 NeMo RL。如果不能，比如需要新增网络、经验回放或不同的更新规则，"
        "就自己持有学习器，执行层复用 FSDP2 或 DeepSpeed，生成层复用 vLLM 或 SGLang。"),
        add=[
            Arrow([(642, 250), (712, 250)], color=MUTED),
            _arrow_label(650, 206, tr("yes", "是"), MUTED),
            Box(715, 190, 580, 120, title=tr("Q2  Does it fit generator + advantage + policy loss?", "Q2  能否落在 generator + 优势 + policy loss 接口内？"),
                tcolor=TEXT, fill=PANEL2, border=AMBER, tsize=27),
            Arrow([(1005, 312), (1005, 392)], color=MUTED),
            _arrow_label(1023, 330, tr("yes", "是"), MUTED),
            Box(715, 395, 580, 330, title=tr("Reuse a complete runtime", "复用完整运行时"),
                body=tr("default prototype: **SkyRL** (FSDP2 + vLLM)\nteam platform, many recipes: **verl V1**\nHF + DeepSpeed assets: **OpenRLHF**\nMegatron + big MoE: **slime** · **Miles** · **NeMo RL**\nmulti-domain pipelines: **ROLL**",
                        "默认原型：**SkyRL**（FSDP2 + vLLM）\n团队平台、多配方：**verl V1**\n已有 HF + DeepSpeed：**OpenRLHF**\nMegatron + 大 MoE：**slime** · **Miles** · **NeMo RL**\n多领域流水线：**ROLL**"),
                tcolor=TEAL, bar=TEAL, bsize=25, lh=1.45),
            Arrow([(1297, 250), (1367, 250)], color=MUTED),
            _arrow_label(1305, 206, tr("no", "否"), MUTED),
            Box(1370, 190, 450, 535, title=tr("Own the learner", "自己持有 learner"),
                body=tr("new critics / Q / target, replay, UTD, inner loops\n\n+ **FSDP2 / DeepSpeed** for execution\n+ **vLLM / SGLang** + weight sync for generation\n\nkeep the algorithm in plain PyTorch",
                        "新增 critic / Q / target、经验回放、UTD、内层优化\n\n+ 执行层：**FSDP2 / DeepSpeed**\n+ 生成层：**vLLM / SGLang** + 权重同步\n\n算法保持纯 PyTorch 实现"),
                tcolor=CORAL, bar=CORAL, bsize=25, lh=1.4),
        ]),
    Beat(tr(
        "Specialized cases. If you already have a working agent, keep its harness and capture traces with rLLM, Agent "
        "Lightning or ART. If asynchrony itself is your research, study AReaL and PipelineRL. For embodied or VLA "
        "training, evaluate RLinf. And if you'd rather not run GPUs at all, Tinker exposes forward-backward, optimizer "
        "step and sampling as an API, within LoRA-centric limits.",
        "还有几种特殊情况。如果你已经有一个运行良好的 Agent，保留它的执行框架，用 rLLM、Agent Lightning 或 ART 捕获轨迹。"
        "如果异步本身就是你的研究课题，重点研究 AReaL 和 PipelineRL。做具身或 VLA 训练，评估 RLinf。如果你根本不想运维 GPU，"
        "Tinker 把前向反向、优化器 step 和采样做成了 API，但能力边界以 LoRA 为主。"),
        add=[
            Box(100, 760, 415, 230, title=tr("Existing agent harness", "已有 Agent 执行框架"), body="rLLM · Agent Lightning · ART", tcolor=VIOLET, bar=VIOLET, bsize=25),
            Box(535, 760, 415, 230, title=tr("Async is the research", "异步本身是研究课题"), body=tr("AReaL · PipelineRL (compare SkyRL, Miles)", "AReaL · PipelineRL（对照 SkyRL、Miles）"), tcolor=GREEN, bar=GREEN, bsize=25),
            Box(970, 760, 415, 230, title=tr("Embodied / VLA", "具身 / VLA"), body=tr("RLinf (+ verl-vla)", "RLinf（并考察 verl-vla）"), tcolor=BLUE, bar=BLUE, bsize=25),
            Box(1405, 760, 415, 230, title=tr("No GPU operations", "不想运维 GPU"), body=tr("Tinker: you write the loop; LoRA-centric", "Tinker：你写循环；以 LoRA 为主"), tcolor=PINK, bar=PINK, bsize=25),
        ]),
]))

# =========================================================== 17. validation
SCENES.append(Scene("validate", P5, tr("Validate: two stages and a code checklist", "验证：两阶段评测与源码检查清单"), beats=[
    Beat(tr(
        "How do you validate a choice? Use two stages. Stage one fixes the trajectories: feed every candidate the same "
        "tokens, masks, rewards and behavior log-probs, and compare loss, gradients, a single parameter update, memory "
        "and time. That separates execution differences from sampling differences. Stage two runs the closed loop with "
        "the same start checkpoint, data, evaluation and budget, aligning every knob, from group size and sampling "
        "parameters to loss reduction, KL, clipping and filtering. Then compare GPU-hours to target across several seeds.",
        "怎样验证一个选择？分两个阶段。第一阶段固定轨迹：给每个候选框架相同的 token、掩码、奖励和行为对数概率，比较 loss、梯度、"
        "一次参数更新、显存和耗时。这样就把执行差异和采样差异分开了。第二阶段跑完整闭环：相同的起始 checkpoint、数据、评测集和预算，"
        "并对齐所有旋钮，从 group size、采样参数，到 loss 归约方式、KL、clip 和样本过滤。然后在多个随机种子下比较达到目标所需的 GPU 小时。"),
        add=[
            Box(100, 190, 840, 290, title=tr("Stage 1 · fixed trajectories → numeric parity", "阶段一 · 固定轨迹 → 数值对齐"),
                body=tr("same tokens, masks, rewards, behavior logprobs\ncompare loss, gradients, one parameter update, memory, time\n→ separates **execution** from **sampling** differences",
                        "相同的 token、掩码、奖励、行为 logprob\n比较 loss、梯度、一次参数更新、显存、耗时\n→ 把**执行差异**与**采样差异**分开"),
                tcolor=BLUE, bar=BLUE, bsize=25, lh=1.45),
            Box(980, 190, 840, 290, title=tr("Stage 2 · closed loop → cost to target", "阶段二 · 完整闭环 → 达标成本"),
                body=tr("same start checkpoint, data, eval set, budget\nalign group size, temperature, top-p/k, max length, reward norm, loss reduction, KL, clip, filtering, updates per batch, precision, GPUs\n→ **GPU-hours to target**, several seeds",
                        "相同起始 checkpoint、数据、评测集、预算\n对齐 group size、温度、top-p/k、最大长度、奖励归一化、loss 归约、KL、clip、过滤、每批更新次数、精度、GPU 数\n→ 多种子下的**达标 GPU 小时**"),
                tcolor=TEAL, bar=TEAL, bsize=25, lh=1.4),
        ]),
    Beat(tr(
        "And when reading code, check these questions, which matter more than the number of built-in algorithms. Where "
        "exactly is the final mean taken, and does the engine scale again? Are group statistics aggregated across ranks? "
        "Are top-p log-probs normalized on the support? Does any branch silently detach a tensor? Does weight sync push "
        "everything, run quantization post-processing, and invalidate caches? Does the buffer delete samples, and does "
        "resume restore everything?",
        "读源码时，请检查下面这些问题，它们比内置算法的数量更重要：最终的均值到底在哪个维度上取，引擎是否又缩放了一次？"
        "同一个 prompt 组的奖励统计是否跨 rank 聚合？top-p 的对数概率是否在支持集上归一化？有没有哪个分支悄悄 detach 了张量？"
        "权重同步是否推送了全部参数、执行了量化后处理、并让缓存失效？buffer 是否在采样后删除数据，恢复时是否把一切都恢复了？"),
        add=[
            Bul(100, 520, 1720, [
                tr("Final mean over tokens, sequences, ranks or micro-batches, and does the engine rescale again?", "最终均值作用于 token、序列、rank 还是微批？引擎是否又缩放一次？"),
                tr("Group reward statistics aggregated across ranks for the same prompt?", "同一 prompt 组的奖励统计是否跨 rank 聚合？"),
                tr("top-p log-probs normalized on the sampling support?", "top-p 对数概率是否在采样支持集上归一化？"),
                tr("Any branch that silently detaches a tensor or converts it to numpy?", "是否有分支悄悄 detach 张量或转成 numpy？"),
                tr("Weight sync: whole model? quantization post-processing? prefix / KV invalidated?", "权重同步：是否完整？是否执行量化后处理？prefix / KV 是否失效？"),
                tr("Buffer deletes after sampling? Resume restores optimizer, sampler, unconsumed data, versions?", "buffer 采样后是否删除？恢复时是否还原优化器、采样器、未消费数据与版本？"),
            ], size=26, gap=16, marker="☐", dot=AMBER),
        ]),
    Beat(tr(
        "Keep a tiny pure-PyTorch reference loss and step. After every kernel, precision or backend change, compare "
        "against it. Then improve efficiency one bottleneck at a time, in profile order. It's slower on day one, and "
        "much faster over the life of a project.",
        "保留一个很小的纯 PyTorch 参考 loss 和 step。每次更换 kernel、精度或后端之后，都和它对比。然后按 profile 的顺序，"
        "一次只解决一个瓶颈。第一天会慢一些，但在整个项目周期里会快得多。"),
        add=[
            Box(100, 900, 1720, 95, body=tr("**Anchor:** tiny pure-PyTorch reference.  **Order:** rollout batching → ref/critic forward → train memory/attention → weight transfer → async overlap → low precision & MoE",
                                             "**锚点：**小型纯 PyTorch 参考实现。**顺序：**rollout batching → ref/critic 前向 → 训练显存/attention → 权重传输 → 异步重叠 → 低精度与 MoE"),
                bsize=24, pad=24, border=TEAL),
        ]),
]))

# =========================================================== 18. takeaways
_take = [
    (tr("One gradient, two choices", "一个梯度，两个选择"),
     tr("Data source μ and weight ψ. μ fixed → trainer problem; μ = π_θ → systems problem.", "数据来源 μ 与权重 ψ。μ 固定 → 训练器问题；μ = π_θ → 系统问题。"), AMBER),
    (tr("Compose by layer", "按层组合"),
     tr("One component per role; verify the whole path (model ↔ backend ↔ weight sync ↔ engine ↔ trajectory ↔ update) before trusting a feature matrix.",
        "每个角色一个部件；在相信功能矩阵之前，先验证整条路径（模型 ↔ 后端 ↔ 权重同步 ↔ 引擎 ↔ 轨迹 ↔ 更新）。"), BLUE),
    (tr("Systems differ beyond algorithm lists", "差距不在算法列表"),
     tr("Scheduling, train–inference consistency, MoE depth, weight transfer and recovery decide the outcome.", "调度、训练推理一致性、MoE 深度、权重传输与恢复能力决定结果。"), TEAL),
    (tr("Own the math, reuse the runtime", "掌握数学，复用运行时"),
     tr("Own the trajectory contract, objective and update rule; reuse engines and sharding backends.", "自己掌握轨迹契约、目标函数与更新规则；复用推理引擎和分片后端。"), VIOLET),
    (tr("Optimize cost-to-target", "优化达标成本"),
     tr("One bottleneck at a time, anchored by a reference implementation.", "一次解决一个瓶颈，以参考实现为锚点。"), CORAL),
]


def _take_els(idx):
    out = []
    for i in idx:
        t, b, c = _take[i]
        y = 190 + i * 160
        out += [
            Box(100, y, 1720, 142, fill=PANEL, border=LINE, bar=c, radius=14),
            T(140, y + 26, f"{i + 1}", size=62, bold=True, color=c),
            T(240, y + 22, t, size=32, bold=True, color=c, w=1540),
            T(240, y + 72, b, size=25, w=1540, color=TEXT),
        ]
    return out


SCENES.append(Scene("summary", tr("SUMMARY", "总结"), tr("Five takeaways", "五条结论"), beats=[
    Beat(tr(
        "Let's wrap up with five takeaways. One: every post-training method is one gradient with two choices, the data "
        "source and the token weight; fixed data makes it a trainer problem, on-policy data makes it a systems problem. "
        "Two: compose by layer, one component per role, and verify the whole path before you trust a feature matrix. "
        "Three: frameworks differ less in their algorithm lists than in scheduling, consistency, MoE depth, weight "
        "transfer and recovery.",
        "最后总结五点。第一，所有后训练方法都是同一个梯度加两个选择：数据来源和 token 权重；数据固定就是训练器问题，在线数据就是系统问题。"
        "第二，按层组合，每个角色一个部件，在相信功能矩阵之前先验证整条路径。第三，框架之间的差距，与其说在算法列表，不如说在调度、"
        "一致性、MoE 支持深度、权重传输和故障恢复。"),
        add=_take_els([0, 1, 2])),
    Beat(tr(
        "Four: own the math and the trajectory contract, and reuse the runtime. Five: optimize cost to target, one "
        "bottleneck at a time, with a small reference implementation as your anchor. Thanks for watching.",
        "第四，自己掌握数学和轨迹契约，复用运行时。第五，优化达到目标的成本，一次解决一个瓶颈，并以一个小型参考实现为锚点。感谢观看。"),
        add=_take_els([3, 4])),
]))
