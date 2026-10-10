/* Post-Training Systems 101 — interactive layer (no framework). */
(function () {
  "use strict";

  // ------------------------------------------------------------ language
  var root = document.documentElement;
  var lang = root.getAttribute("data-lang") || "en";
  var listeners = [];
  function T(v) { return v && typeof v === "object" && !Array.isArray(v) ? (v[lang] != null ? v[lang] : v.en) : v; }
  function bi(en, zh) { return { en: en, zh: zh }; }
  function setLang(l) {
    lang = l;
    root.setAttribute("data-lang", l);
    root.lang = l === "zh" ? "zh-CN" : "en";
    try { localStorage.setItem("pt101-lang", l); } catch (e) {}
    document.querySelectorAll("[data-setlang]").forEach(function (b) { b.setAttribute("aria-pressed", String(b.dataset.setlang === l)); });
    listeners.forEach(function (fn) { fn(l); });
  }
  function onLang(fn) { listeners.push(fn); }
  document.querySelectorAll("[data-setlang]").forEach(function (b) { b.addEventListener("click", function () { setLang(b.dataset.setlang); }); });

  // ------------------------------------------------------------ helpers
  function $(s, el) { return (el || document).querySelector(s); }
  function $$(s, el) { return Array.prototype.slice.call((el || document).querySelectorAll(s)); }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  function md(s) { // **bold** and `code`
    return esc(s).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/`(.+?)`/g, "<code>$1</code>");
  }
  function tex(s, display) { return '<span class="tex"' + (display ? ' data-display="1"' : "") + ' data-tex="' + esc(s) + '"></span>'; }
  function renderMath(el) {
    if (!window.katex) return;
    $$(".tex", el).forEach(function (n) {
      if (n.dataset.done === n.dataset.tex) return;
      try {
        var disp = !!n.dataset.display || !!n.closest(".formula");
        katex.render(n.dataset.tex, n, { displayMode: disp, throwOnError: false, strict: "ignore" });
        n.dataset.done = n.dataset.tex;
      } catch (e) { n.textContent = n.dataset.tex; }
    });
  }
  function mmss(t) { t = Math.max(0, t); var m = Math.floor(t / 60), s = Math.floor(t % 60); return (m < 10 ? "0" : "") + m + ":" + (s < 10 ? "0" : "") + s; }

  var C = { amber: "var(--amber)", cyan: "var(--cyan)", teal: "var(--teal)", violet: "var(--violet)", coral: "var(--coral)", blue: "var(--blue)", green: "var(--green)", pink: "var(--pink)" };

  // ============================================================ DATA
  var METHODS = [
    { id: "sft", name: "SFT", regime: "trainer", mu: bi("fixed dataset D", "固定数据集 D"),
      psi: "m_t\\in\\{0,1\\}\\ \\ (\\text{loss mask})",
      note: bi("Supervised fine-tuning: ψ is one on the masked response tokens.", "监督微调：ψ 在被 mask 选中的回答 token 上等于 1。") },
    { id: "dpo", name: "DPO", regime: "trainer", mu: bi("fixed preference pairs (chosen, rejected)", "固定偏好对（chosen, rejected）"),
      psi: "\\pm\\,\\beta\\,\\sigma(\\hat r_l-\\hat r_w),\\quad \\hat r=\\beta\\log\\frac{\\pi_\\theta}{\\pi_{\\mathrm{ref}}}",
      note: bi("A sigmoid-weighted push: up on the chosen answer, down on the rejected one.", "sigmoid 加权：拉高被选中的回答，压低被拒绝的回答。") },
    { id: "ppo", name: "PPO / GRPO", regime: "systems", mu: bi("fresh samples from the old policy (online)", "从旧策略实时采样（在线）"),
      psi: "\\rho_t\\hat A_t\\ \\ (\\text{clipped}),\\quad \\rho_t=\\frac{\\pi_\\theta}{\\pi_{\\mathrm{old}}}",
      note: bi("Weight each token by a clipped importance ratio times an advantage.", "用截断后的重要性比率乘以优势为每个 token 加权。") },
    { id: "opd", name: bi("On-policy distillation", "在线蒸馏"), regime: "systems", mu: bi("fresh samples from the student", "从学生模型实时采样"),
      psi: "\\log\\pi_{\\mathrm{teacher}}(y_t)-\\log\\pi_\\theta(y_t)",
      note: bi("The teacher-minus-student log-probability is the per-token signal.", "教师与学生的对数概率之差就是逐 token 的信号。") }
  ];

  var LAYERS = [
    { c: C.amber, name: bi("Algorithm & experiment", "算法与实验层"),
      does: bi("reward · advantage · loss · critic · teacher · update rule", "奖励 · 优势 · 损失 · critic · 教师 · 更新规则"),
      tools: ["your code", "TRL", "SkyRL", "slime", "verl"],
      notes: [bi("This is the layer you should own: the objective, the estimator and the update rule.", "这是你应该自己掌握的一层：目标函数、估计器与更新规则。"),
              bi("Algorithm parts of TRL, SkyRL, slime and verl are references to read, not contracts to inherit.", "TRL、SkyRL、slime、verl 的算法实现是用来阅读的参考，而不是必须继承的契约。"),
              bi("Composition rule #1: a trainer can't replace a rollout loop; an engine never runs backward.", "组合规则 #1：训练器替代不了 rollout 循环；推理引擎从不做反向传播。")] },
    { c: C.coral, name: bi("Data & runtime", "数据与运行层"),
      does: bi("rollout · env · multi-turn · buffer · placement · versions", "rollout · 环境 · 多轮 · buffer · 资源布局 · 版本"),
      tools: ["verl", "OpenRLHF", "SkyRL", "slime", "Miles", "AReaL", "ROLL", "NeMo RL"],
      notes: [bi("Owns the loop when μ = π_θ: who generates, who scores, where samples wait, which version produced them.", "当 μ = π_θ 时由它掌管闭环：谁生成、谁打分、样本在哪等待、由哪个版本产生。"),
              bi("Asynchrony is a set of mechanisms (queues, admission, age gates, importance correction, mid-flight updates), not a switch.", "异步是一组机制（队列、准入、年龄门限、重要性修正、中途更新），而不是一个开关。"),
              bi("Ray = processes & resources; NCCL = tensors.", "Ray 管进程与资源；NCCL 管张量。")] },
    { c: C.blue, name: bi("Model & training execution", "模型与训练执行层"),
      does: bi("fwd/bwd · state sharding · TP/PP/EP/CP · optimizer", "前向/反向 · 状态分片 · TP/PP/EP/CP · 优化器"),
      tools: ["PyTorch FSDP2", "DeepSpeed", "Megatron-Core", "TorchTitan", "NeMo AutoModel"],
      notes: [bi("**FSDP2**: per-parameter DTensor sharding via `fully_shard`; plain PyTorch control flow. Contract: local shard ≠ `full_tensor()` (a collective).", "**FSDP2**：通过 `fully_shard` 做逐参数 DTensor 分片；控制流仍是普通 PyTorch。契约：本地分片 ≠ `full_tensor()`（后者是集合通信）。"),
              bi("**DeepSpeed**: ZeRO-1/2/3 + CPU/NVMe offload; AutoTP, AutoEP, Hybrid Engine. Contract: the engine owns accumulation & step boundaries.", "**DeepSpeed**：ZeRO-1/2/3 + CPU/NVMe 卸载；AutoTP、AutoEP、Hybrid Engine。契约：梯度累积与 step 边界由引擎掌握。"),
              bi("**Megatron-Core**: TP · PP · EP · CP for large dense & MoE models; Bridge recipes. Contract: the pipeline schedule owns micro-batches & backward.", "**Megatron-Core**：TP · PP · EP · CP，面向大规模稠密与 MoE 模型；Bridge 配方。契约：流水线调度掌握微批与反向。"),
              bi("DeepSpeed ≠ DeepSpeed-Chat ≠ OpenRLHF: infra · a classic SFT→RM→PPO app · a Ray pipeline on DeepSpeed.", "DeepSpeed ≠ DeepSpeed-Chat ≠ OpenRLHF：基础设施 · 经典 SFT→RM→PPO 应用 · 基于 DeepSpeed 的 Ray 流水线。")] },
    { c: C.teal, name: bi("Inference generation", "推理生成层"),
      does: bi("batching · KV cache · prefill/decode · sampling · hot update", "batching · KV cache · prefill/decode · 采样 · 热更新"),
      tools: ["vLLM", "SGLang"],
      notes: [bi("Prefill processes all prompt tokens in parallel (**compute-bound**); decode emits one token per step and re-reads weights + KV (**bandwidth-bound**).", "Prefill 并行处理全部提示 token（**计算受限**）；decode 每步一个 token，反复读权重与 KV（**带宽受限**）。"),
              bi("Engines amortize each weight read across many sequences: paged KV cache, continuous batching, prefix sharing.", "推理引擎把每次权重读取分摊到许多序列上：分页 KV cache、连续批处理、前缀共享。"),
              bi("Weight sync is a protocol boundary: reshard · cast · transfer · load. Tokens already generated stay **old-policy** tokens.", "权重同步是一条协议边界：重分片 · 转换 · 传输 · 加载。已经生成的 token 仍是**旧策略**的 token。"),
              bi("Rule: whatever the policy needs to generate must be expressible in the engine and synced; learner-only parts (critic, Q-head, target) can stay in the learner.", "规则：策略生成所需的一切都必须能在引擎中表达并同步；只在学习端使用的部分（critic、Q head、target）可以留在学习器。")] },
    { c: C.violet, name: bi("Parameterization & kernels", "参数化与 kernel 层"),
      does: bi("LoRA/QLoRA · attention · fused loss · low precision · comms", "LoRA/QLoRA · attention · 融合损失 · 低精度 · 通信"),
      tools: ["PEFT", "bitsandbytes", "Liger", "FlashAttention", "FlashInfer", "Transformer Engine", "DeepEP"],
      notes: [bi("**PEFT** `get_peft_model()` injects adapters · **bitsandbytes** NF4 storage, 8-bit optimizers · **Unsloth** fused kernels.", "**PEFT** 的 `get_peft_model()` 注入 adapter · **bitsandbytes** 提供 NF4 存储与 8-bit 优化器 · **Unsloth** 提供融合 kernel。"),
              bi("Training only LoRA ≠ syncing only LoRA: online RL may merge/unmerge or gather full weights for the engine.", "只训练 LoRA ≠ 只同步 LoRA：在线 RL 可能需要 merge/unmerge，或为引擎聚合完整权重。")] },
    { c: C.green, name: bi("Data quality & observability", "数据质量与观测层"),
      does: bi("processing · verifiers · eval · logs · profiling · checkpoints", "数据加工 · 验证器 · 评估 · 日志 · profile · checkpoint"),
      tools: ["DataTrove", "distilabel", "Math-Verify", "lm-eval", "W&B", "Nsight"],
      notes: [bi("Log separately: rollout · reward/env · policy/ref/value forward · backward/optimizer · weight sync · idle · peak memory.", "分开记录：rollout · 奖励/环境 · policy/ref/value 前向 · 反向/优化器 · 权重同步 · 空闲 · 峰值显存。"),
              bi("Verifiers are part of the reward, so they are part of the objective.", "验证器是奖励的一部分，因此也是目标函数的一部分。")] }
  ];

  var STEP_ROWS = [
    ["NeMo RL", "DeepSeek-V3 · 512×H100 · BF16 · seq 1536", bi("sync 111 s → one-step-off **64.1 s**/step", "同步 111 秒 → one-step-off **64.1 秒**/步"), bi("overlap helps here", "此处重叠有效")],
    ["NeMo RL", "Qwen3-30B-A3B · 16×GB200 · 4K · gen batch 2048", bi("sync 221 s → async **301 s**/step", "同步 221 秒 → 异步 **301 秒**/步"), bi("async can be slower", "异步可能更慢")],
    ["Miles", "Kimi K2 1T-A32B · 32+32 H100 nodes · 1 GB buckets", bi("NCCL broadcast 53.28 s → P2P **7.23 s**", "NCCL 广播 53.28 秒 → P2P **7.23 秒**"), bi("weight-update phase only", "仅权重更新阶段")]
  ];
  var ASYNC_ROWS = [
    [bi("Bounded queue / back-pressure", "有界队列 / 背压"), bi("pending count, producer lead", "待消费数量、生产者领先程度"), bi("per-sample age at training time", "每条样本训练时的版本年龄")],
    [bi("Admission window", "准入窗口"), bi("whether new tasks may start", "新任务能否启动"), bi("slow in-flight tasks finish fresh", "慢的在途任务完成时仍足够新")],
    [bi("Consume-time age gate", "消费时年龄门限"), bi("filter / wait by version before training", "训练前按版本过滤或等待"), bi("unchanged data distribution after drops", "丢弃后数据分布不变")],
    [bi("Importance correction", "重要性修正"), bi("known-probability distribution shift", "已知概率的分布偏移"), bi("unbiased, low-variance reuse of any old data", "任意旧数据复用都无偏且低方差")],
    [bi("Mid-flight weight update", "生成中途更新权重"), bi("generation / training overlap", "生成与训练的重叠程度"), bi("one trajectory = one fixed policy", "一条轨迹来自同一个固定策略")]
  ];
  var ASYNC_FW = bi("OpenRLHF: queue/slot back-pressure · SkyRL: admission + capacity · Miles: consume-time age filter · prime-rl: hard drop-stale · verl V1: drop | wait · NeMo RL: age-filtered replay path",
    "OpenRLHF：队列/slot 背压 · SkyRL：准入 + 容量 · Miles：消费时年龄过滤 · prime-rl：出队前硬过滤 · verl V1：drop | wait · NeMo RL：按年龄筛选的 replay 路径");
  var CONS_ROWS = [
    [bi("Re-tokenization drift", "重新分词导致 token 变化"), bi("keep exact generated token IDs, masks, logprobs", "保存真实生成的 token、动作掩码与 logprob"), "Miles TITO · rLLM trace capture"],
    [bi("Stale behavior policy", "采样策略比训练策略旧"), bi("record versions, bound staleness, importance correction", "记录版本、限制陈旧度、重要性修正"), bi("AReaL · NeMo RL async", "AReaL · NeMo RL 异步")],
    [bi("MoE routing differs", "MoE 专家路由不同"), bi("replay rollout-time expert routes", "训练时重放采样时的路由"), bi("Miles R3 · SkyRL · NeMo RL routing replay", "Miles R3 · SkyRL · NeMo RL 路由重放")],
    [bi("top-p / top-k truncation", "top-p / top-k 截断"), bi("normalize on the actual sampling support", "在实际采样支持集上归一化"), bi("SkyRL sampler-support replay", "SkyRL 采样支持集重放")],
    [bi("Kernels / precision / batch shape", "kernel / 精度 / batch 形状"), bi("share model definitions, batch-invariant kernels", "共享模型定义与 batch 无关的 kernel"), "TitanRL · vexact (verl ecosystem)"]
  ];
  var FREE_ROWS = [
    [bi("1 Objective", "1 目标函数"), bi("new surrogate, entropy, divergence, distillation", "新 surrogate、熵、散度、蒸馏目标"), bi("current logprob / full logits / hidden, mask, differentiable graph, reducer", "当前 logprob / 全 logits / hidden、mask、可微图、reducer")],
    [bi("2 Estimator", "2 估计器"), bi("advantage, baseline, return, credit", "优势、baseline、回报、credit"), bi("full group / trajectory, value, cross-rank statistics", "完整组 / 轨迹、value、跨 rank 统计")],
    [bi("3 Data lifetime", "3 数据寿命"), bi("replay, priority, dynamic sampling, curriculum", "回放、优先级、动态采样、课程"), bi("buffer write / sample / delete, group IDs, versions, weights", "buffer 写入/采样/删除、group ID、版本、权重")],
    [bi("4 Model state", "4 模型状态"), bi("Q-head, multiple critics, target, teacher", "Q head、多个 critic、target、教师"), bi("model creation, own optimizers, checkpoint, sync policy", "模型创建、独立优化器、checkpoint、同步策略")],
    [bi("5 Update rule", "5 更新规则"), bi("actor/critic ratio, UTD, inner loops, meta-gradients", "actor/critic 比例、UTD、内层优化、元梯度"), bi("learner loop, backward order, step / skip / accumulation", "学习器主循环、反向顺序、step/skip/累积")],
    [bi("6 Execution", "6 执行调度"), bi("async, dynamic resources, partial rollout", "异步、动态资源、partial rollout"), bi("admission, back-pressure, weight publishing, recovery", "准入、背压、权重发布、恢复")]
  ];
  var ENTRY_ROWS = [
    ["TRL", bi("Trainer subclass · `compute_loss_func` · rollout/reward hooks", "Trainer 子类 · `compute_loss_func` · rollout/奖励 hook"), bi("upstream Trainer lifecycle, data refresh", "上游 Trainer 生命周期、数据刷新")],
    ["SkyRL", bi("policy-loss & advantage registries · generator · trainer stages", "policy loss 与 advantage 注册表 · generator · trainer 阶段"), bi("model worker, data fields, main loop", "model worker、数据字段、主循环")],
    ["slime", bi("custom loss gets **logits + reducer** · custom advantage / generation / TIS", "自定义 loss 拿到 **logits + reducer** · 自定义 advantage / generation / TIS"), bi("Megatron actor & schedule, Sample", "Megatron actor 与调度、Sample")],
    ["Miles", bi("shared loss / advantage · custom async DataBuffer", "公共 loss / advantage · 自定义异步 DataBuffer"), bi("backend workers, update protocol", "后端 worker、更新协议")],
    ["OpenRLHF", bi("edit `loss.py` · `experience_maker.py` · `ppo_actor.py`", "修改 `loss.py` · `experience_maker.py` · `ppo_actor.py`"), bi("DeepSpeed wiring, roles", "DeepSpeed 接线、角色")],
    ["verl V1", bi("estimator / loss registries (selected-token logprob) · buffer / sampler", "estimator / loss 注册表（选中 token 的 logprob）· buffer / sampler"), bi("TransferQueue lifecycle, worker outputs", "TransferQueue 生命周期、worker 输出")],
    ["ROLL", bi("custom `Worker.loss_func` on logits · rollout-loop class", "自定义 `Worker.loss_func`（输入 logits）· rollout loop 类"), bi("advantage pipeline, Strategy", "advantage pipeline、Strategy")],
    ["AReaL", bi("workflow · PPOActor estimator/update · TrainEngine loss", "workflow · PPOActor 估计与更新 · TrainEngine loss"), bi("service roles, learner protocol", "服务角色、learner 协议")],
    ["NeMo RL", bi("LossFunction with `global_valid_toks` / `global_valid_seqs`", "带 `global_valid_toks` / `global_valid_seqs` 的 LossFunction"), bi("backend × controller paths", "后端 × controller 路径组合")],
    [bi("Own learner", "自有 learner"), bi("everything: model, optimizers, data, step", "一切：模型、优化器、数据、step"), bi("adapters to engines & runtime", "与推理引擎和运行时的适配")]
  ];
  var FRAMEWORKS = [
    { n: "verl", be: "FSDP/FSDP2 · Megatron · vLLM/SGLang", c: C.amber, tags: ["fsdp", "megatron", "vllm", "sglang"], d: bi("HybridFlow separates control flow from distributed compute; the broadest platform.", "HybridFlow：控制流与分布式计算分离；覆盖最广的平台。") },
    { n: "OpenRLHF", be: "Ray · DeepSpeed · vLLM", c: C.blue, tags: ["deepspeed", "vllm"], d: bi("Clear roles; natural for HF + DeepSpeed teams; easy small forks.", "角色清晰；HF + DeepSpeed 团队上手自然；易于小规模 fork。") },
    { n: "slime", be: "Megatron · SGLang", c: C.teal, tags: ["megatron", "sglang", "moe"], d: bi("Thin algorithm layer; big MoE; persistent rollout.", "算法层很薄；面向大 MoE；持久化 rollout。") },
    { n: "Miles", be: "slime fork · Megatron/FSDP/TorchTitan", c: C.violet, tags: ["megatron", "fsdp", "moe", "async"], d: bi("Weight transfer, TITO/R3, low precision, async buffers, recovery.", "权重传输、TITO/R3、低精度、异步 buffer、恢复。") },
    { n: "SkyRL", be: "FSDP2/Megatron · vLLM", c: C.coral, tags: ["fsdp", "megatron", "vllm"], d: bi("Modular generator/env; sampler-support & routing replay.", "模块化 generator/环境；采样支持集与路由重放。") },
    { n: "AReaL", be: "FSDP2 · Megatron · services", c: C.green, tags: ["fsdp", "megatron", "async", "agent"], d: bi("Async-native; staleness control; decoupled PPO; agents.", "原生异步；陈旧度控制；解耦 PPO；Agent。") },
    { n: "NeMo RL", be: "Megatron-Core · DTensor", c: C.pink, tags: ["megatron", "moe"], d: bi("MoE, multimodal, multi-teacher distillation; NVIDIA stack.", "MoE、多模态、多教师蒸馏；NVIDIA 技术栈。") },
    { n: "ROLL", be: "Ray roles · FSDP2/Megatron", c: C.amber, tags: ["fsdp", "megatron", "agent"], d: bi("Multi-domain sampling, multi-reward, agentic pipelines.", "多领域混采、多奖励、Agent 流水线。") },
    { n: "RLinf", be: "env–rollout–train co-design", c: C.blue, tags: ["embodied"], d: bi("Embodied / VLA, simulation & real robots.", "具身 / VLA，仿真与真机闭环。") },
    { n: "Prime-RL", be: "FSDP2 · vLLM · async", c: C.teal, tags: ["fsdp", "vllm", "async", "agent"], d: bi("Environment-driven agent RL with verifiers.", "以 verifiers 环境驱动的 Agent RL。") }
  ];
  var FILTERS = [["all", bi("All", "全部")], ["fsdp", bi("FSDP / FSDP2", "FSDP / FSDP2")], ["megatron", bi("Megatron", "Megatron")], ["deepspeed", bi("DeepSpeed", "DeepSpeed")],
    ["vllm", bi("vLLM", "vLLM")], ["sglang", bi("SGLang", "SGLang")], ["moe", bi("Large MoE", "大规模 MoE")], ["async", bi("Async", "异步")], ["agent", bi("Agents", "Agent")], ["embodied", bi("Embodied", "具身")]];
  var LAND_EXTRA = [
    { c: C.amber, t: bi("Fine-tuning", "微调工具链"), b: bi("TRL · LLaMA-Factory · ms-swift · Axolotl · Unsloth · Tunix · Open Instruct", "TRL · LLaMA-Factory · ms-swift · Axolotl · Unsloth · Tunix · Open Instruct") },
    { c: C.violet, t: bi("Agents & envs", "Agent 与环境"), b: bi("rLLM · Agent Lightning · ART · verifiers · NeMo Gym · OpenEnv · Atropos", "rLLM · Agent Lightning · ART · verifiers · NeMo Gym · OpenEnv · Atropos") },
    { c: C.pink, t: bi("Hosted", "托管"), b: bi("Tinker: forward_backward · optim_step · sample (LoRA-centric)", "Tinker：forward_backward · optim_step · sample（以 LoRA 为主）") },
    { c: C.coral, t: bi("Maintenance flags", "维护状态"), b: bi("torchtune: not actively developed · ReaLHF: archived → AReaL · TorchForge → TorchTitan · Cosmos-RL: limited", "torchtune 停止积极开发 · ReaLHF 归档 → AReaL · TorchForge → TorchTitan · Cosmos-RL 有限维护") }
  ];
  var ANSWERS = {
    offline: { c: C.blue, t: bi("Offline", "离线"), items: [
      bi("**TRL** (+ PEFT) for SFT / DPO", "SFT / DPO 用 **TRL**（+ PEFT）"),
      bi("Scale: **DeepSpeed** or **FSDP2**", "扩展规模：**DeepSpeed** 或 **FSDP2**"),
      bi("Recipes: LLaMA-Factory · ms-swift · Axolotl", "配方：LLaMA-Factory · ms-swift · Axolotl"),
      bi("Memory-bound: **QLoRA** · **Unsloth**", "显存受限：**QLoRA** · **Unsloth**"),
      bi("TPU / JAX: **Tunix**", "TPU / JAX：**Tunix**")] },
    reuse: { c: C.teal, t: bi("Reuse a complete runtime", "复用完整运行时"), items: [
      bi("Default prototype: **SkyRL** (FSDP2 + vLLM)", "默认原型：**SkyRL**（FSDP2 + vLLM）"),
      bi("Team platform, many recipes: **verl V1**", "团队平台、配方丰富：**verl V1**"),
      bi("HF + DeepSpeed assets: **OpenRLHF**", "已有 HF + DeepSpeed 资产：**OpenRLHF**"),
      bi("Megatron + big MoE: **slime** · **Miles** · **NeMo RL**", "Megatron + 大 MoE：**slime** · **Miles** · **NeMo RL**"),
      bi("Multi-domain pipelines: **ROLL**", "多领域流水线：**ROLL**")] },
    own: { c: C.amber, t: bi("Own the learner", "自己持有 learner"), items: [
      bi("New critics / Q / target, replay, UTD, inner loops", "新的 critic / Q / target、回放、UTD、内层循环"),
      bi("+ **FSDP2 / DeepSpeed** for execution", "+ 用 **FSDP2 / DeepSpeed** 负责执行"),
      bi("+ **vLLM / SGLang** + weight sync for generation", "+ 用 **vLLM / SGLang** + 权重同步负责生成"),
      bi("Keep the algorithm in plain PyTorch", "算法保持为纯 PyTorch")] }
  };
  var SPECIAL = [
    { c: C.violet, t: bi("Existing agent harness", "已有 Agent 执行框架"), b: bi("rLLM · Agent Lightning · ART", "rLLM · Agent Lightning · ART") },
    { c: C.green, t: bi("Async is the research", "异步本身是研究课题"), b: bi("AReaL · PipelineRL (compare SkyRL, Miles)", "AReaL · PipelineRL（对照 SkyRL、Miles）") },
    { c: C.blue, t: bi("Embodied / VLA", "具身 / VLA"), b: bi("RLinf (+ verl-vla)", "RLinf（并考察 verl-vla）") },
    { c: C.pink, t: bi("No GPU operations", "不想运维 GPU"), b: bi("Tinker: you write the loop; LoRA-centric", "Tinker：你写循环；以 LoRA 为主") }
  ];
  var CONTRACT = [
    bi("raw `token_ids` per turn (no re-tokenize)", "每轮原始 `token_ids`（不重新分词）"),
    bi("attention / loss / segment masks", "attention / loss / segment 掩码"),
    bi("`behavior_logprobs` + temperature, top-p, top-k", "`behavior_logprobs` + temperature、top-p、top-k"),
    bi("old / current / ref / teacher logprobs", "old / current / ref / teacher logprob"),
    bi("group / episode / turn / sample IDs", "group / episode / turn / sample ID"),
    bi("reward · advantage · return · value (shape, dtype, stop-grad)", "reward · advantage · return · value（形状、dtype、stop-grad）"),
    bi("policy version per token span", "按 token 片段记录策略版本"),
    bi("terminated vs truncated", "正常终止 vs 截断"),
    bi("sampler support · MoE routes (when needed)", "采样支持集 · MoE 路由（需要时）"),
    bi("sample weight · reuse count · source · filter reason", "样本权重 · 使用次数 · 来源 · 过滤原因")
  ];
  var CODECHECK = [
    bi("Final mean over tokens, sequences, ranks or micro-batches, and does the engine rescale again?", "最终均值作用于 token、序列、rank 还是微批？引擎是否又缩放一次？"),
    bi("Group reward statistics aggregated across ranks for the same prompt?", "同一 prompt 组的奖励统计是否跨 rank 聚合？"),
    bi("top-p log-probs normalized on the sampling support?", "top-p 对数概率是否在采样支持集上归一化？"),
    bi("Any branch that silently detaches a tensor or converts it to numpy?", "是否有分支悄悄 detach 张量或转成 numpy？"),
    bi("Weight sync: whole model? quantization post-processing? prefix / KV invalidated?", "权重同步：是否完整？是否执行量化后处理？prefix / KV 是否失效？"),
    bi("Buffer deletes after sampling? Resume restores optimizer, sampler, unconsumed data, versions?", "buffer 采样后是否删除？恢复时是否还原优化器、采样器、未消费数据与版本？")
  ];
  var MODS = [["trajectory", C.amber, bi("own", "自持")], ["algorithm", C.amber, bi("own", "自持")], ["learner", C.amber, bi("own / reuse", "自持/复用")],
    ["rollout", C.teal, bi("reuse", "复用")], ["runtime", C.teal, bi("reuse", "复用")], ["evaluation", C.blue, bi("define", "定义")]];
  var PSEUDO = [
    "while not budget.exhausted():",
    "    traces  = rollout.collect(prompts, policy_version)",
    "    records = trajectory.encode(traces)",
    "    replay.add(records)",
    "    for spec in algorithm.plan_updates(replay):",
    "        batch = replay.sample(spec)",
    "        out   = learner.forward(batch, spec)",
    "        loss, metrics = algorithm.objective(out, batch)",
    "        learner.backward_and_step(loss, spec)",
    "        algorithm.update_targets_if_needed(learner)",
    "    if algorithm.should_publish():",
    "        policy_version = runtime.publish_weights(learner)"
  ];
  var TAKEAWAYS = [
    { c: C.amber, t: bi("One gradient, two choices", "一个梯度，两个选择"), b: bi("Data source μ and weight ψ. μ fixed → trainer problem; μ = π_θ → systems problem.", "数据来源 μ 与权重 ψ。μ 固定 → 训练器问题；μ = π_θ → 系统问题。") },
    { c: C.blue, t: bi("Compose by layer", "按层组合"), b: bi("One component per role; verify the whole path (model ↔ backend ↔ weight sync ↔ engine ↔ trajectory ↔ update) before trusting a feature matrix.", "每个角色一个部件；在相信功能矩阵之前，先验证整条路径（模型 ↔ 后端 ↔ 权重同步 ↔ 引擎 ↔ 轨迹 ↔ 更新）。") },
    { c: C.teal, t: bi("Systems differ beyond algorithm lists", "差距不在算法列表"), b: bi("Scheduling, train–inference consistency, MoE depth, weight transfer and recovery decide the outcome.", "调度、训练推理一致性、MoE 深度、权重传输与恢复能力决定结果。") },
    { c: C.violet, t: bi("Own the math, reuse the runtime", "掌握数学，复用运行时"), b: bi("Own the trajectory contract, objective and update rule; reuse engines and sharding backends.", "自己掌握轨迹契约、目标函数与更新规则；复用推理引擎和分片后端。") },
    { c: C.coral, t: bi("Optimize cost-to-target", "优化达标成本"), b: bi("One bottleneck at a time, anchored by a reference implementation.", "一次解决一个瓶颈，以参考实现为锚点。") }
  ];
  var TRACKS = [
    { s: "\\mu", c: C.blue, n: bi("behavior", "行为策略"), d: bi("What the engine actually sampled (rollout logprobs).", "引擎实际采样的分布（rollout logprob）。") },
    { s: "\\pi_{\\text{old}}", c: C.teal, n: bi("old / proximal", "旧策略"), d: bi("Trainer recompute at step start, frozen.", "步首由训练端重算，固定不变。") },
    { s: "\\pi_\\theta", c: C.amber, n: bi("current", "当前策略"), d: bi("Differentiable, being updated.", "可微，正在被更新。") },
    { s: "\\pi_{\\text{ref}},\\ \\pi_{\\text{T}}", c: C.violet, n: bi("anchor / teacher", "参考 / 教师"), d: bi("KL anchor or distillation signal.", "KL 锚点或蒸馏信号。") }
  ];

  // ============================================================ RENDERERS
  function table(el, head, rows) {
    var h = "<table><thead><tr>" + head.map(function (x) { return "<th>" + esc(T(x)) + "</th>"; }).join("") + "</tr></thead><tbody>";
    rows.forEach(function (r) { h += "<tr>" + r.map(function (x) { return "<td>" + md(T(x)) + "</td>"; }).join("") + "</tr>"; });
    el.innerHTML = h + "</tbody></table>";
  }

  // --- methods
  var curMethod = "sft";
  function renderMethods() {
    $("#methods").innerHTML = METHODS.map(function (m) {
      return '<button class="chip" data-m="' + m.id + '" aria-pressed="' + (m.id === curMethod) + '">' + esc(T(m.name)) + "</button>";
    }).join("");
    var m = METHODS.filter(function (x) { return x.id === curMethod; })[0];
    var sys = m.regime === "systems";
    $("#method-card").innerHTML =
      '<div><span class="mlabel" style="color:var(--cyan)">' + (lang === "zh" ? "数据来源 μ" : "DATA SOURCE μ") + '</span><div class="mval">' + esc(T(m.mu)) + "</div>" +
      '<span class="regime" style="color:' + (sys ? "var(--coral)" : "var(--blue)") + '">' +
      (sys ? (lang === "zh" ? "μ = π_θ · 系统问题" : "μ = π_θ · SYSTEMS PROBLEM") : (lang === "zh" ? "μ 固定 · 训练器问题" : "μ FIXED · TRAINER PROBLEM")) + "</span></div>" +
      '<div><span class="mlabel" style="color:var(--amber)">' + (lang === "zh" ? "TOKEN 权重 ψ" : "TOKEN WEIGHT ψ") + '</span><div class="mval overflow-x">' + tex(m.psi, false) + '</div><p class="muted" style="margin-top:14px;font-size:15px">' + esc(T(m.note)) + "</p></div>";
    $$("#methods .chip").forEach(function (b) { b.addEventListener("click", function () { curMethod = b.dataset.m; renderMethods(); }); });
    renderMath($("#method-card"));
  }

  // --- stack
  var curLayer = 0;
  function renderLayers() {
    $("#layers").innerHTML = LAYERS.map(function (L, i) {
      return '<button class="layer" style="--c:' + L.c + '" data-l="' + i + '" aria-pressed="' + (i === curLayer) + '">' +
        '<span class="ix">0' + (i + 1) + '</span><span class="nm">' + esc(T(L.name)) + '</span><span class="sub">' + esc(T(L.does)) + "</span></button>";
    }).join("");
    $$("#layers .layer").forEach(function (b) {
      b.addEventListener("click", function () { curLayer = +b.dataset.l; renderLayers(); });
    });
    var L = LAYERS[curLayer];
    var d = $("#layer-detail");
    d.style.setProperty("--c", L.c);
    d.innerHTML = '<span class="tag" style="color:' + L.c + '">' + (lang === "zh" ? "第 " + (curLayer + 1) + " 层" : "Layer 0" + (curLayer + 1)) + "</span>" +
      "<h3>" + esc(T(L.name)) + '</h3><p class="mono" style="font-size:13.5px;color:var(--muted)">' + esc(T(L.does)) + "</p>" +
      '<div class="tools">' + L.tools.map(function (t) { return "<span>" + esc(t) + "</span>"; }).join("") + "</div>" +
      '<ul class="notes">' + L.notes.map(function (n) { return "<li>" + md(T(n)) + "</li>"; }).join("") + "</ul>";
  }

  // --- calculators
  function slider(id, label, min, max, step, val, fmt) {
    return '<div class="row"><label for="' + id + '">' + label + '</label><input type="range" id="' + id + '" min="' + min + '" max="' + max + '" step="' + step + '" value="' + val + '"><output id="' + id + '-o"></output></div>';
  }
  function bars(rows, max, log) {
    return rows.map(function (r) {
      var w = log ? Math.max(0.5, 100 * Math.log10(1 + r.v) / Math.log10(1 + max)) : Math.max(0.4, 100 * r.v / max);
      return '<div class="bar" style="--c:' + r.c + '"><span class="k">' + esc(r.k) + '</span><div class="track"><div class="fill" style="width:' + w.toFixed(2) + '%"></div></div><span class="v">' + r.s + "</span></div>";
    }).join("");
  }
  function fill(inp) { var p = (inp.value - inp.min) / (inp.max - inp.min) * 100; inp.style.setProperty("--p", p + "%"); }
  function gb(x) { return x >= 100 ? x.toFixed(0) : x >= 10 ? x.toFixed(1) : x.toFixed(2); }

  var DK = [512, 1024, 2048, 3072, 4096, 5120, 7168, 8192, 12288, 16384];
  function setupCalcs() {
    var L = lang === "zh";
    // LoRA
    var el = $("#calc-lora");
    var st = { dk: el.dataset.dk || 4, r: el.dataset.r || 8 };
    el.innerHTML = slider("lo-dk", "d = k", 0, DK.length - 1, 1, st.dk) + slider("lo-r", "rank r", 1, 256, 1, st.r) +
      '<div class="big-number" id="lo-pct"></div><p class="fine" id="lo-txt"></p>';
    function upLoRA() {
      var d = DK[+$("#lo-dk").value], r = +$("#lo-r").value;
      el.dataset.dk = $("#lo-dk").value; el.dataset.r = r;
      $("#lo-dk-o").textContent = d; $("#lo-r-o").textContent = r;
      var tr = r * (d + d), full = d * d, pct = 100 * tr / full;
      $("#lo-pct").innerHTML = (pct < 1 ? pct.toFixed(2) : pct.toFixed(1)) + "%<small>" + (L ? "可训练" : "trainable") + "</small>";
      $("#lo-txt").textContent = (L ? "每个投影矩阵：" : "Per projection: ") + tr.toLocaleString() + (L ? " 个可训练参数，对比 " : " trainable vs ") + full.toLocaleString() + (L ? "（全量）。" : " (full).");
      $$("#calc-lora input").forEach(fill);
    }
    $$("#calc-lora input").forEach(function (i) { i.addEventListener("input", upLoRA); }); upLoRA();

    // memory per parameter
    var em = $("#calc-mem");
    var psi0 = em.dataset.psi || 7;
    em.innerHTML = slider("me-psi", L ? "模型 Ψ (B)" : "model Ψ (B)", 0.5, 72, 0.5, psi0) + '<div class="bars" id="me-bars"></div><p class="fine">' +
      (L ? "权重与优化器状态，未计激活；LoRA/QLoRA 另加 adapter 及其优化器状态。QLoRA 的 NF4 约 0.5 字节/参数（含量化常数略高）。" :
        "Weights and optimizer state, before activations; LoRA/QLoRA add adapters and their optimizer state. NF4 is about 0.5 byte/param (slightly more with quantization constants).") + "</p>";
    function upMem() {
      var p = +$("#me-psi").value; em.dataset.psi = p; $("#me-psi-o").textContent = p + "B";
      var rows = [{ k: L ? "全参数 Adam" : "Full FT (Adam)", v: 16 * p, c: "var(--coral)" }, { k: L ? "LoRA bf16 基座" : "LoRA, bf16 base", v: 2 * p, c: "var(--amber)" }, { k: L ? "QLoRA NF4 基座" : "QLoRA, NF4 base", v: 0.5 * p, c: "var(--teal)" }];
      rows.forEach(function (r) { r.s = "≈ " + gb(r.v) + " GB"; });
      $("#me-bars").innerHTML = bars(rows, 16 * p);
      fill($("#me-psi"));
    }
    $("#me-psi").addEventListener("input", upMem); upMem();

    // ZeRO
    var ez = $("#calc-zero");
    var zs = { psi: ez.dataset.psi || 7.5, n: ez.dataset.n || 6 };
    var NS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
    ez.innerHTML = slider("ze-psi", L ? "模型 Ψ (B)" : "model Ψ (B)", 0.5, 200, 0.5, zs.psi) + slider("ze-n", L ? "GPU 数 N" : "GPUs N", 0, NS.length - 1, 1, zs.n) +
      '<div class="bars" id="ze-bars"></div><p class="fine">' + (L ? "K = 12（混合精度 Adam）。默认值 Ψ = 7.5B、N = 64 即 ZeRO 论文示例：120 / 31.4 / 16.6 / 1.9 GB。对数刻度。" :
        "K = 12 (mixed-precision Adam). Defaults Ψ = 7.5B, N = 64 reproduce the ZeRO paper: 120 / 31.4 / 16.6 / 1.9 GB. Log scale.") + "</p>";
    function upZero() {
      var p = +$("#ze-psi").value, n = NS[+$("#ze-n").value], K = 12;
      ez.dataset.psi = p; ez.dataset.n = $("#ze-n").value;
      $("#ze-psi-o").textContent = p + "B"; $("#ze-n-o").textContent = n;
      var rows = [
        { k: "DDP", v: (4 + K) * p, c: "var(--coral)" },
        { k: "ZeRO-1", v: 4 * p + K * p / n, c: "var(--amber)" },
        { k: "ZeRO-2", v: 2 * p + (2 + K) * p / n, c: "var(--violet)" },
        { k: "ZeRO-3 / FSDP2", v: (4 + K) * p / n, c: "var(--teal)" }];
      rows.forEach(function (r) { r.s = gb(r.v) + " GB"; });
      $("#ze-bars").innerHTML = bars(rows, rows[0].v, true);
      $$("#calc-zero input").forEach(fill);
    }
    $$("#calc-zero input").forEach(function (i) { i.addEventListener("input", upZero); }); upZero();

    // step time
    var es = $("#calc-step");
    var sv = es.dataset.v ? JSON.parse(es.dataset.v) : { g: 60, r: 8, t: 30, w: 12 };
    es.innerHTML = slider("st-g", L ? "生成 T_gen" : "T_gen (s)", 1, 120, 1, sv.g) + slider("st-r", L ? "奖励 T_rew" : "T_rew (s)", 0, 60, 1, sv.r) +
      slider("st-t", L ? "训练 T_train" : "T_train (s)", 1, 120, 1, sv.t) + slider("st-w", L ? "同步 T_w" : "T_w (s)", 0, 60, 1, sv.w) +
      '<div class="gantt" id="st-gantt"></div><div class="legend"><span><i style="background:var(--teal)"></i>' + (L ? "生成" : "generate") + '</span><span><i style="background:var(--amber)"></i>' + (L ? "奖励" : "reward") +
      '</span><span><i style="background:var(--blue)"></i>' + (L ? "训练" : "train") + '</span><span><i style="background:var(--coral)"></i>' + (L ? "权重同步" : "weight sync") + '</span></div><p class="fine" id="st-txt"></p>';
    function upStep() {
      var g = +$("#st-g").value, r = +$("#st-r").value, t = +$("#st-t").value, w = +$("#st-w").value;
      es.dataset.v = JSON.stringify({ g: g, r: r, t: t, w: w });
      $("#st-g-o").textContent = g + "s"; $("#st-r-o").textContent = r + "s"; $("#st-t-o").textContent = t + "s"; $("#st-w-o").textContent = w + "s";
      var seq = g + r + t + w, pipe = Math.max(g + r, t) + w, M = seq;
      function blk(x0, d, c, lbl) { return d <= 0 ? "" : '<div class="blk" style="left:' + (100 * x0 / M) + "%;width:" + (100 * d / M) + "%;background:" + c + '">' + lbl + "</div>"; }
      var pg = (g + r >= t) ? 0 : t - g - r;
      $("#st-gantt").innerHTML =
        '<div class="lane"><span>' + (L ? "串行" : "sequential") + '</span><div class="tl">' + blk(0, g, "var(--teal)", "gen") + blk(g, r, "var(--amber)", "") + blk(g + r, t, "var(--blue)", "train") + blk(g + r + t, w, "var(--coral)", "") + "</div></div>" +
        '<div class="lane"><span>' + (L ? "流水化·生成" : "pipelined · gen") + '</span><div class="tl">' + blk(0, g, "var(--teal)", "gen") + blk(g, r, "var(--amber)", "") + "</div></div>" +
        '<div class="lane"><span>' + (L ? "流水化·训练" : "pipelined · train") + '</span><div class="tl">' + blk(0, t, "var(--blue)", "train") + blk(Math.max(g + r, t), w, "var(--coral)", "") + "</div></div>";
      $("#st-txt").innerHTML = (L ? "串行 " : "Sequential ") + "<b class='acc'>" + seq + "s</b> · " + (L ? "流水化 ≈ " : "pipelined ≈ ") + "<b class='acc'>" + pipe + "s</b> (" + (seq / pipe).toFixed(2) + "×). " +
        (L ? "重叠的前提是允许一步的策略滞后；权重同步仍在关键路径上。" : "Overlap requires accepting one step of policy lag; weight sync stays on the critical path.");
      $$("#calc-step input").forEach(fill);
    }
    $$("#calc-step input").forEach(function (i) { i.addEventListener("input", upStep); }); upStep();
  }

  // --- loop diagram (SVG)
  function renderLoop() {
    var L = lang === "zh";
    var nodes = [
      { x: 300, y: 70, t: L ? "提示 / 任务" : "Prompts / tasks", s: "", c: "#8c94a8" },
      { x: 500, y: 175, t: L ? "Rollout 引擎" : "Rollout engine", s: "vLLM · SGLang", c: "#3fe0b8" },
      { x: 500, y: 345, t: L ? "奖励 · 环境 · 验证器" : "Reward · env · verifier", s: "", c: "#ffb547" },
      { x: 300, y: 450, t: L ? "优势估计" : "Advantage", s: "GAE · group norm", c: "#a98bff" },
      { x: 100, y: 345, t: L ? "学习器更新" : "Learner update", s: "FSDP2 · DeepSpeed · Megatron", c: "#5f9dff" },
      { x: 100, y: 175, t: L ? "发布权重" : "Publish weights", s: L ? "版本 v+1" : "version v+1", c: "#ff6f73" }
    ];
    var s = '<svg class="loop-svg" viewBox="-20 0 640 520" role="img" aria-label="RL loop"><defs><filter id="glow"><feGaussianBlur stdDeviation="3"/></filter></defs>';
    s += '<ellipse cx="300" cy="260" rx="200" ry="190" fill="none" stroke="rgba(255,255,255,0.08)"/>';
    s += '<path id="orbit" d="M300,70 A200,190 0 1,1 299.9,70" fill="none" stroke="none"/>';
    s += '<circle r="5" fill="#fff" filter="url(#glow)"><animateMotion dur="7s" repeatCount="indefinite"><mpath href="#orbit"/></animateMotion></circle>';
    s += '<circle r="2.5" fill="#fff"><animateMotion dur="7s" repeatCount="indefinite"><mpath href="#orbit"/></animateMotion></circle>';
    s += '<text x="300" y="255" text-anchor="middle" fill="#5b6478" font-size="13" letter-spacing="3">' + (L ? "一个 RL 步" : "ONE RL STEP") + "</text>";
    s += '<text x="300" y="278" text-anchor="middle" fill="#5b6478" font-size="11">μ = π_θ</text>';
    nodes.forEach(function (n) {
      s += '<g><rect x="' + (n.x - 95) + '" y="' + (n.y - 30) + '" width="190" height="' + (n.s ? 60 : 44) + '" rx="10" fill="#0a0d15" stroke="' + n.c + '" stroke-opacity="0.7"/>';
      s += '<text x="' + n.x + '" y="' + (n.y - (n.s ? 6 : -1)) + '" text-anchor="middle" fill="' + n.c + '" font-size="13.5" font-family="Inter, Noto Sans SC, sans-serif" font-weight="500">' + esc(n.t) + "</text>";
      if (n.s) s += '<text x="' + n.x + '" y="' + (n.y + 15) + '" text-anchor="middle" fill="#8c94a8" font-size="10.5">' + esc(n.s) + "</text>";
      s += "</g>";
    });
    s += "</svg>";
    $("#loop-diagram").innerHTML = s;
    $("#tracks").innerHTML = TRACKS.map(function (t) {
      return '<div class="ptrack" style="--c:' + t.c + '"><div class="sym">' + tex(t.s) + '</div><div class="nm">' + esc(T(t.n)) + "</div><p>" + esc(T(t.d)) + "</p></div>";
    }).join("");
    renderMath($("#tracks"));
  }

  // --- tables
  function renderTables() {
    table($("#tbl-step"), [bi("Case", "案例"), bi("Setup", "配置"), bi("Result", "结果"), bi("Lesson", "说明")], STEP_ROWS);
    table($("#tbl-async"), [bi("Mechanism", "机制"), bi("What it constrains", "它实际约束什么"), bi("What it does NOT guarantee", "它不保证什么")], ASYNC_ROWS);
    var p = document.createElement("p"); p.className = "fine"; p.style.cssText = "padding:14px 18px;margin:0;border-top:1px solid var(--line)"; p.textContent = T(ASYNC_FW);
    $("#tbl-async").appendChild(p);
    table($("#tbl-cons"), [bi("Error source", "误差来源"), bi("What to do", "需要解决什么"), bi("Representative mechanism", "代表性机制")], CONS_ROWS);
    table($("#tbl-free"), [bi("Freedom", "自由度"), bi("Typical research change", "典型研究改动"), bi("Control you actually need", "实际需要的控制")], FREE_ROWS);
    table($("#tbl-entry"), [bi("Route", "路线"), bi("Built-in entry points", "内置入口"), bi("Beyond that, you modify", "超出接口后要改哪里")], ENTRY_ROWS);
  }

  // --- landscape
  var curFilter = "all";
  function renderLandscape() {
    $("#filters").innerHTML = FILTERS.map(function (f) { return '<button class="chip" data-f="' + f[0] + '" aria-pressed="' + (f[0] === curFilter) + '">' + esc(T(f[1])) + "</button>"; }).join("");
    $$("#filters .chip").forEach(function (b) { b.addEventListener("click", function () { curFilter = b.dataset.f; renderLandscape(); }); });
    $("#frameworks").innerHTML = FRAMEWORKS.map(function (f) {
      var on = curFilter === "all" || f.tags.indexOf(curFilter) >= 0;
      return '<div class="card accent-top fw' + (on ? "" : " dim") + '" style="--c:' + f.c + '"><div class="name">' + esc(f.n) + '</div><div class="be">' + esc(f.be) + "</div><p>" + esc(T(f.d)) + "</p></div>";
    }).join("");
    $("#landscape-extra").innerHTML = LAND_EXTRA.map(function (x) {
      return '<div class="card" style="--c:' + x.c + '"><span class="tag">' + esc(T(x.t)) + '</span><p style="font-size:14px">' + esc(T(x.b)) + "</p></div>";
    }).join("");
  }

  // --- decision
  var q1 = null, q2 = null;
  function renderDecision() {
    $("#q2").className = "q " + (q1 === "yes" ? "on" : "off");
    $$("[data-q1]").forEach(function (b) { b.setAttribute("aria-pressed", String(b.dataset.q1 === q1)); });
    $$("[data-q2]").forEach(function (b) { b.setAttribute("aria-pressed", String(b.dataset.q2 === q2)); });
    var key = q1 === "no" ? "offline" : q1 === "yes" && q2 === "yes" ? "reuse" : q1 === "yes" && q2 === "no" ? "own" : null;
    var a = $("#answer");
    if (!key) {
      a.innerHTML = '<span class="tag">' + (lang === "zh" ? "建议" : "Recommendation") + "</span><h3>" + (lang === "zh" ? "回答左侧的问题" : "Answer the questions") + '</h3><p class="muted">' +
        (lang === "zh" ? "两个问题决定路线：数据是否在线产生？你的想法是否落在现有接口内？" : "Two questions pick the route: is the data produced online, and does your idea fit an existing interface?") + "</p>";
      a.style.removeProperty("--c");
      return;
    }
    var A = ANSWERS[key];
    a.style.setProperty("--c", A.c);
    a.innerHTML = '<span class="tag" style="color:' + A.c + '">' + (lang === "zh" ? "建议" : "Recommendation") + "</span><h3>" + esc(T(A.t)) + "</h3><ul>" + A.items.map(function (i) { return "<li>" + md(T(i)) + "</li>"; }).join("") + "</ul>";
  }
  $$("[data-q1]").forEach(function (b) { b.addEventListener("click", function () { q1 = b.dataset.q1; if (q1 === "no") q2 = null; renderDecision(); }); });
  $$("[data-q2]").forEach(function (b) { b.addEventListener("click", function () { q2 = b.dataset.q2; renderDecision(); }); });

  function renderCompose() {
    $("#special").innerHTML = SPECIAL.map(function (x) { return '<div class="card accent-top" style="--c:' + x.c + '"><span class="tag">' + esc(T(x.t)) + '</span><p style="font-size:14px">' + esc(T(x.b)) + "</p></div>"; }).join("");
    function checklist(el, items, key) {
      var done = {};
      try { done = JSON.parse(localStorage.getItem(key) || "{}"); } catch (e) {}
      el.innerHTML = items.map(function (x, i) { return '<li data-i="' + i + '" class="' + (done[i] ? "done" : "") + '"><span class="bx"></span><span>' + md(T(x)) + "</span></li>"; }).join("");
      $$("li", el).forEach(function (li) {
        li.addEventListener("click", function () {
          li.classList.toggle("done"); done[li.dataset.i] = li.classList.contains("done");
          try { localStorage.setItem(key, JSON.stringify(done)); } catch (e) {}
        });
      });
    }
    checklist($("#contract"), CONTRACT, "pt101-contract");
    checklist($("#codecheck"), CODECHECK, "pt101-codecheck");
    $("#mods").innerHTML = MODS.map(function (m) { return '<div style="--c:' + m[1] + ';border-color:color-mix(in srgb,' + m[1] + ' 35%, transparent)">' + m[0] + "<small>" + esc(T(m[2])) + "</small></div>"; }).join("");
    var hl = [4, 7, 8];
    $("#pseudo").innerHTML = PSEUDO.map(function (l, i) { return hl.indexOf(i) >= 0 ? '<span class="hl">' + esc(l) + "</span>" : esc(l); }).join("\n");
    $("#takeaways-list").innerHTML = TAKEAWAYS.map(function (x, i) {
      return '<div class="tk" style="--c:' + x.c + '"><span class="ix">0' + (i + 1) + "</span><h3>" + esc(T(x.t)) + "</h3><p>" + esc(T(x.b)) + "</p></div>";
    }).join("");
  }

  // ============================================================ VIDEO
  var CH = window.CHAPTERS || { chapters: [], total: {} };
  var filmLang = lang;
  var film = $("#film-video");
  function setFilm(l, keepTime) {
    var t = keepTime ? film.currentTime : 0;
    var frac = keepTime && CH.total[filmLang] ? t / CH.total[filmLang] : 0;
    filmLang = l;
    $$("[data-film]").forEach(function (b) { b.setAttribute("aria-pressed", String(b.dataset.film === l)); });
    var other = l === "en" ? "zh" : "en";
    film.innerHTML = '<source src="media/full_' + l + '.mp4" type="video/mp4">' +
      '<track kind="subtitles" srclang="' + (l === "en" ? "en" : "zh") + '" label="' + (l === "en" ? "English" : "中文") + '" src="media/full_' + l + "." + l + '.vtt" default>' +
      '<track kind="subtitles" srclang="' + (other === "en" ? "en" : "zh") + '" label="' + (other === "en" ? "English" : "中文") + '" src="media/full_' + l + "." + other + '.vtt">';
    film.poster = "media/full_poster_" + l + ".jpg";
    film.load();
    if (frac > 0) film.addEventListener("loadedmetadata", function f() { film.currentTime = frac * film.duration; film.removeEventListener("loadedmetadata", f); });
    $("#dl-film").href = "media/full_" + l + ".mp4";
    $("#dl-srt-en").href = "media/full_" + l + ".en.srt";
    $("#dl-srt-zh").href = "media/full_" + l + ".zh.srt";
    $("#film-label").textContent = l === "en" ? "Deep dive · English · " + mmss(CH.total.en || 0) : "完整版 · 中文 · " + mmss(CH.total.zh || 0);
    renderChapters();
  }
  function renderChapters() {
    var box = $("#chapters");
    var head = box.querySelector(".head").outerHTML;
    box.innerHTML = head + CH.chapters.map(function (c, i) {
      return '<div class="chap" data-i="' + i + '" tabindex="0" role="button"><span class="t">' + mmss(c.start[filmLang]) + '</span><span class="p">' + esc(c.part[lang] || (lang === "zh" ? "开场" : "Opening")) +
        '</span><span class="n">' + esc(c.title[lang]) + "</span></div>";
    }).join("");
    $$(".chap", box).forEach(function (el) {
      function go() { var c = CH.chapters[+el.dataset.i]; film.currentTime = c.start[filmLang] + 0.05; film.play(); }
      el.addEventListener("click", go);
      el.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } });
    });
    markChapter();
  }
  function markChapter() {
    var t = film.currentTime || 0, idx = 0;
    CH.chapters.forEach(function (c, i) { if (c.start[filmLang] <= t + 0.2) idx = i; });
    $$(".chap").forEach(function (el, i) { el.classList.toggle("active", i === idx); });
  }
  film.addEventListener("timeupdate", markChapter);
  function fitChapters() {
    var box = $("#chapters"), fr = film.parentNode;
    box.style.maxHeight = window.innerWidth > 1080 ? fr.getBoundingClientRect().height + "px" : "420px";
  }
  window.addEventListener("resize", fitChapters);
  film.addEventListener("loadedmetadata", fitChapters);
  setTimeout(fitChapters, 50);
  $$("[data-film]").forEach(function (b) { b.addEventListener("click", function () { setFilm(b.dataset.film, true); }); });

  var tmode = "one";
  function renderTranscript() {
    $("#transcript").innerHTML = CH.chapters.map(function (c, i) {
      var body;
      if (tmode === "both") {
        body = '<div class="bil">' + c.beats.map(function (b) { return "<p>" + esc(b.en) + "</p><p>" + esc(b.zh) + "</p>"; }).join("") + "</div>";
      } else {
        body = c.beats.map(function (b) { return "<p>" + esc(b[lang]) + "</p>"; }).join("");
      }
      return "<details><summary><span class=\"i\">" + (i < 10 ? "0" : "") + i + '</span><span class="tt">' + esc(c.title[lang]) + '</span><a class="go" href="#films" data-seek="' + i + '">▶ ' + mmss(c.start[filmLang]) + "</a></summary><div class=\"body\">" + body + "</div></details>";
    }).join("");
    $$("[data-seek]").forEach(function (a) {
      a.addEventListener("click", function (e) {
        e.stopPropagation();
        var c = CH.chapters[+a.dataset.seek]; film.currentTime = c.start[filmLang] + 0.05; film.play();
      });
    });
  }
  $$("[data-tmode]").forEach(function (b) {
    b.addEventListener("click", function () {
      tmode = b.dataset.tmode;
      $$("[data-tmode]").forEach(function (x) { x.setAttribute("aria-pressed", String(x === b)); });
      renderTranscript();
    });
  });

  var trailer = $("#trailer-video");
  function setTrailer(l) {
    $$("[data-trailer]").forEach(function (b) { b.setAttribute("aria-pressed", String(b.dataset.trailer === l)); });
    var playing = !trailer.paused;
    trailer.innerHTML = '<source src="media/promo_' + l + '.mp4" type="video/mp4">' +
      '<track kind="subtitles" srclang="' + (l === "en" ? "en" : "zh") + '" label="' + (l === "en" ? "English" : "中文") + '" src="media/promo_' + l + '.vtt">';
    trailer.load();
    if (playing) trailer.play();
  }
  $$("[data-trailer]").forEach(function (b) { b.addEventListener("click", function () { setTrailer(b.dataset.trailer); }); });

  // ============================================================ chrome
  var nav = $("#nav"), prog = $("#progress");
  var sections = $$("section[id]");
  function onScroll() {
    var y = window.scrollY, h = document.documentElement.scrollHeight - window.innerHeight;
    prog.style.width = (h > 0 ? 100 * y / h : 0) + "%";
    nav.classList.toggle("scrolled", y > 20);
    var cur = null;
    sections.forEach(function (s) { if (s.getBoundingClientRect().top < window.innerHeight * 0.35) cur = s.id; });
    $$(".nav-links a").forEach(function (a) { a.classList.toggle("active", a.getAttribute("href") === "#" + cur); });
  }
  window.addEventListener("scroll", onScroll, { passive: true }); onScroll();

  var io = "IntersectionObserver" in window ? new IntersectionObserver(function (es) {
    es.forEach(function (e) { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } });
  }, { rootMargin: "0px 0px -8% 0px" }) : null;
  $$(".rv").forEach(function (el) { if (io) io.observe(el); else el.classList.add("in"); });

  // ============================================================ hero canvas
  (function hero() {
    var cv = $("#hero-canvas"); if (!cv) return;
    var ctx = cv.getContext("2d"), W = 0, H = 0, dpr = Math.min(2, window.devicePixelRatio || 1);
    var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    var cols = ["#ffb547", "#ff6f73", "#5f9dff", "#3fe0b8", "#a98bff", "#7fe28a"];
    var pts = [];
    for (var i = 0; i < 140; i++) pts.push({ x: Math.random(), y: Math.random(), z: Math.random(), s: 0.2 + Math.random() * 0.8 });
    function size() { W = cv.clientWidth; H = cv.clientHeight; cv.width = W * dpr; cv.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0); }
    size(); window.addEventListener("resize", size);
    var visible = true;
    if ("IntersectionObserver" in window) new IntersectionObserver(function (e) { visible = e[0].isIntersecting; }).observe(cv);
    var t0 = performance.now();
    function frame(now) {
      var t = (now - t0) / 1000;
      if (visible) {
        ctx.clearRect(0, 0, W, H);
        // six slabs, gently breathing, at the right
        var cx = W * (W > 900 ? 0.72 : 0.5), cy = H * 0.52, w = Math.min(560, W * 0.42), dx = w * 0.22, dy = -w * 0.08;
        for (var k = 0; k < 6; k++) {
          var y = cy + (k - 2.5) * Math.min(64, H * 0.065) + Math.sin(t * 0.6 + k * 0.7) * 3;
          var x = cx - w / 2;
          ctx.beginPath();
          ctx.moveTo(x, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w + dx, y + dy); ctx.lineTo(x + dx, y + dy); ctx.closePath();
          var a = 0.10 + 0.06 * Math.sin(t * 0.8 + k);
          ctx.fillStyle = cols[k] + "14"; ctx.fill();
          ctx.strokeStyle = cols[k]; ctx.globalAlpha = 0.25 + a; ctx.lineWidth = 1; ctx.stroke(); ctx.globalAlpha = 1;
          // a pulse traveling along each slab edge
          var p = ((t * 0.12 + k * 0.17) % 1);
          ctx.beginPath(); ctx.arc(x + p * w, y, 2, 0, 6.283); ctx.fillStyle = cols[k]; ctx.shadowColor = cols[k]; ctx.shadowBlur = 12; ctx.fill(); ctx.shadowBlur = 0;
        }
        // drifting dust
        for (var j = 0; j < pts.length; j++) {
          var q = pts[j];
          var px = ((q.x + t * 0.004 * q.s) % 1) * W, py = ((q.y + Math.sin(t * 0.2 + j) * 0.003) % 1) * H;
          ctx.globalAlpha = 0.12 + 0.25 * q.z; ctx.fillStyle = "#cfe0ff";
          ctx.fillRect(px, py, 1.2 * q.s + 0.4, 1.2 * q.s + 0.4);
        }
        ctx.globalAlpha = 1;
      }
      if (!reduce) requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  })();

  // ============================================================ boot
  function renderAll() {
    renderMethods(); renderLayers(); setupCalcs(); renderLoop(); renderTables(); renderLandscape(); renderDecision(); renderCompose();
    renderTranscript();
    renderChapters();
    renderMath(document);
  }
  onLang(function () { renderAll(); });
  setLang(lang);
  setFilm(lang, false);
  setTrailer(lang);
  if (!window.katex) window.addEventListener("load", function () { renderMath(document); });
})();
