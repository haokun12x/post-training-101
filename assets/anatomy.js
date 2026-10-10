/* Anatomy data: atoms → recipes, formula positions → framework addresses, trade-off axes.
   Sources: reports/post-training-investigate.md and reports/post_training_engineering_review_2026-10-10.pdf
   (source code pinned to releases as of 2026-10-09/10). Cells the review did not cover are null. */
(function () {
  "use strict";
  function bi(en, zh) { return { en: en, zh: zh }; }

  // ------------------------------------------------------------ columns
  var FW = [
    { id: "trl", n: "TRL" }, { id: "openrlhf", n: "OpenRLHF" }, { id: "verl", n: "verl V1" }, { id: "skyrl", n: "SkyRL" },
    { id: "slime", n: "slime" }, { id: "miles", n: "Miles" }, { id: "areal", n: "AReaL" }, { id: "nemo", n: "NeMo RL" },
    { id: "roll", n: "ROLL" }, { id: "prime", n: "prime-rl" }, { id: "own", n: bi("Own learner", "自有 learner") }
  ];

  // ------------------------------------------------------------ depth ladder
  var LEVELS = {
    0: { k: "L0", c: "var(--teal)", n: bi("Config", "配置"), d: bi("Flip a built-in option.", "打开一个内置选项。") },
    1: { k: "L1", c: "var(--green)", n: bi("Hook", "钩子"), d: bi("Plug your function or class into a declared extension point.", "把自己的函数或类接到框架声明的扩展点上。") },
    2: { k: "L2", c: "var(--amber)", n: bi("Override", "覆写"), d: bi("Subclass and override one stage of the trainer or worker.", "继承并覆写 trainer 或 worker 的某个阶段。") },
    3: { k: "L3", c: "var(--coral)", n: bi("Fork", "改源码"), d: bi("Edit a few known framework files (a small fork).", "直接修改框架里几个明确的文件（小 fork）。") },
    4: { k: "L4", c: "var(--pink)", n: bi("Cross", "跨边界"), d: bi("The change crosses worker / backend / runtime boundaries.", "改动要跨越 worker / 后端 / 运行时的边界。") },
    own: { k: "★", c: "var(--cyan)", n: bi("Yours", "自持"), d: bi("You write it; the cost moves into adapters to engines and runtime.", "你自己写；成本转移到与推理引擎、运行时的适配上。") },
    na: { k: "✕", c: "var(--faint)", n: bi("Not exposed", "不开放"), d: bi("Not reachable on this route.", "这条路线上无法触及。") }
  };

  // ------------------------------------------------------------ atoms
  var ATOMS = [
    { id: "fsdp2", n: "FSDP2", g: "exec" }, { id: "zero", n: "DeepSpeed ZeRO", g: "exec" }, { id: "megatron", n: "Megatron-Core", g: "exec" },
    { id: "titan", n: "TorchTitan", g: "exec" }, { id: "archon", n: "Archon", g: "exec" }, { id: "dtensor", n: "DTensor / AutoModel", g: "exec" },
    { id: "vllm", n: "vLLM", g: "gen" }, { id: "sglang", n: "SGLang", g: "gen" },
    { id: "ray", n: "Ray", g: "orch" }, { id: "accel", n: "Accelerate", g: "orch" }, { id: "svc", n: bi("services", "服务化"), g: "orch" },
    { id: "k8s", n: "Kubernetes", g: "orch" }, { id: "hf", n: "Transformers", g: "model" }, { id: "peft", n: "PEFT", g: "model" }
  ];
  var GROUPS = {
    exec: { n: bi("Training execution", "训练执行"), c: "var(--blue)" },
    gen: { n: bi("Generation", "推理生成"), c: "var(--teal)" },
    orch: { n: bi("Orchestration", "编排"), c: "var(--coral)" },
    model: { n: bi("Model & params", "模型与参数化"), c: "var(--violet)" }
  };

  var RECIPES = [
    { id: "trl", exec: ["zero", "fsdp2"], gen: ["vllm"], orch: ["accel"], model: ["hf", "peft"],
      data: bi("Transformers `Trainer` batches; `_prepare_inputs` caches, splits and refreshes generations", "Transformers `Trainer` 批次；`_prepare_inputs` 负责生成缓存、拆分与刷新"),
      sync: bi("`trl/generation/vllm_generation.py`: FSDP2 `full_tensor`, ZeRO gather, PEFT merge/unmerge; server vs colocate", "`trl/generation/vllm_generation.py`：FSDP2 `full_tensor`、ZeRO gather、PEFT merge/unmerge；server 与 colocate 两种传输"),
      edge: [bi("Experimental `AsyncGRPOTrainer`: distributed path is FSDP2 only, not ZeRO.", "实验性 `AsyncGRPOTrainer`：分布式只支持 FSDP2，不支持 ZeRO。"),
             bi("A custom SFT `compute_loss_func` leaves the fused LM-head path and adds a no-grad forward.", "SFT 自定义 `compute_loss_func` 会离开 fused LM-head 路径，并额外做一次无梯度前向。")] },
    { id: "openrlhf", exec: ["zero"], gen: ["vllm"], orch: ["ray"], model: ["hf"],
      data: bi("`Experience` dataclass: sequences B×T; logprobs, mask, value, advantage B×A; reward B", "`Experience` dataclass：序列 B×T；logprob、mask、value、advantage 为 B×A；reward 为 B"),
      sync: bi("Broadcast to vLLM; ZeRO-3 needs `GatheredParameters`. Hybrid Engine (colocated) or async (queue + slot back-pressure)", "广播到 vLLM；ZeRO-3 需要 `GatheredParameters`。Hybrid Engine（共置）或异步（队列 + slot 背压）"),
      edge: [bi("Current guide: the Ray + vLLM online-RL path does not support LoRA.", "当前指南明确：Ray + vLLM 在线 RL 路径不支持 LoRA。"),
             bi("Async mode has limited compatibility with vLLM sleep.", "异步模式与 vLLM sleep 的兼容性有限。")] },
    { id: "verl", exec: ["fsdp2", "megatron"], gen: ["vllm", "sglang"], orch: ["ray"], model: ["hf"],
      data: bi("TransferQueue stores data, `KVBatchMeta` holds references, workers compute on TensorDict; DataProto at the advantage stage", "TransferQueue 存数据，`KVBatchMeta` 持引用，worker 在 TensorDict 上计算；优势阶段借用 DataProto"),
      sync: bi("HybridFlow separates control flow from distributed compute and reshards between training and generation; modes sync / colocate_async / separate_async", "HybridFlow 把控制流与分布式计算分开，并处理训练与生成间的布局切换；模式 sync / colocate_async / separate_async"),
      edge: [bi("Fully async is documented for a specific backend + vLLM server + AgentLoop route; do not infer SGLang × async × LoRA.", "fully async 的官方路线是特定训练后端 + vLLM server + AgentLoop；不能据此推断 SGLang × 异步 × LoRA 已验证。")] },
    { id: "skyrl", exec: ["fsdp2", "megatron"], gen: ["vllm"], orch: ["ray"], model: ["hf"],
      data: bi("`GeneratorOutput`: raw tokens, reward, loss mask, rollout logprobs, group ids, MoE routes, sampler support", "`GeneratorOutput`：原始 token、reward、loss mask、rollout logprob、组标识、MoE 路由、采样支持集"),
      sync: bi("Incremental weight sync, RDT + NIXL; also a JAX route and a Tinker-compatible API", "增量权重同步、RDT + NIXL；另有 JAX 路线与 Tinker 兼容 API"),
      edge: [bi("`MegatronCriticWorkerBase` is unimplemented: actor-critic PPO needs the FSDP path.", "`MegatronCriticWorkerBase` 未实现：actor-critic PPO 需要走 FSDP 路径。"),
             bi("VLM × padding removal × sequence parallel hits an assertion.", "VLM × padding removal × sequence parallel 存在组合断言。")] },
    { id: "slime", exec: ["megatron"], gen: ["sglang"], orch: ["ray"], model: [],
      data: bi("`Sample`: raw token IDs, response_length, loss_mask, rollout_log_probs, weight version; `rollout_id` affects sample weights", "`Sample`：原始 token ID、response_length、loss_mask、rollout_log_probs、权重版本；`rollout_id` 影响样本加权"),
      sync: bi("Megatron → SGLang; straw persists tasks, partial generations and batches across failures", "Megatron → SGLang；straw 让任务、部分生成结果和批次跨故障保留"),
      edge: [bi("FSDP was removed; Megatron + SGLang is the route.", "当前版本已移除 FSDP；路线就是 Megatron + SGLang。"),
             bi("Under TP, `logits` are vocab-sharded; Megatron applies outer micro-batch scaling.", "TP 下 `logits` 按词表分片；Megatron 外层还有微批缩放。")] },
    { id: "miles", exec: ["megatron", "fsdp2", "titan"], gen: ["sglang"], orch: ["ray", "k8s"], model: [],
      data: bi("Replaceable async DataBuffer (`put/get` see `current_version`, `trainer_model_id`); TITO keeps token-in/token-out; R3 routing replay", "可替换的异步 DataBuffer（`put/get` 可见 `current_version`、`trainer_model_id`）；TITO 保留 token 进出；R3 路由重放"),
      sync: bi("`WeightUpdater`: pause engine → send buckets → finish → publish version → resume; NCCL broadcast or P2P/RDMA", "`WeightUpdater`：暂停引擎 → 发送权重桶 → 结束更新 → 发布版本 → 恢复；NCCL 广播或 P2P/RDMA"),
      edge: [bi("Miles' FSDP2 backend: data-parallel only (no TP/PP/CP/EP), no LoRA. TorchTitan backend: no LoRA.", "Miles 的 FSDP2 后端只有数据并行（无 TP/PP/CP/EP），无 LoRA；TorchTitan 后端无 LoRA。"),
             bi("TITO sessions do not accept image/video input. Kubernetes route still in development; Ray is the default.", "TITO session 不接受图像/视频输入。Kubernetes 路线仍在开发，默认仍是 Ray。")] },
    { id: "areal", exec: ["fsdp2", "megatron", "archon"], gen: ["sglang", "vllm"], orch: ["svc", "ray"], model: [],
      data: bi("`RolloutWorkflow.arun_episode` episodes; token/turn-level signals; old / proximal / behavior probabilities kept apart", "`RolloutWorkflow.arun_episode` 产生 episode；token/turn 粒度信号；old / proximal / behavior 概率分开"),
      sync: bi("Separate training, inference, agent and weight-update services; 2.1 adds AWEX colocation and Ray scheduling", "training、inference、agent、weight-update 服务拆分；2.1 加入 AWEX 共置与 Ray 调度"),
      edge: [bi("Decoupled PPO combined with newer objectives (e.g. SAPO) is still a research question.", "解耦 PPO 与 SAPO 等新目标的组合仍是研究问题。")] },
    { id: "nemo", exec: ["dtensor", "megatron"], gen: ["vllm", "sglang"], orch: ["ray"], model: ["hf"],
      data: bi("`PolicyInterface` (logprob / top-k / train); legacy, SingleController and TransferQueue paths", "`PolicyInterface`（logprob / top-k / train）；legacy、SingleController、TransferQueue 三类路径"),
      sync: bi("Async GRPO requires separate resources and importance correction; multi-teacher distillation adds teacher placement", "异步 GRPO 要求分离资源并做重要性修正；多教师蒸馏还要安排教师资源"),
      edge: [bi("The async GRPO replay path excludes SGLang, although other paths support it.", "异步 GRPO replay 路径排除了 SGLang，尽管其他路径支持 SGLang。"),
             bi("Full-distribution OPD needs SingleController.", "全分布 OPD 的部分能力要求 SingleController。")] },
    { id: "roll", exec: ["fsdp2", "megatron"], gen: ["vllm", "sglang"], orch: ["ray"], model: [],
      data: bi("Pipeline → Cluster/Ray dispatch → Worker → Strategy; DynamicSamplingScheduler, Router, ReplayBuffer", "Pipeline → Cluster/Ray dispatch → Worker → Strategy；DynamicSamplingScheduler、Router、ReplayBuffer"),
      sync: bi("Async RLVR: inference and scoring continue during training; sampling pauses only before the next weight publication", "异步 RLVR：训练时推理和评分继续，下一轮权重发布前才暂停采样"),
      edge: [bi("The README capability table lags the source: async RLVR is already implemented.", "README 能力表落后于源码：异步 RLVR 已经实现。")] },
    { id: "prime", exec: ["fsdp2"], gen: ["vllm"], orch: ["svc"], model: [],
      data: bi("Orchestrator scores episodes/groups with verifiers environments; trainer consumes; `_drop_stale` before dequeue", "orchestrator 用 verifiers 环境给 episode / group 打分；trainer 消费；出队前 `_drop_stale`"),
      sync: bi("Asynchronous orchestrator ↔ trainer; models from a native implementation set", "orchestrator 与 trainer 异步；模型来自 native implementation 范围"),
      edge: [bi("Do not assume any HF architecture plugs in: supported models are an explicit set.", "不能假设任意 HF 架构都能接入：受支持模型是一个明确集合。")] },
    { id: "own", exec: ["fsdp2", "zero"], gen: ["vllm", "sglang"], orch: ["ray"], model: ["hf"],
      data: bi("Your trajectory contract: raw tokens, masks, behavior logprobs, group/turn ids, versions, terminated vs truncated", "你自己的轨迹契约：原始 token、mask、behavior logprob、group/turn ID、版本、终止与截断"),
      sync: bi("vLLM `WeightSource` / `TrainerWeightTransferEngine`, or SGLang update sessions (pause/continue, release/resume memory)", "vLLM `WeightSource` / `TrainerWeightTransferEngine`，或 SGLang 权重更新 session（暂停/继续、显存释放/恢复）"),
      edge: [bi("`Accelerator.prepare` (FSDP2 branch) rejects several models in one call: wire multi-network setups yourself.", "`Accelerator.prepare`（FSDP2 分支）拒绝一次传入多个模型：多网络需自己接线。"),
             bi("ZeRO-3 `step` takes no closure; don't double-scale gradient accumulation.", "ZeRO-3 的 `step` 不支持 closure；不要重复缩放梯度累积。")] }
  ];

  // ------------------------------------------------------------ trade-off axes
  var AXES = [
    { c: "var(--blue)", t: bi("Training backbone", "训练底座"),
      poles: [
        { n: "FSDP2", g: bi("Your PyTorch model and control flow stay intact.", "保留普通 PyTorch 模型与控制流。"), p: bi("Rollout, critic, sync and async are yours to wire.", "rollout、critic、同步与异步都要上层接线。") },
        { n: "DeepSpeed ZeRO", g: bi("Shards params, grads and optimizer state; offload extends scale.", "分片参数、梯度和优化器状态；offload 扩大可运行规模。"), p: bi("Engine owns accumulation & step; no DDP `no_sync`; ZeRO-3 step has no closure.", "引擎掌握累积与 step；不能照搬 `no_sync`；ZeRO-3 step 无 closure。") },
        { n: "Megatron-Core", g: bi("TP · PP · EP · CP and large MoE.", "TP · PP · EP · CP 与大 MoE。"), p: bi("Model mapping, schedule and weight export enter training; inner loops touch the schedule.", "模型映射、调度与权重导出深度参与训练；内层优化会触及调度。") }
      ],
      who: bi("FSDP2: SkyRL, prime-rl, TRL, Miles(DP) · ZeRO: OpenRLHF, TRL · Megatron: slime, Miles, NeMo RL; verl / ROLL / AReaL ship both", "FSDP2：SkyRL、prime-rl、TRL、Miles（仅 DP）· ZeRO：OpenRLHF、TRL · Megatron：slime、Miles、NeMo RL；verl / ROLL / AReaL 两者都有") },
    { c: "var(--teal)", t: bi("Generation engine", "推理引擎"),
      poles: [
        { n: "vLLM", g: bi("Native weight transfer: `WeightSource`, trainer & inference transfer engines; NCCL, IPC, resharding.", "原生权重传输：`WeightSource`、训练侧与推理侧传输引擎；NCCL、IPC、重分片。"), p: bi("`keep` mode resumes paused requests on new weights; `clear_cache` decides whether old KV survives.", "`keep` 模式让暂停请求用新权重续写；`clear_cache` 决定旧 KV 是否保留。") },
        { n: "SGLang", g: bi("Disk / tensor-IPC / distributed-group updates; pause/continue; memory release/resume.", "磁盘 / tensor-IPC / 分布式通信组更新；暂停/继续；显存释放/恢复。"), p: bi("Update sessions write buckets, then run quantization post-processing once at the end.", "更新 session 先写入各桶，结束时只做一次量化后处理。") }
      ],
      who: bi("vLLM: TRL, OpenRLHF, SkyRL, prime-rl · SGLang: slime, Miles · both: verl, NeMo RL, ROLL, AReaL", "vLLM：TRL、OpenRLHF、SkyRL、prime-rl · SGLang：slime、Miles · 两者皆可：verl、NeMo RL、ROLL、AReaL") },
    { c: "var(--coral)", t: bi("Placement", "资源布局"),
      poles: [
        { n: bi("Colocated · sync", "共置 · 同步"), g: bi("Fresh policy; fewer resident copies; phases share GPUs.", "策略新鲜；常驻副本少；阶段间复用 GPU。"), p: bi("Long-tail rollouts leave trainers idle.", "长尾 rollout 让训练资源空等。") },
        { n: bi("Separated · async", "分离 · 异步"), g: bi("Overlaps generation and training; hides the long tail.", "生成与训练重叠；掩盖长尾。"), p: bi("Weight transfer, staleness, buffers, importance correction. Not monotone: DeepSeek-V3 111→64.1 s/step, Qwen3-30B-A3B 221→301 s/step.", "权重传输、陈旧度、缓冲、重要性修正。并非单调：DeepSeek-V3 111→64.1 秒/步，Qwen3-30B-A3B 221→301 秒/步。") }
      ],
      who: bi("Async-first: AReaL, prime-rl · both modes: verl V1, OpenRLHF, SkyRL, slime, Miles, ROLL, NeMo RL · TRL async is experimental", "异步优先：AReaL、prime-rl · 两种模式：verl V1、OpenRLHF、SkyRL、slime、Miles、ROLL、NeMo RL · TRL 的异步仍是实验性") },
    { c: "var(--amber)", t: bi("Connective layer", "连接层厚度"),
      poles: [
        { n: bi("Thin", "薄"), g: bi("Few files between the math and the backend; hooks receive real tensors.", "数学与后端之间只隔几个文件；hook 拿到真实张量。"), p: bi("You inherit the backend's complexity directly (Megatron mapping, DeepSpeed rules).", "直接面对后端复杂度（Megatron 映射、DeepSpeed 约定）。") },
        { n: bi("Platform", "平台"), g: bi("Many backends, roles, tasks and run modes behind one runtime.", "一个运行时承载多后端、多角色、多任务、多运行模式。"), p: bi("Control & data abstractions to learn before a new learning rule is visible.", "看清新学习规则如何改参数之前，要先掌握控制与数据抽象。") }
      ],
      who: bi("Thin: slime, OpenRLHF, SkyRL · Platform: verl V1, ROLL, NeMo RL, AReaL", "薄：slime、OpenRLHF、SkyRL · 平台：verl V1、ROLL、NeMo RL、AReaL") }
  ];

  // Map placement (judgment): x = connective layer 0 thin … 1 platform; backbones spanned.
  var MAP = [
    { id: "trl", x: 0.02, b: ["hf"] }, { id: "slime", x: 0.13, b: ["mg"] }, { id: "openrlhf", x: 0.24, b: ["hf"] },
    { id: "skyrl", x: 0.35, b: ["hf", "mg"] }, { id: "prime", x: 0.46, b: ["hf"] }, { id: "miles", x: 0.57, b: ["hf", "mg"] },
    { id: "areal", x: 0.68, b: ["hf", "mg"] }, { id: "nemo", x: 0.79, b: ["hf", "mg"] }, { id: "roll", x: 0.89, b: ["hf", "mg"] },
    { id: "verl", x: 0.99, b: ["hf", "mg"] }
  ];

  // ------------------------------------------------------------ formula positions × frameworks
  // cell: [level, en, zh]; null = not covered by the source review
  function c(l, en, zh) { return [l, en, zh]; }
  var POS = [
    // ---------------- data rule
    { n: 1, g: "data", term: "x\\sim\\mathcal D", t: bi("Which prompts", "用哪些 prompt"), d: bi("Task mixture, curriculum, difficulty, dynamic sampling.", "任务混合、课程、难度、动态采样。"),
      cells: {
        trl: c(2, "Dataset is config; dynamic or curriculum sampling means overriding `_prepare_inputs` (cache, shuffle, split, refresh).", "数据集是配置；动态或课程采样要覆写 `_prepare_inputs`（缓存、shuffle、拆分、刷新）。"),
        verl: c(1, "Custom ReplayBuffer / sampler in V1.", "V1 支持自定义 ReplayBuffer / sampler。"),
        miles: c(1, "`custom_async_data_buffer_path`: select by difficulty, priority, version or source.", "`custom_async_data_buffer_path`：按难度、优先级、版本、来源选择数据。"),
        roll: c(0, "Multi-domain mixing via `DynamicSamplingScheduler`; `process_new_prompt(context)` for custom logic.", "`DynamicSamplingScheduler` 做多领域混采；`process_new_prompt(context)` 接自定义逻辑。"),
        prime: c(1, "Tasks come from the verifiers environment.", "任务来自 verifiers 环境。"),
        own: c("own", "Your sampler over your dataset.", "自己的数据集与采样器。") } },
    { n: 2, g: "data", term: "y\\sim\\mu", t: bi("Who generates, how", "谁生成、怎么生成"), d: bi("Generation logic, multi-turn, tools, environments.", "生成逻辑、多轮、工具、环境。"),
      cells: {
        trl: c(1, "`rollout_func` in GRPO.", "GRPO 的 `rollout_func`。"),
        openrlhf: c(1, "Agent training interface: multi-turn, unified token trajectories.", "Agent 训练接口：多轮交互、统一 token 轨迹。"),
        verl: c(1, "AgentLoop (the documented fully-async route).", "AgentLoop（官方 fully async 路线所用）。"),
        skyrl: c(1, "`GeneratorInterface.generate(GeneratorInput) → GeneratorOutput`.", "`GeneratorInterface.generate(GeneratorInput) → GeneratorOutput`。"),
        slime: c(1, "Custom generation; `RolloutManager.generate`.", "custom generation；`RolloutManager.generate`。"),
        miles: c(1, "Generation function; `FullyAsyncRolloutFn` produces in the background.", "生成函数；`FullyAsyncRolloutFn` 后台持续生产。"),
        areal: c(1, "`RolloutWorkflow.arun_episode`.", "`RolloutWorkflow.arun_episode`。"),
        nemo: c(1, "Environments via NeMo Gym.", "通过 NeMo Gym 接入环境。"),
        roll: c(1, "`user_defined_rollout_loop_cls`.", "`user_defined_rollout_loop_cls`。"),
        prime: c(1, "verifiers environments.", "verifiers 环境。"),
        own: c("own", "Your rollout on vLLM / SGLang; keep raw tokens, never re-tokenize text.", "在 vLLM / SGLang 上自己写 rollout；保存原始 token，不要重新 tokenize。") } },
    { n: 3, g: "data", term: "r(x,y)", t: bi("Reward", "奖励"), d: bi("Verifier, reward model, environment score.", "验证器、奖励模型、环境评分。"),
      cells: {
        trl: c(1, "`reward_funcs`.", "`reward_funcs`。"),
        skyrl: c(1, "Reward is a field of `GeneratorOutput` (generator / environment).", "reward 是 `GeneratorOutput` 的字段（由 generator / 环境给出）。"),
        areal: c(1, "Inside the workflow; `token_rewards` modes have constraints.", "在 workflow 内；`token_rewards` 的模式有约束。"),
        nemo: c(1, "NeMo Gym environments.", "NeMo Gym 环境。"),
        roll: c(0, "Reward roles and multi-reward clusters per domain.", "奖励角色，可按领域配置多个奖励集群。"),
        prime: c(1, "`Algorithm.score_episode` / `score_group`.", "`Algorithm.score_episode` / `score_group`。"),
        own: c("own", "Your verifier (e.g. Math-Verify); count parse failures, timeouts and truncations separately.", "自己的验证器（如 Math-Verify）；解析失败、超时、截断分开统计。") } },
    { n: 4, g: "data", term: "\\text{buffer}", t: bi("Data lifetime", "数据寿命"), d: bi("Replay, priority, reuse count, deletion.", "回放、优先级、重用次数、删除。"),
      cells: {
        trl: c(2, "Override `_prepare_inputs`: generation cache, reuse, split and refresh rules.", "覆写 `_prepare_inputs`：生成缓存、复用、拆分与刷新规则。"),
        verl: c(1, "Custom ReplayBuffer / sampler. The default buffer is not classic long-term replay.", "自定义 ReplayBuffer / sampler。默认 buffer 不是经典长期回放。"),
        skyrl: c(4, "Historical replay: extend worker, data protocol and learner.", "历史 replay：要扩展 worker、数据协议和 learner。"),
        miles: c(1, "Replace the DataBuffer. The default removes samples on consumption.", "替换 DataBuffer。默认 buffer 消费后即移除。"),
        nemo: c(0, "Async GRPO replay path with age filtering (excludes SGLang).", "异步 GRPO replay 路径，按年龄筛选（排除 SGLang）。"),
        roll: c(3, "ReplayBuffer keyed by `sampling_start_step`, removed on consumption; long-term reuse means editing it.", "ReplayBuffer 按 `sampling_start_step` 组织、消费即移除；长期重用要改它。"),
        own: c("own", "Your replay: write, sample, delete, reuse count, weights.", "自己的 replay：写入、采样、删除、重用次数、权重。") } },
    { n: 5, g: "data", term: "v\\ge v_{\\text{now}}-k", t: bi("How stale", "允许多旧"), d: bi("Async consumption: admission, back-pressure, age gate.", "异步消费：准入、背压、年龄门限。"),
      cells: {
        trl: c(0, "Experimental `AsyncGRPOTrainer` (FSDP2 only).", "实验性 `AsyncGRPOTrainer`（仅 FSDP2）。"),
        openrlhf: c(0, "Bounded queue + slot back-pressure; no per-sample version filter.", "有界队列 + slot 背压；没有逐样本版本过滤。"),
        verl: c(0, "`drop` or `wait`; its off-policy count is not other frameworks' unit.", "`drop` 或 `wait`；其 off-policy 计数单位与别的框架不同。"),
        skyrl: c(0, "`max_staleness_steps` acts on admission/capacity, not a strict per-trajectory age.", "`max_staleness_steps` 作用于准入/容量，不是每条轨迹的严格年龄。"),
        slime: c(1, "Back-pressure, age metrics and a filter extension; no default hard age gate.", "背压、年龄指标与过滤扩展；默认没有硬年龄门限。"),
        miles: c(0, "`max_weight_staleness`, evaluated at consumption; off when unset.", "`max_weight_staleness`，在消费时计算；未设置则不启用。"),
        areal: c(0, "Explicit policy-version lag control.", "显式控制策略版本滞后。"),
        nemo: c(0, "Consumer-side age control on the async path.", "异步路径上的消费侧年龄控制。"),
        roll: c(0, "Age window organized around the prompt's start step.", "年龄窗口围绕 prompt 起始训练步组织。"),
        prime: c(0, "`_drop_stale`: hard filter before dequeue.", "`_drop_stale`：出队前硬过滤。"),
        own: c("own", "Your consumption rule; stratify drops by task, length and age.", "自己的消费规则；按任务、长度、年龄分层观察丢弃。") } },
    { n: 6, g: "data", term: "w_t=\\tfrac{\\pi_{\\text{old}}}{\\mu}", t: bi("Behavior correction", "行为策略修正"), d: bi("Train–inference mismatch, off-policy correction.", "训练推理不一致、离策略修正。"),
      cells: {
        openrlhf: c(0, "`PolicyLoss` keeps current, old and rollout logprobs apart (`models/loss.py`).", "`PolicyLoss` 区分当前、old 与 rollout 三种 logprob（`models/loss.py`）。"),
        skyrl: c(0, "Sampler-support replay (top-p/top-k) and MoE routing replay.", "采样支持集重放（top-p/top-k）与 MoE 路由重放。"),
        slime: c(1, "Custom TIS (train–inference correction) hook.", "custom TIS（训练/推理偏差修正）入口。"),
        miles: c(0, "TITO (exact tokens) and R3 (routing replay).", "TITO（精确 token）与 R3（路由重放）。"),
        areal: c(0, "Decoupled PPO: old / proximal / behavior probabilities.", "解耦 PPO：old / proximal / behavior 概率。"),
        nemo: c(0, "Async path requires importance correction.", "异步路径要求重要性修正。"),
        own: c("own", "Store behavior logprobs from the engine, and say whether they are pre- or post-top-p.", "保存引擎给出的 behavior logprob，并注明是 top-p 截断前还是截断后。") } },
    // ---------------- learning rule
    { n: 7, g: "learn", term: "\\hat A_t", t: bi("Advantage", "优势估计"), d: bi("Baseline, return, credit, group statistics.", "baseline、回报、credit、组统计。"),
      cells: {
        trl: c(2, "Generation-and-scoring stage of the trainer.", "trainer 的生成与评分阶段。"),
        openrlhf: c(3, "`ppo_utils/experience_maker.py`; group rewards restored by prompt id after reordering.", "`ppo_utils/experience_maker.py`；重排后按 prompt 标识恢复组奖励。"),
        verl: c(1, "`register_adv_est`.", "`register_adv_est`。"),
        skyrl: c(1, "`AdvantageEstimatorRegistry`.", "`AdvantageEstimatorRegistry`。"),
        slime: c(1, "`custom_advantage(args, rollout_data)` writes advantages / returns in place.", "`custom_advantage(args, rollout_data)` 原地写入 advantages / returns。"),
        miles: c(3, "`hub/advantages.py`: if/elif dispatch, no external custom-advantage hook.", "`hub/advantages.py`：if/elif 派发，没有外部 custom-advantage hook。"),
        areal: c(2, "`PPOActor._compute_advantages` (token/turn GAE, truncation bootstrap).", "`PPOActor._compute_advantages`（token/turn GAE、截断 bootstrap）。"),
        roll: c(3, "Fixed branches in `compute_advantage`, or replace the pipeline call.", "`compute_advantage` 的固定分支，或替换 pipeline 调用。"),
        prime: c(1, "`score_group` decides the group signal.", "`score_group` 决定组信号。"),
        own: c("own", "Pure PyTorch; aggregate group statistics across ranks first.", "纯 PyTorch；先跨 rank 聚合同组统计。") } },
    { n: 8, g: "learn", term: "\\min(\\rho_t\\hat A_t,\\,\\mathrm{clip})", t: bi("Surrogate on sampled tokens", "采样 token 上的目标"), d: bi("Clip, IS ratio, KL estimator: anything that needs only selected-token log-probs.", "clip、重要性比率、KL 估计：只需选中 token logprob 的目标。"),
      cells: {
        trl: c(2, "Override `_compute_loss` (SFT: `compute_loss_func`).", "覆写 `_compute_loss`（SFT 用 `compute_loss_func`）。"),
        openrlhf: c(3, "`models/loss.py`: surrogate, clip, IS, reduction.", "`models/loss.py`：surrogate、clip、IS、reduction。"),
        verl: c(1, "`register_policy_loss`: selected-token logprob, advantage, mask, weights.", "`register_policy_loss`：选中 token logprob、advantage、mask、权重。"),
        skyrl: c(1, "`PolicyLossRegistry`.", "`PolicyLossRegistry`。"),
        slime: c(1, "`custom_loss(args, batch, logits, sum_of_sample_mean) → loss, metrics`.", "`custom_loss(args, batch, logits, sum_of_sample_mean) → loss, metrics`。"),
        miles: c(3, "`training_utils/loss/objective.py`, `hub/losses.py`.", "`training_utils/loss/objective.py`、`hub/losses.py`。"),
        areal: c(2, "Loss function handed to `engine.train_batch` from `PPOActor`.", "由 `PPOActor` 传给 `engine.train_batch` 的 loss 函数。"),
        nemo: c(1, "Implement `LossFunction` (declares token/sequence level and input type).", "实现 `LossFunction`（声明 token/sequence 级与输入类型）。"),
        roll: c(2, "Custom ActorWorker via `actor_train.worker_cls`, override `loss_func(data, output_tensor)`.", "用 `actor_train.worker_cls` 指定自定义 ActorWorker，覆写 `loss_func(data, output_tensor)`。"),
        prime: c(1, "`CustomLoss` on per-sample selected-token logprobs.", "`CustomLoss`，输入为每条样本的选中 token logprob。"),
        own: c("own", "Pure PyTorch reference loss; keep it to check every fused kernel.", "纯 PyTorch 参考 loss；保留它来核对每个融合 kernel。") } },
    { n: 9, g: "learn", term: "\\pi_\\theta(\\cdot\\mid h_t)", t: bi("Full distribution or hidden state", "全分布或 hidden state"), d: bi("Exact entropy, full-vocab KL, top-k / full distillation.", "精确熵、全词表 KL、top-k / 全分布蒸馏。"),
      cells: {
        trl: c(1, "SFT `compute_loss_func` gets full logits, but leaves the fused LM-head path.", "SFT `compute_loss_func` 拿到全 logits，但离开 fused LM-head 路径。"),
        openrlhf: c(3, "`ray/ppo_actor.py`: entropy, auxiliary objectives, optimizer wiring.", "`ray/ppo_actor.py`：熵、辅助目标、优化器接线。"),
        verl: c(4, "Policy-loss registry passes selected-token logprobs; full logits/hidden need worker / model-output changes.", "policy-loss 注册表只传选中 token logprob；全 logits / hidden 需改 worker / model-output 层。"),
        skyrl: c(4, "Beyond the registries: the model worker.", "超出注册表：要改 model worker。"),
        slime: c(1, "`custom_loss` receives differentiable `logits` (vocab-sharded under TP) and the reducer.", "`custom_loss` 直接拿到可微 `logits`（TP 下按词表分片）与 reducer。"),
        nemo: c(1, "`LossFunction` input types LOGIT / LOGPROB / OPD_FULL / DISTILLATION.", "`LossFunction` 输入类型 LOGIT / LOGPROB / OPD_FULL / DISTILLATION。"),
        roll: c(2, "`loss_func` receives forward logits; use Strategy logprob/entropy ops for the TP layout.", "`loss_func` 接收前向 logits；TP 布局用 Strategy 的 logprob/entropy 算子。"),
        prime: c(4, "`CustomLoss` is selected-token only; full logits / hidden / cross-sample need extension.", "`CustomLoss` 只有选中 token；全 logits / hidden / 跨样本目标需扩展。"),
        own: c("own", "You hold the logits; watch TP vocab sharding and chunked/fused losses (Liger, TorchTitan).", "logits 在你手里；注意 TP 词表分片与分块/融合 loss（Liger、TorchTitan）。") } },
    { n: 10, g: "learn", term: "\\tfrac1Z\\textstyle\\sum_t m_t", t: bi("Normalization", "归一化"), d: bi("Token-mean vs sequence-mean, global denominators, masks.", "token 平均还是 sequence 平均、全局分母、mask。"),
      cells: {
        trl: c(2, "Inside `_compute_loss`.", "在 `_compute_loss` 里。"),
        openrlhf: c(3, "Reduction in `models/loss.py`; DeepSpeed scales accumulation again.", "`models/loss.py` 中的 reduction；DeepSpeed 还会按累积再缩放。"),
        verl: c(1, "Registry receives mask and weights.", "注册表接收 mask 与权重。"),
        slime: c(1, "`sum_of_sample_mean` reducer is passed in; Megatron scales outside, so don't re-average.", "传入 `sum_of_sample_mean` reducer；Megatron 外层还会缩放，不要再平均一次。"),
        nemo: c(1, "TOKEN_LEVEL / SEQUENCE_LEVEL plus `global_valid_toks` / `global_valid_seqs` in the contract.", "契约内含 TOKEN_LEVEL / SEQUENCE_LEVEL 与 `global_valid_toks` / `global_valid_seqs`。"),
        roll: c(2, "Sample weights in `loss_func`; micro-batching and backward in Strategy.", "`loss_func` 中的样本权重；微批与反向在 Strategy。"),
        own: c("own", "Define the global denominator; check every outer scaling.", "定义全局分母；检查每一层外部缩放。") } },
    { n: 11, g: "learn", term: "V_\\phi,\\,Q,\\,\\bar\\theta,\\,\\pi_T", t: bi("Extra networks", "额外网络"), d: bi("Critic, Q-head, target network, teacher, ensembles.", "critic、Q head、target 网络、教师、集成。"),
      cells: {
        trl: c(4, "`Accelerator.prepare` rejects several models in one call (FSDP2 / DeepSpeed branches): take over the lifecycle.", "`Accelerator.prepare` 拒绝一次准备多个模型（FSDP2 / DeepSpeed 分支）：要接管生命周期。"),
        openrlhf: c(4, "Actor/critic roles built in; new Q / target means DeepSpeed wiring, roles and loop.", "内置 actor/critic 角色；新增 Q / target 要改 DeepSpeed 接线、角色与循环。"),
        verl: c(4, "Critic built in; other networks change trainer/worker timing.", "内置 critic；其他网络会改 trainer/worker 时序。"),
        skyrl: c(4, "FSDP critic only (Megatron critic unimplemented); Q / target: worker, data protocol, learner.", "仅 FSDP 有 critic（Megatron critic 未实现）；Q / target 要改 worker、数据协议、learner。"),
        slime: c(4, "Megatron actor and schedule.", "Megatron actor 与调度。"),
        miles: c(4, "Backend workers and the multi-model update protocol.", "后端 worker 与多模型更新协议。"),
        areal: c(4, "Service roles and learner protocol.", "服务角色与 learner 协议。"),
        nemo: c(0, "Teachers are built in (multi-teacher OPD); a new Q-network still crosses backends.", "教师是内置角色（多教师 OPD）；新增 Q 网络仍要跨后端。"),
        roll: c(3, "Add a Cluster / Worker role in the pipeline.", "在 pipeline 中增加 Cluster / Worker 角色。"),
        prime: c(4, "Native model set and trainer protocol.", "native 模型范围与 trainer 协议。"),
        own: c("own", "Create each model and optimizer yourself; only generation-relevant params go to the engine.", "自己创建每个模型和优化器；只有影响生成的参数才同步给引擎。") } },
    { n: 12, g: "learn", term: "\\theta\\leftarrow\\mathrm{Opt}(\\theta,\\nabla\\mathcal J)", t: bi("Update rule", "更新规则"), d: bi("Update-to-data ratio, several optimizers, inner loops, target updates.", "更新数据比、多个优化器、内层优化、target 更新。"),
      cells: {
        trl: c(4, "Generation refresh, micro-step and optimizer step are different clocks; owned by the upstream Trainer.", "生成刷新、microstep、optimizer step 是不同的时钟，归上游 Trainer 管。"),
        openrlhf: c(3, "`BasePPOTrainer.train_step` + DeepSpeed accumulation/step rules.", "`BasePPOTrainer.train_step` + DeepSpeed 累积/step 约定。"),
        verl: c(4, "TransferQueue lifecycle and trainer/worker timing.", "TransferQueue 生命周期与 trainer/worker 时序。"),
        skyrl: c(3, "Override a trainer stage (as the OPD example does); update ratios live in the main loop.", "覆写 trainer 阶段（OPD 示例即如此）；更新比率在主循环里。"),
        slime: c(4, "Megatron schedule: `forward_step` returns a loss closure; extra backward passes touch it.", "Megatron 调度：`forward_step` 返回 loss closure；多次反传会触及它。"),
        miles: c(4, "Backend worker and update protocol.", "后端 worker 与更新协议。"),
        areal: c(2, "`PPOActor._ppo_update`.", "`PPOActor._ppo_update`。"),
        roll: c(3, "`BaseActorWorker.train_step` (epochs, batches) + Strategy (micro-batch, backward, optimizer).", "`BaseActorWorker.train_step`（epoch、batch）+ Strategy（微批、反向、优化器）。"),
        own: c("own", "Your loop; respect backend rules (no closure on ZeRO-3, `full_tensor` is a collective).", "自己的循环；遵守后端约定（ZeRO-3 无 closure，`full_tensor` 是集合通信）。") } },
    // ---------------- system
    { n: 13, g: "sys", term: "\\mu\\leftarrow\\mathrm{publish}(\\theta)", t: bi("Publish weights", "发布权重"), d: bi("Weight sync, versions, LoRA merge, KV invalidation.", "权重同步、版本、LoRA merge、KV 失效。"),
      cells: {
        trl: c(3, "`trl/generation/vllm_generation.py`: `full_tensor`, ZeRO gather, PEFT merge/unmerge.", "`trl/generation/vllm_generation.py`：`full_tensor`、ZeRO gather、PEFT merge/unmerge。"),
        openrlhf: c(3, "vLLM broadcast; ZeRO-3 via `GatheredParameters`; no LoRA on Ray + vLLM.", "vLLM 广播；ZeRO-3 用 `GatheredParameters`；Ray + vLLM 不支持 LoRA。"),
        verl: c(0, "HybridFlow resharding between training and generation layouts.", "HybridFlow 处理训练与生成布局之间的重分片。"),
        skyrl: c(0, "Incremental weight sync; RDT + NIXL.", "增量权重同步；RDT + NIXL。"),
        miles: c(0, "`WeightUpdater`; P2P/RDMA (Kimi K2: 53.28 s → 7.23 s for the update phase).", "`WeightUpdater`；P2P/RDMA（Kimi K2：更新阶段 53.28 秒 → 7.23 秒）。"),
        areal: c(0, "Dedicated weight-update service.", "独立的 weight-update 服务。"),
        roll: c(0, "Sampling pauses only before the next weight publication.", "下一轮权重发布前才暂停采样。"),
        own: c("own", "vLLM weight-transfer engines or SGLang update sessions; you own names, layouts, collective order, KV.", "vLLM 权重传输引擎或 SGLang 更新 session；参数名、布局、collective 顺序、KV 都归你。") } }
  ];

  var GROUPS_POS = {
    data: { c: "var(--cyan)", n: bi("Data rule: which samples enter, with what weight", "数据规则：哪些样本进入、带多大权重") },
    learn: { c: "var(--amber)", n: bi("Learning rule: what you do with them", "学习规则：拿这些样本做什么") },
    sys: { c: "var(--teal)", n: bi("System: closing the loop", "系统：闭合循环") }
  };

  // The objective with clickable positions (KaTeX \htmlClass, trusted).
  function H(n, s) { return "\\htmlClass{ft f" + n + "}{" + s + "}"; }
  var FORMULA = [
    "\\mathcal J(\\theta)=\\mathbb E_{" + H(1, "x\\sim\\mathcal D") + ",\\ " + H(2, "y\\sim\\mu") + "}\\Big[" + H(10, "\\tfrac1Z\\textstyle\\sum_t m_t") + "\\," + H(6, "w_t") + "\\," +
      H(8, "\\min\\!\\big(\\rho_t" + H(7, "\\hat A_t") + ",\\,\\mathrm{clip}(\\rho_t,1\\!\\pm\\!\\epsilon)\\hat A_t\\big)-\\beta\\,\\mathrm{KL}_t") + "\\Big]",
    "\\rho_t=\\frac{" + H(9, "\\pi_\\theta(y_t\\mid h_t)") + "}{\\pi_{\\text{old}}(y_t\\mid h_t)},\\quad " + H(6, "w_t=\\frac{\\pi_{\\text{old}}}{\\mu}") + ",\\quad " +
      H(7, "\\hat A_t=f\\big(" + H(3, "r(x,y)") + ",\\," + H(11, "V_\\phi") + ",\\dots\\big)"),
    "\\mu=\\text{engine}(\\theta_v)\\ \\text{through}\\ " + H(4, "\\text{buffer}") + ",\\ " + H(5, "v\\ge v_{\\text{now}}-k") + "\\qquad " +
      H(12, "\\theta\\leftarrow\\mathrm{Opt}(\\theta,\\nabla_\\theta\\mathcal J)") + "\\qquad " + H(13, "\\mu\\leftarrow\\mathrm{publish}(\\theta)") + "\\qquad " + H(11, "\\{V_\\phi,Q,\\bar\\theta,\\pi_{\\text{ref}},\\pi_T\\}")
  ];

  // ------------------------------------------------------------ three traced changes
  var TRACES = [
    { c: "var(--cyan)", tag: bi("Data rule", "数据规则"), pos: [1, 4, 5, 6],
      t: bi("Prioritize hard prompts and reuse each group up to k times", "优先采样难题，每组数据最多重用 k 次"),
      best: bi("Miles `custom_async_data_buffer_path` (L1) · verl V1 custom ReplayBuffer / sampler (L1)", "Miles `custom_async_data_buffer_path`（L1）· verl V1 自定义 ReplayBuffer / sampler（L1）"),
      steps: [bi("① + ④: write the selection and reuse rule as a buffer: `put` records difficulty, version and reuse count; `get` samples by priority.", "① + ④：把选择与重用规则写成 buffer：`put` 记录难度、版本和重用次数；`get` 按优先级采样。"),
              bi("⑤: decide the age gate at consumption, not at enqueue (Miles evaluates staleness when consumed).", "⑤：在消费时而非入队时判断年龄（Miles 就是在消费时计算陈旧度）。"),
              bi("⑥: a reused sample is off-policy; it needs its stored behavior logprobs and version for w = π_old / μ.", "⑥：被重用的样本是离策略的；需要它保存的 behavior logprob 与版本来计算 w = π_old / μ。")],
      warn: bi("Default buffers in Miles and ROLL remove samples on consumption; verl's 'ReplayBuffer' is not classic replay by default. Elsewhere: TRL overrides `_prepare_inputs` (L2); SkyRL extends worker, data protocol and learner (L4).", "Miles 与 ROLL 的默认 buffer 消费即移除；verl 的 “ReplayBuffer” 默认也不是经典回放。其他路线：TRL 覆写 `_prepare_inputs`（L2）；SkyRL 需扩展 worker、数据协议与 learner（L4）。") },
    { c: "var(--amber)", tag: bi("Learning rule · full distribution", "学习规则 · 全分布"), pos: [9, 10],
      t: bi("An exact entropy bonus or full-vocabulary KL", "精确熵奖励或全词表 KL"),
      best: bi("slime `custom_loss` (logits + reducer) · NeMo RL `LossFunction` (LOGIT input) · ROLL `loss_func` (logits)", "slime `custom_loss`（logits + reducer）· NeMo RL `LossFunction`（LOGIT 输入）· ROLL `loss_func`（logits）"),
      steps: [bi("⑨: the hook must receive logits. A selected-token logprob gives only a sampled estimate, and a naive −log π bonus has zero expected gradient.", "⑨：hook 必须拿到 logits。只有选中 token 的 logprob 只能做采样估计，朴素的 −log π 奖励期望梯度为零。"),
              bi("Under TP the logits are vocab-sharded: use vocab-parallel log-softmax / entropy ops, not a local softmax.", "TP 下 logits 按词表分片：用词表并行的 log-softmax / 熵算子，而不是本地 softmax。"),
              bi("⑩: reduce with the framework's reducer or global counts; do not average again outside.", "⑩：用框架提供的 reducer 或全局计数归约；不要在外面再平均一次。")],
      warn: bi("verl, SkyRL and prime-rl registries pass selected-token logprobs: this becomes L4 (worker / model output). TRL's custom SFT loss leaves the fused LM-head path. Liger's fused PPO treats advantages as constants; TorchTitan's ChunkedLossWrapper ignores outer grad_output. Keep a pure-PyTorch reference.", "verl、SkyRL、prime-rl 的注册表只传选中 token logprob：这会变成 L4（worker / 模型输出）。TRL 的 SFT 自定义 loss 会离开 fused LM-head 路径。Liger 的 fused PPO 把 advantage 当常量；TorchTitan 的 ChunkedLossWrapper 忽略外层 grad_output。保留一个纯 PyTorch 参考实现。") },
    { c: "var(--pink)", tag: bi("Learning rule · new state", "学习规则 · 新状态"), pos: [11, 12, 13],
      t: bi("A Q-critic with a target network, its own optimizer, 4 updates per batch", "带 target 网络的 Q critic，独立优化器，每批更新 4 次"),
      best: bi("Own learner on FSDP2 or DeepSpeed + vLLM / SGLang weight transfer (every framework column is L4 here)", "自有 learner + FSDP2 或 DeepSpeed + vLLM / SGLang 权重传输（这一行在所有框架里都是 L4）"),
      steps: [bi("⑪: create the Q-network, target and optimizers yourself. `Accelerator.prepare` won't take several models in one call; wire FSDP2 / DeepSpeed directly.", "⑪：自己创建 Q 网络、target 和各自优化器。`Accelerator.prepare` 不接受一次多个模型；直接接 FSDP2 / DeepSpeed。"),
              bi("⑫: the loop owns backward order, step/skip and target updates. On ZeRO-3, no closure in `step`; on FSDP2, a local shard is not `full_tensor()`.", "⑫：主循环掌握反向顺序、step/skip 与 target 更新。ZeRO-3 的 `step` 不支持 closure；FSDP2 的本地分片不是 `full_tensor()`。"),
              bi("⑬: publish only the policy. Q and target stay in the learner unless Q selects actions at generation time.", "⑬：只发布策略参数。Q 与 target 留在 learner，除非 Q 参与生成时的动作选择。")],
      warn: bi("Checkpoint every optimizer, the target, the buffer position and the version: an optimizer step, a published version and a recoverable checkpoint are three different clocks.", "checkpoint 要包含每个优化器、target、buffer 位置与版本：一次 optimizer step、一个发布版本、一个可恢复 checkpoint 是三个不同的时钟。") }
  ];

  window.ANATOMY = { FW: FW, LEVELS: LEVELS, ATOMS: ATOMS, GROUPS: GROUPS, RECIPES: RECIPES, AXES: AXES, MAP: MAP, POS: POS, GROUPS_POS: GROUPS_POS, FORMULA: FORMULA, TRACES: TRACES };
})();
