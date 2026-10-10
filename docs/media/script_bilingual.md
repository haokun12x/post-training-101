# Post-Training Systems, Layer by Layer · 后训练工具体系：分层拆解

Bilingual narration script for `post_training_systems_en.mp4` and `post_training_systems_zh.mp4`.
中英对照解说稿，对应英文版与中文版视频。时间戳格式 EN / 中文。

Duration 时长: EN 31:37 · 中文 32:15

## Contents 目录

00. Title · 标题  `00:00 / 00:00`
01. Six parts · 六个部分  `00:41 / 00:44`
02. One gradient behind every method · 一个梯度统一所有后训练方法  `01:22 / 01:25`
03. Six layers, one component per role · 六层工具栈：每个角色选一个部件  `02:41 / 02:42`
04. Parameterization: LoRA & QLoRA arithmetic · 参数化层：LoRA 与 QLoRA 的算术  `04:21 / 04:12`
05. Execution: the sharding math · 训练执行层：分片的数学  `05:57 / 05:49`
06. Generation engines & the weight-sync boundary · 推理生成层与权重同步边界  `08:06 / 07:51`
07. Offline post-training: SFT & DPO · 离线后训练：SFT 与 DPO  `09:35 / 09:23`
08. Online RL: the loop and its objectives · 在线 RL：闭环与目标函数  `11:16 / 11:11`
09. Step time: find the bottleneck first · 单步时间：先找到瓶颈  `12:37 / 12:32`
10. Asynchrony is not a boolean · 异步不是一个布尔开关  `14:04 / 14:03`
11. Train–inference consistency: five error sources · 训练—推理一致性：五个误差来源  `15:30 / 15:35`
12. What "customize every detail" really means · “定制每个算法细节”到底意味着什么  `16:33 / 16:38`
13. Where each framework lets you in · 各框架的算法入口  `17:33 / 17:42`
14. The framework landscape at a glance · 主流框架定位速览  `18:40 / 18:52`
15. Same atoms, different recipes · 同一批原子，不同的配方  `20:02 / 20:08`
16. Four trade-off axes · 四条取舍轴  `21:34 / 21:43`
17. Every term of the objective has an address · 目标函数里每一项都有地址  `22:54 / 23:05`
18. The address book: position × framework · 地址簿：位置 × 框架  `23:47 / 24:00`
19. Three changes, traced · 三个改动，逐项追踪  `24:59 / 25:19`
20. Own the trajectory contract; reuse the runtime · 自己掌握轨迹契约，复用运行时  `26:29 / 26:48`
21. A decision procedure · 选型决策流程  `27:52 / 28:17`
22. Validate: two stages and a code checklist · 验证：两阶段评测与源码检查清单  `29:24 / 29:56`
23. Five takeaways · 五条结论  `30:46 / 31:24`

## 00 · Post-Training Systems, Layer by Layer · 后训练工具体系：分层拆解

⏱ EN 00:00 · 中文 00:00

| English | 中文 |
|---|---|
| Welcome. In this video we'll build a systematic mental model of the post-training tool ecosystem, the way a senior post-training engineer sees it. Not a list of frameworks, but a layered map: what each layer computes, what math it must preserve, and how the pieces compose into a working system. | 欢迎。这期视频从一位资深后训练工程师的视角，为后训练工具生态建立一套系统化的理解。我们不罗列框架，而是画一张分层地图：每一层在计算什么，必须守住哪些数学语义，以及这些部件怎样组合成一个能真正运行的系统。 |
| Everything we discuss comes back to three forces that pull against each other: how freely you can change the algorithm, how much engineering complexity you take on, and how efficiently the whole loop executes. The material comes from the two research reports in this folder, one of which reviews the source code of the major frameworks at their October 2026 releases. | 贯穿全片的是三股彼此拉扯的力量：算法能改得多自由，要承担多少工程复杂度，以及整个循环跑得多高效。素材来自这个目录下的两份调研报告，其中一份对主流框架做了源码级审查，版本固定在 2026 年 10 月的发布。 |

## 01 · Six parts · 六个部分
*ROADMAP · 路线图*

⏱ EN 00:41 · 中文 00:44

| English | 中文 |
|---|---|
| Here's the roadmap. Part one derives a single equation that covers nearly every post-training method. Part two walks the stack layer by layer, with the memory math. Part three contrasts offline and online training. Part four covers where systems really differ: step time, asynchrony, consistency, and what algorithmic freedom actually means. Part five takes the frameworks apart: the shared atoms they are built from, the trade-offs between recipes, and the exact address of every term of the objective in each codebase. Part six turns it all into a data contract, an architecture, a decision procedure, and a validation protocol. | 先看路线图。第一部分推导一个几乎涵盖所有后训练方法的统一公式。第二部分逐层走过工具栈，并算清显存账。第三部分对比离线训练与在线训练。第四部分讨论系统真正拉开差距的地方：单步时间、异步、训练推理一致性，以及“算法自由度”究竟指什么。第五部分把框架拆开：它们由哪些共享原子组成，配方之间如何取舍，以及目标函数的每一项在各个代码库里的确切地址。第六部分把这些落成数据契约、最小架构、选型流程和验证方法。 |

## 02 · One gradient behind every method · 一个梯度统一所有后训练方法
*PART 1 · ONE EQUATION · 第一部分 · 一个公式*

⏱ EN 01:22 · 中文 01:25

| English | 中文 |
|---|---|
| Start with the math. Almost every post-training method can be written as one gradient. Sample a prompt x and a response y from some data distribution mu, then push up the log-probability of each token, weighted by a coefficient psi. Methods differ in only two places: where mu comes from, and how psi is computed. | 先从数学开始。几乎所有后训练方法都能写成同一个梯度：从某个数据分布 mu 中取出提示 x 和回答 y，然后提高每个 token 的对数概率，并用系数 psi 加权。不同方法只在两处不同：mu 从哪里来，psi 怎么算。 |
| Supervised fine-tuning uses a fixed dataset, and psi is one on the masked response tokens. DPO also uses fixed data, preference pairs, and psi is a sigmoid-weighted push up on the chosen answer and down on the rejected one. PPO and GRPO sample from the policy itself and weight tokens by a clipped importance ratio times an advantage. On-policy distillation samples from the student and uses the teacher-minus-student log-probability as the signal. | 监督微调用固定数据集，psi 在被 mask 选中的回答 token 上等于 1。DPO 同样用固定数据，也就是偏好对，psi 是一个 sigmoid 加权：拉高被选中的回答，压低被拒绝的回答。PPO 和 GRPO 从策略本身采样，用截断后的重要性比率乘以优势来加权。在线蒸馏从学生模型采样，用教师与学生的对数概率之差作为信号。 |
| This one observation organizes the whole tool landscape. If mu is a fixed dataset, post-training is a trainer problem: load data, compute the loss, step the optimizer. If mu is the policy itself, it becomes a systems problem: you need a generation engine, reward computation, weight synchronization, and bookkeeping for which policy produced which token. That is why RL frameworks look so different from fine-tuning libraries. | 这一个观察就能组织起整个工具版图。如果 mu 是固定数据集，后训练就是一个训练器问题：读数据、算损失、走优化器。如果 mu 就是策略本身，它就变成一个系统问题：需要生成引擎、奖励计算、权重同步，还要记录每个 token 是由哪个版本的策略产生的。这正是强化学习框架和微调库看起来差别如此之大的原因。 |

## 03 · Six layers, one component per role · 六层工具栈：每个角色选一个部件
*PART 2 · THE STACK, LAYER BY LAYER · 第二部分 · 逐层拆解工具栈*

⏱ EN 02:41 · 中文 02:42

| English | 中文 |
|---|---|
| Now the stack. The source-code review organizes the ecosystem into six layers. On top is the algorithm layer: rewards, advantages, losses, critics, teachers, and update rules. That's where your research lives. Below it is the data and runtime layer: rollouts, environments, multi-turn trajectories, buffers, resource placement, and policy versions. This is what verl, OpenRLHF, SkyRL, AReaL, Miles, ROLL and NeMo RL actually compete on. | 接下来看工具栈。源码审查报告把生态分成六层。最上面是算法层：奖励、优势、损失、critic、教师和更新规则，这是你的研究所在。下面是数据与运行层：rollout、环境、多轮轨迹、buffer、资源布局和策略版本。verl、OpenRLHF、SkyRL、AReaL、Miles、ROLL 和 NeMo RL 真正竞争的，正是这一层。 |
| Next is model execution: forward and backward passes, state sharding, and tensor, pipeline, expert and context parallelism, provided by FSDP2, DeepSpeed, Megatron-Core or TorchTitan. Beside it, but separate, is the generation layer. vLLM and SGLang handle batching, the KV cache, sampling, and hot weight updates. They never run a backward pass. | 再往下是模型与训练执行层：前向、反向、状态分片，以及张量、流水线、专家和上下文并行，由 FSDP2、DeepSpeed、Megatron-Core 或 TorchTitan 提供。和它并列但相互独立的是推理生成层：vLLM 和 SGLang 负责 batching、KV cache、采样和权重热更新，它们从不做反向传播。 |
| At the bottom are parameterization and kernels, such as LoRA, quantization, fused losses and attention kernels, and the data quality and observability layer: data processing, verifiers, evaluation, logging and profiling. | 最底层是参数化与 kernel 层，比如 LoRA、量化、融合损失和 attention kernel；以及数据质量与观测层：数据加工、验证器、评估、日志和性能分析。 |
| Two more layers sit beside this stack: agent and environment adapters, and hosted training services such as Tinker. Easy-to-use toolchains like LLaMA-Factory, ms-swift, Axolotl and Unsloth cut across several layers at once. The first rule of composition follows directly. A training backend cannot replace a rollout loop, and an inference engine cannot replace a trainer. Pick one component per role, then verify the whole path. | 在这个栈旁边还有两层：Agent 与环境接入层，以及像 Tinker 这样的托管训练服务。LLaMA-Factory、ms-swift、Axolotl、Unsloth 这类易用工具链则同时横跨好几层。由此直接得到第一条组合规则：训练后端替代不了 rollout 闭环，推理引擎也替代不了训练器。每个角色选一个部件，然后验证整条路径是否贯通。 |

## 04 · Parameterization: LoRA & QLoRA arithmetic · 参数化层：LoRA 与 QLoRA 的算术
*PART 2 · THE STACK, LAYER BY LAYER · 第二部分 · 逐层拆解工具栈*

⏱ EN 04:21 · 中文 04:12

| English | 中文 |
|---|---|
| Let's walk the layers with math, starting near the bottom. LoRA freezes the pretrained weight W-zero and learns a low-rank update: B times A, scaled by alpha over r. B starts at zero, so training begins exactly at the base model. Trainable parameters drop from d times k to r times d-plus-k. For a 4096 by 4096 projection at rank 8, that's about sixty-five thousand parameters instead of sixteen point eight million: roughly 0.4 percent. | 我们带着数学逐层往上走，先从底部的参数化开始。LoRA 冻结预训练权重 W0，只学习一个低秩更新：B 乘 A，再乘以 alpha 除以 r。B 初始化为零，所以训练恰好从基座模型出发。可训练参数从 d 乘 k 降到 r 乘以 d 加 k。对一个 4096 乘 4096 的投影矩阵、秩为 8 时，只有约六万五千个参数，而不是一千六百八十万，大约千分之四。 |
| Why this matters becomes clear from memory accounting. Full fine-tuning with mixed-precision Adam stores about sixteen bytes per parameter: two for bf16 weights, two for gradients, and twelve for the fp32 master copy and the two Adam moments. A 7-billion-parameter model needs about 112 gigabytes before activations. With LoRA, the frozen base costs about two bytes per parameter. QLoRA stores the base in 4-bit NormalFloat, about half a byte per parameter, which is how the QLoRA paper fine-tuned a 65-billion model on a single 48-gigabyte GPU. | 显存账能说明它为什么重要。混合精度 Adam 的全参数微调，每个参数大约占 16 字节：bf16 权重 2 字节，梯度 2 字节，fp32 主权重和两个 Adam 动量共 12 字节。一个 70 亿参数的模型，在不算激活的情况下就需要约 112 GB。用 LoRA 时，冻结的基座每个参数约 2 字节。QLoRA 把基座存成 4 位 NormalFloat，每个参数约半个字节，QLoRA 论文正是这样在单张 48 GB 显卡上微调了 650 亿参数的模型。 |
| In tools: PEFT injects adapters with get_peft_model, bitsandbytes supplies the 4-bit storage and 8-bit optimizers, and Unsloth fuses kernels for memory and speed. One engineering caveat: training only LoRA does not mean syncing only LoRA. In online RL, some integrations merge the adapter into the base, or gather the full weights, before pushing them to the inference engine. | 对应到工具：PEFT 用 get_peft_model 注入 adapter，bitsandbytes 提供 4 位存储和 8 位优化器，Unsloth 用融合 kernel 节省显存、提升速度。有一个工程上的提醒：只训练 LoRA 不等于只同步 LoRA。在在线强化学习里，有些集成会先把 adapter 合并进基座，或者收集完整权重，再推送给推理引擎。 |

## 05 · Execution: the sharding math · 训练执行层：分片的数学
*PART 2 · THE STACK, LAYER BY LAYER · 第二部分 · 逐层拆解工具栈*

⏱ EN 05:57 · 中文 05:49

| English | 中文 |
|---|---|
| One layer up is execution, and ZeRO's analysis is the cleanest way to reason about it. Let psi be the parameter count, K the optimizer bytes per parameter, twelve for mixed-precision Adam, and N the number of GPUs. Plain data parallelism replicates everything: sixteen psi per GPU. ZeRO stage one shards the optimizer states, stage two also shards gradients, and stage three, which is also what FSDP2 does, shards the parameters too. For a 7.5-billion model on 64 GPUs, that's 120 gigabytes per GPU, then 31, then 17, and finally under 2. | 再往上一层是训练执行，ZeRO 的分析是最清晰的推理方式。设 psi 为参数量，K 为每个参数的优化器字节数，混合精度 Adam 下 K 等于 12，N 为 GPU 数。普通数据并行把一切都复制一份：每张卡 16 psi。ZeRO 第一阶段分片优化器状态，第二阶段再分片梯度，第三阶段连参数也分片，FSDP2 做的也是这一件事。对 75 亿参数、64 张卡，每卡显存从 120 GB 依次降到 31、17，最后不到 2 GB。 |
| Three backends implement this layer. FSDP2 shards each parameter as a DTensor and keeps ordinary PyTorch control flow. DeepSpeed adds offload to CPU and NVMe, plus AutoTP, AutoEP and its hybrid engine. Megatron-Core brings tensor, pipeline, expert and context parallelism for very large dense and MoE models. Each has a contract you must respect. In FSDP2, the local shard is not the full tensor. In DeepSpeed, the engine owns gradient-accumulation boundaries. In Megatron, the pipeline schedule owns the backward pass. | 三个后端实现了这一层。FSDP2 把每个参数切成 DTensor，同时保留普通的 PyTorch 控制流。DeepSpeed 增加了 CPU 与 NVMe offload，还有 AutoTP、AutoEP 和 Hybrid Engine。Megatron-Core 为超大的稠密模型和 MoE 提供张量、流水线、专家与上下文并行。每个后端都有必须遵守的契约：在 FSDP2 里，本地分片不是完整张量；在 DeepSpeed 里，梯度累积的边界归引擎管理；在 Megatron 里，反向传播由流水线调度器掌控。 |
| Composition rules. FSDP2 and ZeRO-3 are alternative ways to manage the same model's state, so don't wrap one inside the other. Ray orchestrates processes and resources, while NCCL moves tensors; they live at different layers. And a classic bug: dividing the loss by the accumulation steps yourself while the engine scales it again, which shrinks your gradients twice. | 几条组合规则。FSDP2 和 ZeRO-3 是管理同一个模型状态的两种替代方案，不要互相嵌套。Ray 负责进程与资源编排，NCCL 负责张量通信，它们属于不同层次。还有一个经典 bug：你自己已经把 loss 除以累积步数，引擎又缩放一次，梯度就被缩小了两次。 |
| And a correction worth stating plainly. DeepSpeed, DeepSpeed-Chat and OpenRLHF are three different things. DeepSpeed is execution infrastructure. DeepSpeed-Chat is one classic three-stage RLHF application built on it. OpenRLHF is a Ray pipeline that still trains with DeepSpeed. With identical math, sharding changes the cost of training, not the objective being learned. | 还有一处值得明确的纠正：DeepSpeed、DeepSpeed-Chat 和 OpenRLHF 是三样不同的东西。DeepSpeed 是执行基础设施；DeepSpeed-Chat 是构建在它之上的一个经典三阶段 RLHF 应用；OpenRLHF 是一条 Ray 流水线，训练仍然由 DeepSpeed 执行。在数学完全一致的前提下，分片改变的是训练成本，而不是学习目标。 |

## 06 · Generation engines & the weight-sync boundary · 推理生成层与权重同步边界
*PART 2 · THE STACK, LAYER BY LAYER · 第二部分 · 逐层拆解工具栈*

⏱ EN 08:06 · 中文 07:51

| English | 中文 |
|---|---|
| The generation layer has different physics. Prefill processes the prompt in parallel and is compute-bound. Decode emits one token per step, and every step must re-read the weights and the KV cache from memory, so it's bandwidth-bound. That's why engines like vLLM and SGLang exist: paged KV cache, continuous batching and prefix sharing let many sequences share each weight read. With long reasoning traces, this autoregressive phase is frequently the largest part of an RL step. | 推理生成层遵循另一套物理规律。Prefill 并行处理整段提示，是计算受限的。Decode 每一步只产生一个 token，而每一步都要把权重和 KV cache 从显存重新读一遍，所以是带宽受限的。这就是 vLLM、SGLang 这类引擎存在的理由：分页 KV cache、连续批处理和前缀共享，让许多序列分摊每一次权重读取。在长推理轨迹下，这个自回归阶段常常是 RL 单步里最大的开销。 |
| The trainer and the engine also lay out the same weights differently: FSDP shards or Megatron partitions on one side, inference tensor parallelism on the other. So every policy update means resharding, possibly converting precision, transferring and reloading. vLLM now separates the weight source, the trainer-side transfer engine and the inference-side receiver. SGLang supports updates from disk, tensors or a distributed group, with pause and resume. Weight sync is not plumbing; it's a protocol boundary. | 训练端和推理引擎对同一份权重的布局也不同：一边是 FSDP 分片或 Megatron 切分，另一边是推理侧的张量并行。因此每次策略更新都意味着重新分片、可能的精度转换、传输和重新加载。vLLM 现在把权重来源、训练侧传输引擎和推理侧接收端分开；SGLang 支持从磁盘、张量或分布式通信组更新，并支持暂停与恢复。权重同步不是管道杂活，而是一条协议边界。 |
| A practical rule: anything the policy needs in order to generate must be representable in the engine and synced correctly. Learner-only parts like critics, Q-heads or target networks can stay on the training side. DeepSpeed's hybrid engine was an early answer to this tension: the same actor, switching between a training layout and a generation layout. | 一条实用规则：凡是策略在生成时需要用到的参数，都必须能在推理引擎中表达，并被正确同步；只用于学习的部件，比如 critic、Q head 或 target 网络，可以留在训练侧。DeepSpeed 的 Hybrid Engine 是对这一矛盾的早期回答：同一个 actor，在训练布局和生成布局之间切换。 |

## 07 · Offline post-training: SFT & DPO · 离线后训练：SFT 与 DPO
*PART 3 · OFFLINE VS. ONLINE · 第三部分 · 离线与在线*

⏱ EN 09:35 · 中文 09:23

| English | 中文 |
|---|---|
| Part three: offline versus online. Offline methods train on a fixed dataset. SFT minimizes masked negative log-likelihood. DPO turns a preference pair into a logistic loss on the difference of policy-to-reference log-ratios, scaled by beta. Neither needs a generation engine, and the frozen reference model's log-probabilities can even be precomputed. | 第三部分：离线与在线。离线方法在固定数据集上训练。SFT 最小化带 mask 的负对数似然。DPO 把一个偏好对变成逻辑回归损失，作用在策略相对参考模型的对数比之差上，再乘以 beta。两者都不需要生成引擎，冻结参考模型的对数概率甚至可以提前算好。 |
| The subtle part is the normalizer Z. It looks like a detail, but it's part of the objective. Token-mean gives every token the same weight. Sequence-mean averages within each response first, so a token in a 1000-token answer weighs ten times less than a token in a 100-token answer. And with gradient accumulation or multiple ranks, an average of local means is not the global mean. These aren't system knobs; they change what you optimize. | 微妙之处在于归一化因子 Z。它看起来像细节，实际上是目标函数的一部分。token 平均让每个 token 权重相同；序列平均先在每个回答内部求平均，于是 1000 个 token 的回答中的一个 token，权重只有 100 个 token 回答中的十分之一。而在梯度累积或多卡情况下，局部均值的平均并不等于全局均值。这些不是系统旋钮，它们改变的是你在优化什么。 |
| The standard offline composition is: Transformers for the model, PEFT for adapters, bitsandbytes when memory is tight, Accelerate, DeepSpeed or FSDP2 for scale, and TRL's SFT or DPO trainer for the objective and lifecycle. LLaMA-Factory, ms-swift and Axolotl package the same pieces as config-driven recipes, and Unsloth focuses on memory and speed. | 标准的离线组合是：Transformers 提供模型，PEFT 提供 adapter，显存紧张时加上 bitsandbytes，用 Accelerate、DeepSpeed 或 FSDP2 扩展规模，再用 TRL 的 SFT 或 DPO 训练器承载目标函数和训练生命周期。LLaMA-Factory、ms-swift 和 Axolotl 把同样的部件封装成配置驱动的配方，Unsloth 则专注于显存和速度。 |
| One efficiency boundary from the code review: in the TRL snapshot reviewed, a custom compute-loss function runs a full-logits forward pass, plus an extra no-gradient fused pass for metrics, while the default path can use a fused LM head. So when you change the loss, re-check memory and throughput instead of inheriting the defaults. | 源码审查发现了一个效率边界：在审查的 TRL 快照中，自定义 compute_loss_func 会走完整 logits 的前向，还要额外做一次无梯度的融合前向来计算指标；而默认路径可以直接使用融合的 LM head。所以换了损失函数后，要重新核对显存和吞吐，不能直接沿用默认路径的预期。 |

## 08 · Online RL: the loop and its objectives · 在线 RL：闭环与目标函数
*PART 3 · OFFLINE VS. ONLINE · 第三部分 · 离线与在线*

⏱ EN 11:16 · 中文 11:11

| English | 中文 |
|---|---|
| Online RL closes a loop. Sample prompts, generate rollouts with the inference engine, score them with a reward model, verifier or environment, estimate advantages, update the learner, publish new weights to the engine, and repeat. Every arrow in this loop is a place where a framework makes design decisions. | 在线强化学习是一个闭环：取提示，用推理引擎生成 rollout，用奖励模型、验证器或环境打分，估计优势，更新学习器，把新权重发布给推理引擎，然后循环。这个闭环里的每一条箭头，都是框架做设计决策的地方。 |
| The objectives are compact. PPO maximizes a clipped surrogate: the probability ratio between the current and old policy, times the advantage, clipped to one plus or minus epsilon. GRPO removes the critic. It samples a group of G responses per prompt and standardizes each reward against the group mean and standard deviation, with an optional KL penalty toward a reference model. | 目标函数本身很紧凑。PPO 最大化一个截断代理目标：当前策略与旧策略的概率比乘以优势，并把比率截断在一加减 epsilon 之内。GRPO 去掉了 critic：对每个提示采样一组 G 个回答，用组内均值和标准差标准化每个奖励，并可选地加上对参考模型的 KL 惩罚。 |
| Here's what separates careful implementations. There are four probability tracks. Mu is the behavior distribution the engine actually sampled from. Pi-old is the trainer's recomputation at the start of the step. Pi-theta is the current, differentiable policy. And pi-ref is the KL anchor, or a teacher in distillation. The true importance weight factorizes into the PPO ratio times a train-inference mismatch term. OpenRLHF, AReaL and others keep these tracks separate, and truncated importance sampling caps the mismatch term. | 真正区分细致实现的，是下面这一点：一共有四条概率轨道。mu 是推理引擎实际采样时的行为分布；pi old 是训练器在这一步开始时的重算；pi theta 是当前可微的策略；pi ref 是 KL 的锚点，在蒸馏里则是教师。真实的重要性权重可以分解为 PPO 比率乘以一个训练推理失配项。OpenRLHF、AReaL 等框架会把这几条轨道分开保存，截断重要性采样则给失配项设一个上限。 |

## 09 · Step time: find the bottleneck first · 单步时间：先找到瓶颈
*PART 4 · WHERE SYSTEMS REALLY DIFFER · 第四部分 · 系统真正的差异*

⏱ EN 12:37 · 中文 12:32

| English | 中文 |
|---|---|
| Part four is where systems really differ. Write down the time of one synchronous step: generation, reward, training and weight sync, all in sequence. With pipelining on independent resources, the ideal step time approaches the slower of generation and training, plus sync. Amdahl's law sets the priority. If generation is 80 percent of the step, making training twice as fast saves only ten percent. | 第四部分讲系统真正拉开差距的地方。先写出一次同步单步的时间：生成、奖励、训练和权重同步依次进行。在资源相互独立的理想流水化下，单步时间趋近于生成与训练中较慢的那一个，再加上同步。Amdahl 定律决定了优先级：如果生成占单步的 80%，把训练加速一倍也只能省下 10%。 |
| Published numbers confirm that async is not automatically faster. In NeMo RL's performance table, DeepSeek-V3 on 512 H100s dropped from 111 to about 64 seconds per step with one-step-off async. But Qwen3 30B-A3B on 16 GB200s went from 221 to 301 seconds: splitting resources made it slower. And Miles cut a trillion-parameter Kimi K2 weight update from 53 seconds with NCCL broadcast to about 7 seconds with peer-to-peer transfer, which shows weight sync can be a first-class bottleneck. | 公开数据也说明异步并不自动更快。在 NeMo RL 的性能表里，DeepSeek-V3 在 512 张 H100 上，one-step-off 异步把单步从 111 秒降到约 64 秒；但 Qwen3-30B-A3B 在 16 张 GB200 上，却从 221 秒变成 301 秒，资源拆分反而让它更慢。Miles 则把万亿参数 Kimi K2 的一次权重更新，从 NCCL 广播的 53 秒降到点对点传输的约 7 秒，说明权重同步本身就可能是头等瓶颈。 |
| So the metric that matters is cost to target: total GPU-hours, across all roles, to reach the same validation score. Tokens per second can reward long, useless generations; steps per second can hide smaller batches; wall time can hide extra GPUs. Profile each phase, and fix the biggest one first. | 所以真正重要的指标是达到目标的成本：所有角色加起来，达到同一验证分数所消耗的 GPU 小时。只看每秒 token 数，可能奖励了又长又无效的生成；只看每秒步数，可能掩盖了更小的 batch；只看墙钟时间，可能掩盖了用了更多 GPU。把每个阶段分别 profile，先解决最大的那个。 |

## 10 · Asynchrony is not a boolean · 异步不是一个布尔开关
*PART 4 · WHERE SYSTEMS REALLY DIFFER · 第四部分 · 系统真正的差异*

⏱ EN 14:04 · 中文 14:03

| English | 中文 |
|---|---|
| Asynchrony introduces staleness. Define a sample's age as the learner's current version minus the version that generated it. With mid-flight weight updates, one trajectory can even span several versions, so the version belongs to token spans, not just to samples. A strict gate only consumes data with age at most k, and importance weights, usually truncated, correct the remaining shift. | 异步带来了数据陈旧。把样本的年龄定义为学习器当前版本减去生成它的版本。如果生成中途更新了权重，一条轨迹甚至会跨越多个版本，所以版本号应该绑定到 token 片段，而不只是绑定到样本。严格的门限只消费年龄不超过 k 的数据，剩下的分布偏移再用重要性权重修正，通常还会做截断。 |
| The key insight from the code review is that async is not a boolean. A bounded queue limits how far producers run ahead, but not the age of each sample when it's trained on. An admission window decides whether new tasks start, but a slow in-flight task can still finish stale. A consume-time gate filters by version, but dropping samples changes the data distribution. Importance correction fixes known probability shifts, but not for free in variance. And mid-flight updates buy overlap at the cost of mixed-policy trajectories. | 源码审查最重要的结论之一是：异步不是一个布尔开关。有界队列限制生产者领先多少，但不限制每条样本被训练时的年龄。准入窗口决定新任务能否启动，但一个很慢的在途任务完成时仍可能已经过时。消费时的年龄门限按版本过滤，但丢弃样本会改变数据分布。重要性修正能处理已知概率的偏移，但要付出方差代价。生成中途更新权重换来了重叠，代价是轨迹混入了多个策略。 |
| Real frameworks implement different subsets. OpenRLHF mainly uses queue back-pressure. SkyRL's staleness setting governs admission and capacity. Miles and prime-rl can filter by age at consumption, and verl offers drop or wait. A staleness value of two may not mean the same age in two frameworks. And if long trajectories are discarded, hard problems quietly disappear from training. Measure age, length and difficulty explicitly. | 真实框架实现的是不同的子集：OpenRLHF 主要靠队列背压；SkyRL 的陈旧度配置作用于准入和总容量；Miles 和 prime-rl 可以在消费时按年龄过滤；verl 提供 drop 和 wait 两种策略。同样配置成 2，在两个框架里不一定代表同样的年龄。而如果长轨迹被丢弃，难题就会悄悄从训练中消失。要按年龄、长度和难度分层，显式地度量。 |

## 11 · Train–inference consistency: five error sources · 训练—推理一致性：五个误差来源
*PART 4 · WHERE SYSTEMS REALLY DIFFER · 第四部分 · 系统真正的差异*

⏱ EN 15:30 · 中文 15:35

| English | 中文 |
|---|---|
| Consistency between training and inference has at least five independent sources of error. Re-tokenizing text can change the token sequence, so store the exact generated IDs. The behavior policy may be stale, so record versions and correct for it. MoE routers may pick different experts in the trainer, so replay the rollout-time routes. Top-p or top-k sampling truncates the distribution. And different kernels, precision or batch shapes produce different log-probs for the same weights. Each mechanism fixes one source; none substitutes for another. | 训练与推理之间的一致性，至少有五个相互独立的误差来源。把文本重新分词可能改变 token 序列，所以要保存实际生成的 token ID。行为策略可能已经陈旧，所以要记录版本并做修正。MoE 路由器在训练端可能选中不同的专家，所以要重放 rollout 时的路由。top-p 或 top-k 采样截断了分布。不同的 kernel、精度或 batch 形状，也会让同一份权重算出不同的对数概率。每种机制只修复一个来源，彼此不能替代。 |
| The top-p case is a clean piece of math. Nucleus sampling draws from the smallest set S whose probability mass reaches p, renormalized by Z_S. So the true behavior log-prob is log pi minus log Z_S. If the trainer recomputes a full-vocabulary softmax, every ratio is off by a factor Z_S that varies from token to token. Sampler-support replay fixes this by normalizing over the same candidate set during training. | top-p 是一个很干净的数学例子。核采样从概率质量达到 p 的最小集合 S 中抽取，并用 Z_S 重新归一化。所以真实的行为对数概率是 log pi 减去 log Z_S。如果训练端用全词表 softmax 重算，每一个比率都会偏差一个随 token 变化的因子 Z_S。采样支持集重放的做法，是在训练时也在同一个候选集合上归一化。 |

## 12 · What "customize every detail" really means · “定制每个算法细节”到底意味着什么
*PART 4 · WHERE SYSTEMS REALLY DIFFER · 第四部分 · 系统真正的差异*

⏱ EN 16:33 · 中文 16:38

| English | 中文 |
|---|---|
| Now, what does it mean to customize every detail of an algorithm? There are six distinct freedoms: the objective, the estimator, the data lifetime, the model state, the update rule, and execution scheduling. Each needs a different kind of access. A custom-loss hook usually covers only part of the first row. A new critic, a target network or replay touches rows three to five, which live in the learner loop, not in a loss callback. | 那么，“能定制算法的每个细节”到底是什么意思？一共有六种不同的自由度：目标函数、估计器、数据寿命、模型状态、更新规则和执行调度。每一种需要的访问权限都不一样。框架提供的自定义 loss 接口，通常只覆盖第一行的一部分。新增 critic、target 网络或经验回放，会触及第三到第五行，它们位于学习器主循环里，而不在损失回调里。 |
| Here's a concrete example of why the inputs matter. Exact entropy needs the full vocabulary distribution. If your hook only receives the log-prob of the sampled token, the naive move, adding minus log pi of that token as a bonus, has zero expected gradient, because the score function has mean zero under on-policy sampling. The correct estimator weights the score by the log-prob itself. And exact entropy needs full logits, which under tensor parallelism are sharded across the vocabulary. | 举一个具体例子，说明输入为什么重要。精确的熵需要完整的词表分布。如果你的接口只拿到被采样 token 的对数概率，那么朴素做法，也就是把这个 token 的负对数概率当作奖励项，期望梯度为零，因为在同策略采样下，得分函数的期望是零。正确的估计器要用对数概率本身给得分函数加权。而精确熵需要完整 logits，在张量并行下，这些 logits 是按词表切分在不同卡上的。 |

## 13 · Where each framework lets you in · 各框架的算法入口
*PART 4 · WHERE SYSTEMS REALLY DIFFER · 第四部分 · 系统真正的差异*

⏱ EN 17:33 · 中文 17:42

| English | 中文 |
|---|---|
| With those freedoms in mind, here's where each framework lets you in, based on the pinned source code. TRL gives you trainer subclasses and loss functions, but you inherit the upstream Transformers lifecycle. SkyRL has separate registries for policy losses and advantage estimators, plus a clean generator interface. slime's custom loss receives differentiable logits and a reducer, which is unusually direct. verl's registries pass selected-token log-probs, so full-logit objectives need deeper worker changes. ROLL lets you override the worker's loss on logits. And OpenRLHF is a few explicit files that are easy to fork. | 带着这些自由度，再看各框架基于固定版本源码的算法入口。TRL 提供 Trainer 子类和损失函数入口，但你继承的是上游 Transformers 的生命周期。SkyRL 为 policy loss 和优势估计器分别提供注册表，还有清晰的 generator 接口。slime 的自定义 loss 直接拿到可微的 logits 和 reducer，这一点非常直接。verl 的注册表传入的是选中 token 的对数概率，因此需要全 logits 的目标得深入修改 worker。ROLL 允许在 logits 上覆写 worker 的损失。OpenRLHF 则是几个明确的文件，很容易维护一个小 fork。 |
| NeMo RL has a design worth copying everywhere: its loss interface receives the global count of valid tokens and sequences, so distributed normalization is part of the contract instead of an accident of micro-batching. The right comparison is not which framework lists more algorithms. It's how many boundaries your change must cross, which fast paths survive it, and what it costs to maintain. | NeMo RL 有一个值得到处借鉴的设计：它的损失接口直接接收全局有效 token 数和序列数，于是分布式归一化成了契约的一部分，而不是微批划分的偶然结果。正确的比较方式不是看谁列出的算法更多，而是看你的改动要跨越多少边界、哪些高性能路径还能保留，以及长期维护要付出多少成本。 |

## 14 · The framework landscape at a glance · 主流框架定位速览
*PART 4 · WHERE SYSTEMS REALLY DIFFER · 第四部分 · 系统真正的差异*

⏱ EN 18:40 · 中文 18:52

| English | 中文 |
|---|---|
| Here is the landscape at a glance. These are judgments about architecture fit, not a speed ranking. verl is the broadest platform, with HybridFlow separating control flow from distributed compute. OpenRLHF composes Ray, DeepSpeed and vLLM in a way HF teams find natural. slime is a thin algorithm layer on Megatron and SGLang, strong for large MoE. Miles, forked from slime, invests in weight transfer, consistency and recovery. SkyRL offers modular generator and environment interfaces. AReaL co-designs asynchronous sampling with the learning rule. NeMo RL integrates NVIDIA's large-model stack and distillation. ROLL handles multi-domain, multi-reward pipelines. RLinf targets embodied and VLA training, and Prime-RL couples async training with verifier environments. | 这是主流框架的速览。这里是架构适配上的判断，不是速度排名。verl 是覆盖面最广的平台，HybridFlow 把控制流与分布式计算分开。OpenRLHF 以 HF 团队熟悉的方式组合了 Ray、DeepSpeed 和 vLLM。slime 是架在 Megatron 和 SGLang 上的薄算法层，擅长大 MoE。从 slime 分叉出来的 Miles，重点投入权重传输、一致性和故障恢复。SkyRL 提供模块化的 generator 与环境接口。AReaL 把异步采样和学习规则一起设计。NeMo RL 整合了 NVIDIA 的大模型技术栈和蒸馏。ROLL 擅长多领域、多奖励的流水线。RLinf 面向具身与 VLA 训练，Prime-RL 则把异步训练和验证器环境结合在一起。 |
| Around them: fine-tuning toolchains for SFT and preference work; agent adapters that capture real agent traces; environment hubs; and Tinker, a hosted API where you write the loop and the service runs distributed training. Also check maintenance status. torchtune is no longer actively developed, ReaLHF is archived in favor of AReaL, and TorchForge's direction moved into TorchTitan. | 围绕它们的还有：做 SFT 和偏好优化的微调工具链；捕获真实 Agent 轨迹的接入层；环境平台；以及 Tinker，一个托管 API，你来写训练循环，服务端负责分布式训练。另外要核对维护状态：torchtune 已不再积极开发，ReaLHF 已归档、转向 AReaL，TorchForge 的方向则并入了 TorchTitan。 |

## 15 · Same atoms, different recipes · 同一批原子，不同的配方
*PART 5 · ANATOMY: ATOMS, ADDRESSES, DEPTH · 第五部分 · 拆解：原子、地址与深度*

⏱ EN 20:02 · 中文 20:08

| English | 中文 |
|---|---|
| Zoom out, and the frameworks stop looking like a dozen unrelated systems. They are recipes built from a handful of shared atoms. At the bottom, everyone shares PyTorch, torch dot distributed with NCCL for tensors, and the same CUDA kernels. Above that, every recipe picks one training backbone, FSDP2, DeepSpeed ZeRO or Megatron-Core; one generation engine, vLLM or SGLang; and an orchestrator, usually Ray. | 把视角拉远，这些框架就不再像十几个互不相关的系统，而是用少数几种共享“原子”搭出来的配方。最底层大家共用 PyTorch、负责张量通信的 torch.distributed 与 NCCL，以及同一批 CUDA kernel。往上，每个配方选一个训练底座：FSDP2、DeepSpeed ZeRO 或 Megatron-Core；选一个推理引擎：vLLM 或 SGLang；再选一个编排层，通常是 Ray。 |
| Now read the recipes. OpenRLHF is Ray, plus DeepSpeed ZeRO, plus vLLM. slime is Megatron plus SGLang. SkyRL is FSDP2 or Megatron, plus vLLM. verl, ROLL and NeMo RL ship both backbones and both engines. The atoms repeat; what each framework really adds is the last column, its connective layer: the trajectory format, the control loop, and the hooks it exposes. That layer decides what you can change, and how cheaply. | 再来读配方。OpenRLHF 是 Ray 加 DeepSpeed ZeRO 加 vLLM。slime 是 Megatron 加 SGLang。SkyRL 是 FSDP2 或 Megatron 加 vLLM。verl、ROLL、NeMo RL 两种底座、两种引擎都带。原子在反复出现；每个框架真正新增的是最后一列，也就是它的连接层：轨迹格式、控制循环，以及它开放的钩子。正是这一层决定了你能改什么、改起来有多便宜。 |
| One warning before combining atoms yourself. A recipe is a path that is already connected end to end: model, training backend, weight export, engine, trajectory, update. Feature checkboxes do not add up. The source review found concrete holes: OpenRLHF's Ray plus vLLM online path has no LoRA; Miles' FSDP2 backend is data-parallel only; NeMo RL's async replay path excludes SGLang; and TRL's async trainer runs on FSDP2, not ZeRO. | 自己组合原子之前，先记住一点：一个配方是一条已经端到端贯通的路径——模型、训练后端、权重导出、推理引擎、轨迹、更新。功能列表上的勾不能相加。源码审查找到了具体的缺口：OpenRLHF 的 Ray 加 vLLM 在线路径不支持 LoRA；Miles 的 FSDP2 后端只有数据并行；NeMo RL 的异步 replay 路径排除了 SGLang；TRL 的异步 trainer 只跑在 FSDP2 上，不支持 ZeRO。 |

## 16 · Four trade-off axes · 四条取舍轴
*PART 5 · ANATOMY: ATOMS, ADDRESSES, DEPTH · 第五部分 · 拆解：原子、地址与深度*

⏱ EN 21:34 · 中文 21:43

| English | 中文 |
|---|---|
| Why do recipes differ? Each one is a point on four trade-off axes. The first is the training backbone. FSDP2 keeps your PyTorch model and control flow intact, but leaves rollout, critics and sync to you. DeepSpeed ZeRO shards states and offloads, but its engine owns accumulation and the step boundary. Megatron-Core brings tensor, pipeline, expert and context parallelism for large MoE, at the price of model mapping and a schedule that reaches into training. The second axis is the generation engine: vLLM with its native weight-transfer engines, or SGLang with update sessions, pause, and memory release. | 为什么配方各不相同？每个配方都是四条取舍轴上的一个点。第一条是训练底座。FSDP2 保留你的 PyTorch 模型和控制流，但 rollout、critic 和同步都要你自己接。DeepSpeed ZeRO 做状态分片和 offload，但梯度累积和 step 边界由它的引擎掌握。Megatron-Core 带来张量、流水线、专家和上下文并行，能训练大 MoE，代价是模型映射，以及深入训练过程的调度。第二条是推理引擎：vLLM 有原生的权重传输引擎；SGLang 有权重更新 session、暂停和显存释放。 |
| The third axis is placement. Colocated and synchronous keeps the policy fresh with fewer copies; separated and asynchronous hides the long tail, but adds staleness, weight transfer and correction, and it is not monotone: NeMo RL's own table shows async faster on one model and slower on another. The fourth axis is the connective layer. A thin one, like slime or OpenRLHF, hands your hook real tensors but exposes the backend directly. A platform, like verl or ROLL, carries many backends and roles, but you learn its abstractions before your learning rule is visible. | 第三条是资源布局。共置同步让策略保持新鲜、副本更少；分离异步能掩盖长尾，但会带来陈旧、权重传输和修正，而且并非单调提速：NeMo RL 自己的性能表里，异步在一个模型上更快，在另一个模型上反而更慢。第四条是连接层的厚度。像 slime、OpenRLHF 这样的薄连接层，让你的钩子直接拿到真实张量，但后端的复杂度也直接摆在你面前；像 verl、ROLL 这样的平台承载多后端、多角色，但要先学会它的抽象，才能看清你的学习规则。 |

## 17 · Every term of the objective has an address · 目标函数里每一项都有地址
*PART 5 · ANATOMY: ATOMS, ADDRESSES, DEPTH · 第五部分 · 拆解：原子、地址与深度*

⏱ EN 22:54 · 中文 23:05

| English | 中文 |
|---|---|
| So far, frameworks as recipes. Now let's go the other way, from the math to the code. Write out the full online objective, and number every place you might want to change. Positions one to six are the data rule: which prompts, who generates, the reward, how long data lives, how stale it may be, and the behavior correction. They decide which samples enter the gradient, and with what weight. | 前面把框架看成配方。现在反过来，从数学走向代码。把完整的在线目标函数写出来，给每一个你可能想改的位置编上号。第一到第六个位置是数据规则：用哪些 prompt、谁来生成、奖励、数据活多久、允许多旧，以及行为策略修正。它们决定哪些样本进入梯度，以及带着多大的权重进入。 |
| Positions seven to twelve are the learning rule: the advantage, the surrogate on sampled tokens, anything that needs the full distribution, the normalization, extra networks, and the update rule itself. Thirteen closes the loop by publishing weights back to the engine. Notice that the two halves are coupled through position six: the moment a data rule reuses or delays samples, the learning rule must correct for the behavior policy that actually produced them. | 第七到第十二个位置是学习规则：优势估计、采样 token 上的目标、任何需要完整分布的项、归一化、额外网络，以及更新规则本身。第十三个位置把权重发布回推理引擎，闭合循环。注意两半是通过第六个位置耦合在一起的：数据规则一旦重用或延迟样本，学习规则就必须针对真正产生这些样本的行为策略做修正。 |

## 18 · The address book: position × framework · 地址簿：位置 × 框架
*PART 5 · ANATOMY: ATOMS, ADDRESSES, DEPTH · 第五部分 · 拆解：原子、地址与深度*

⏱ EN 23:47 · 中文 24:00

| English | 中文 |
|---|---|
| Here is the address book: thirteen positions against eleven routes, each cell colored by how deep you must go. L zero is a config flag. L one is a declared hook. L two means subclassing a stage, L three means editing framework files, and L four means the change crosses worker or backend boundaries. Dashed cells were not covered by the source review. Read it by rows. Generation and staleness, rows two and five, are config or hooks almost everywhere: this is what these runtimes are built for. | 这是一本地址簿：十三个位置对十一条路线，每个格子的颜色表示你需要挖多深。L0 是改一个配置。L1 是接入一个声明好的钩子。L2 是继承并覆写某个阶段，L3 是修改框架源文件，L4 则意味着改动要跨越 worker 或后端的边界。虚线格是源码审查没有覆盖的位置。按行来读。生成和陈旧度，也就是第二行和第五行，几乎在所有框架里都是配置或钩子：这正是这些运行时擅长的事情。 |
| Rows seven and eight, advantage and surrogate, are hooks in verl, SkyRL, slime, NeMo RL and prime-rl, but file edits in OpenRLHF and Miles. Row nine splits the field: slime and NeMo RL hand your loss the logits, while verl, SkyRL and prime-rl pass only selected-token log-probs, so an exact entropy becomes an L four change. And rows eleven and twelve, new networks and new update rules, are L four nearly everywhere. That pattern is the decision rule: pick the route where your change is shallowest; if it is L four on every route, own the learner and reuse the runtime. | 第七、第八行，优势和目标函数，在 verl、SkyRL、slime、NeMo RL、prime-rl 里是钩子，在 OpenRLHF 和 Miles 里则要改源文件。第九行把框架分成两派：slime 和 NeMo RL 把 logits 交给你的 loss，而 verl、SkyRL、prime-rl 只传选中 token 的对数概率，于是精确熵就成了一个 L4 的改动。第十一、十二行，新增网络和新的更新规则，几乎处处都是 L4。这个规律本身就是决策规则：选你的改动最浅的那条路线；如果它在每条路线上都是 L4，就自己持有 learner，复用运行时。 |

## 19 · Three changes, traced · 三个改动，逐项追踪
*PART 5 · ANATOMY: ATOMS, ADDRESSES, DEPTH · 第五部分 · 拆解：原子、地址与深度*

⏱ EN 24:59 · 中文 25:19

| English | 中文 |
|---|---|
| Let's trace three concrete changes through the book. First, a data rule: prioritize hard prompts and reuse each group up to k times. That touches positions one, four, five and six. The shallowest routes are Miles' replaceable data buffer and verl's custom replay buffer. Write the rule as put and get; gate age at consumption; and remember that a reused sample is off-policy, so it must carry its behavior log-probs and version. Note that the default buffers in Miles and ROLL delete samples once consumed. | 我们用地址簿追踪三个具体改动。第一个是数据规则：优先采样难题，每组数据最多重用 k 次。它触及第一、四、五、六个位置。最浅的路线是 Miles 可替换的 data buffer 和 verl 的自定义 replay buffer。把规则写成 put 和 get；在消费时判断年龄；并且记住，被重用的样本是离策略的，必须带着它的行为概率和版本号。注意，Miles 和 ROLL 的默认 buffer 在消费后就会删除样本。 |
| Second, a learning rule that needs the full distribution: an exact entropy bonus. That is positions nine and ten. Your hook must receive logits, because a naive minus log-pi bonus on sampled tokens has zero expected gradient. slime, NeMo RL and ROLL hand you logits; under tensor parallelism they are vocabulary-sharded, so use vocab-parallel ops, and reduce with the framework's reducer instead of averaging twice. | 第二个是需要完整分布的学习规则：精确熵奖励。它对应第九和第十个位置。你的钩子必须拿到 logits，因为在采样 token 上加朴素的负对数概率奖励，期望梯度为零。slime、NeMo RL 和 ROLL 会把 logits 交给你；张量并行下这些 logits 按词表切分，所以要用词表并行的算子，并用框架自带的 reducer 归约，而不是平均两次。 |
| Third, new state: a Q-critic with a target network, its own optimizer, and four updates per batch. Positions eleven, twelve and thirteen, and every framework column says L four. So own the learner: create the models and optimizers yourself on FSDP2 or DeepSpeed, respect their step rules, publish only the policy to vLLM or SGLang, and checkpoint every optimizer, the target, the buffer position and the version. | 第三个是新增状态：带 target 网络的 Q critic、独立的优化器、每批更新四次。它对应第十一、十二、十三个位置，而每个框架这一列都是 L4。所以自己持有 learner：在 FSDP2 或 DeepSpeed 上自己创建模型和优化器，遵守它们的 step 约定，只把策略参数发布给 vLLM 或 SGLang，checkpoint 时保存每个优化器、target、buffer 位置和版本号。 |

## 20 · Own the trajectory contract; reuse the runtime · 自己掌握轨迹契约，复用运行时
*PART 6 · CHOOSE & COMPOSE · 第六部分 · 选型与组合*

⏱ EN 26:29 · 中文 26:48

| English | 中文 |
|---|---|
| So what should you own? The review's answer is: own the trajectory contract. Keep exact token IDs per turn. Separate attention, loss and packing masks. Store behavior log-probs together with the sampling parameters. Keep old, reference and teacher log-probs distinct. Carry group, episode and turn IDs. Record versions per token span. And distinguish terminated from truncated, because they bootstrap differently. When you switch backends, this contract tells you exactly what information was lost. | 那么你应该自己掌握什么？报告的回答是：掌握轨迹契约。保存每一轮真实的 token ID；把 attention、loss 和 packing 的掩码分开；把行为对数概率和采样参数一起保存；旧策略、参考模型和教师的对数概率分别保存；携带 group、episode 和 turn 的 ID；按 token 片段记录版本；区分正常终止和截断，因为两者的 bootstrap 方式不同。切换后端时，这份契约能准确告诉你丢失了哪些信息。 |
| Then keep a minimal architecture of six modules: trajectory, algorithm, learner, rollout, runtime and evaluation. Write the algorithm in plain PyTorch, and reuse engines and runtimes for the rest. This loop makes the control flow explicit: collect, encode, plan updates, sample, forward, compute the objective, step, update targets, and publish. If your method is a single-actor policy gradient, hand most of these steps to an existing trainer. If it has multiple networks, own the learner loop. | 然后保持一个由六个模块组成的最小架构：trajectory、algorithm、learner、rollout、runtime 和 evaluation。算法用纯 PyTorch 写，其余部分复用现有的推理引擎和运行时。这个循环让控制流一目了然：采集、编码、规划更新、采样、前向、计算目标、更新参数、更新 target、发布权重。如果你的方法只是单 actor 的策略梯度，把大部分步骤交给现成的 trainer；如果涉及多个网络，就自己持有学习器主循环。 |
| Finally, name your clocks. An optimizer step, a published policy version and a recoverable checkpoint are three different events. And a real checkpoint for online training also needs the rollout queue, trajectory versions, replay position, environment state and sampler state, not just model weights. | 最后，给你的时钟命名。一次优化器 step、一个发布的策略版本、一个可恢复的 checkpoint，是三种不同的事件。而在线训练真正的 checkpoint，除了模型权重，还需要 rollout 队列、轨迹版本、replay 消费位置、环境状态和采样器状态。 |

## 21 · A decision procedure · 选型决策流程
*PART 6 · CHOOSE & COMPOSE · 第六部分 · 选型与组合*

⏱ EN 27:52 · 中文 28:17

| English | 中文 |
|---|---|
| Now we can turn all of this into a decision procedure. First question: does your training data come from the current policy? If not, you're offline. Start with TRL plus PEFT; add DeepSpeed or FSDP2 for scale; use LLaMA-Factory, ms-swift or Axolotl for config-driven recipes; QLoRA or Unsloth when memory is the constraint; and Tunix on TPUs. | 现在可以把这些整理成一个决策流程。第一个问题：训练数据是否由当前策略产生？如果不是，你处在离线场景：从 TRL 加 PEFT 起步；需要扩展规模时加上 DeepSpeed 或 FSDP2；想要配置驱动的配方，用 LLaMA-Factory、ms-swift 或 Axolotl；显存是瓶颈时用 QLoRA 或 Unsloth；在 TPU 上则选 Tunix。 |
| If you're online, ask: does your idea fit inside the generator, advantage and policy-loss interfaces? If yes, reuse a complete runtime. The review's default first prototype is SkyRL on FSDP2 with vLLM; verl for a shared, multi-backend team platform; OpenRLHF if you already have HF and DeepSpeed assets; slime, Miles or NeMo RL once you've committed to Megatron and large MoE. If not, because you need new networks, replay or a different update rule, own the learner, and reuse FSDP2 or DeepSpeed for execution and vLLM or SGLang for generation. | 如果是在线场景，再问：你的想法能否完全落在 generator、优势估计和 policy loss 这三个接口里？如果能，就复用一个完整运行时。报告推荐的第一个原型是 SkyRL 的 FSDP2 加 vLLM 路径；团队共享、多后端的平台选 verl；已有 HF 和 DeepSpeed 积累选 OpenRLHF；确定走 Megatron 和大 MoE 路线后，选 slime、Miles 或 NeMo RL。如果不能，比如需要新增网络、经验回放或不同的更新规则，就自己持有学习器，执行层复用 FSDP2 或 DeepSpeed，生成层复用 vLLM 或 SGLang。 |
| Specialized cases. If you already have a working agent, keep its harness and capture traces with rLLM, Agent Lightning or ART. If asynchrony itself is your research, study AReaL and PipelineRL. For embodied or VLA training, evaluate RLinf. And if you'd rather not run GPUs at all, Tinker exposes forward-backward, optimizer step and sampling as an API, within LoRA-centric limits. | 还有几种特殊情况。如果你已经有一个运行良好的 Agent，保留它的执行框架，用 rLLM、Agent Lightning 或 ART 捕获轨迹。如果异步本身就是你的研究课题，重点研究 AReaL 和 PipelineRL。做具身或 VLA 训练，评估 RLinf。如果你根本不想运维 GPU，Tinker 把前向反向、优化器 step 和采样做成了 API，但能力边界以 LoRA 为主。 |

## 22 · Validate: two stages and a code checklist · 验证：两阶段评测与源码检查清单
*PART 6 · CHOOSE & COMPOSE · 第六部分 · 选型与组合*

⏱ EN 29:24 · 中文 29:56

| English | 中文 |
|---|---|
| How do you validate a choice? Use two stages. Stage one fixes the trajectories: feed every candidate the same tokens, masks, rewards and behavior log-probs, and compare loss, gradients, a single parameter update, memory and time. That separates execution differences from sampling differences. Stage two runs the closed loop with the same start checkpoint, data, evaluation and budget, aligning every knob, from group size and sampling parameters to loss reduction, KL, clipping and filtering. Then compare GPU-hours to target across several seeds. | 怎样验证一个选择？分两个阶段。第一阶段固定轨迹：给每个候选框架相同的 token、掩码、奖励和行为对数概率，比较 loss、梯度、一次参数更新、显存和耗时。这样就把执行差异和采样差异分开了。第二阶段跑完整闭环：相同的起始 checkpoint、数据、评测集和预算，并对齐所有旋钮，从 group size、采样参数，到 loss 归约方式、KL、clip 和样本过滤。然后在多个随机种子下比较达到目标所需的 GPU 小时。 |
| And when reading code, check these questions, which matter more than the number of built-in algorithms. Where exactly is the final mean taken, and does the engine scale again? Are group statistics aggregated across ranks? Are top-p log-probs normalized on the support? Does any branch silently detach a tensor? Does weight sync push everything, run quantization post-processing, and invalidate caches? Does the buffer delete samples, and does resume restore everything? | 读源码时，请检查下面这些问题，它们比内置算法的数量更重要：最终的均值到底在哪个维度上取，引擎是否又缩放了一次？同一个 prompt 组的奖励统计是否跨 rank 聚合？top-p 的对数概率是否在支持集上归一化？有没有哪个分支悄悄 detach 了张量？权重同步是否推送了全部参数、执行了量化后处理、并让缓存失效？buffer 是否在采样后删除数据，恢复时是否把一切都恢复了？ |
| Keep a tiny pure-PyTorch reference loss and step. After every kernel, precision or backend change, compare against it. Then improve efficiency one bottleneck at a time, in profile order. It's slower on day one, and much faster over the life of a project. | 保留一个很小的纯 PyTorch 参考 loss 和 step。每次更换 kernel、精度或后端之后，都和它对比。然后按 profile 的顺序，一次只解决一个瓶颈。第一天会慢一些，但在整个项目周期里会快得多。 |

## 23 · Five takeaways · 五条结论
*SUMMARY · 总结*

⏱ EN 30:46 · 中文 31:24

| English | 中文 |
|---|---|
| Let's wrap up with five takeaways. One: every post-training method is one gradient with two choices, the data source and the token weight; fixed data makes it a trainer problem, on-policy data makes it a systems problem. Two: compose by layer, one component per role, and verify the whole path before you trust a feature matrix. Three: frameworks differ less in their algorithm lists than in scheduling, consistency, MoE depth, weight transfer and recovery. | 最后总结五点。第一，所有后训练方法都是同一个梯度加两个选择：数据来源和 token 权重；数据固定就是训练器问题，在线数据就是系统问题。第二，按层组合，每个角色一个部件，在相信功能矩阵之前先验证整条路径。第三，框架之间的差距，与其说在算法列表，不如说在调度、一致性、MoE 支持深度、权重传输和故障恢复。 |
| Four: find your change's address in the objective. If it stays in hooks, reuse a framework; if it crosses boundaries everywhere, own the learner and the trajectory contract, and reuse the runtime. Five: optimize cost to target, one bottleneck at a time, with a small reference implementation as your anchor. Thanks for watching. | 第四，先找到你的改动在目标函数里的地址：落在钩子里就复用现成框架；处处都要跨边界，就自己掌握 learner 与轨迹契约，复用运行时。第五，优化达到目标的成本，一次解决一个瓶颈，并以一个小型参考实现为锚点。感谢观看。 |
