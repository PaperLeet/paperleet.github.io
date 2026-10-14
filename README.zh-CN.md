# PaperLeet

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="web/public/brand/paperleet-logo-dark.svg">
  <img src="web/public/brand/paperleet-logo-light.svg" alt="PaperLeet" width="320">
</picture>

[English](README.md) | **简体中文**

读论文，写实现，用数学性质和真实行为验证理解。

当前试点包含 **46 篇论文、60 个核心模块**。先选择论文，了解核心贡献和本次练习范围，再进入模块的 LeetCode 风格分栏工作区。辅助代码直接提供，练习聚焦论文的关键机制。“论文完成”表示选定核心模块全部通过。论文组织保存在 `papers.yaml`，题面、实现、测试和可视化数据来自 `problems/`。

## 本地启动

产品名称为 **PaperLeet**，CLI 使用 `paperleet` 命令；此前的 `leetpaper` 别名继续兼容。

需要 Python 3.10+、Node.js 20.19+。在仓库根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
paperleet export
npm --prefix web ci
make dev
```

打开终端显示的地址（默认 **http://127.0.0.1:5173**）。以后在根目录运行 **`make dev`** 即可同时启动网页与判题服务；按 Ctrl-C 会关闭两个进程。

## 部署邀请制内测

仓库提供小规模在线内测的部署包，用户可以在自己的电脑上打开网页，完成 **60 个精选模块**。默认机构模式需要邀请和机构邮箱验证。首批测试可以使用 `PAPERLEET_AUTH_MODE=test`：管理员为每位测试者创建登录账号和访问码，无需邮箱注册，也无需配置 Supabase 或 SMTP，具体见 [内测账号指南](docs/test-accounts.zh-CN.md)。线上判题在限制资源的独立沙箱中运行；草稿、判题结果与提交历史按账号私密保存。浏览器草稿缓存按用户隔离；其他设备更新了同一草稿时，需要明确选择保留哪个版本。

按照 [部署指南](docs/hosting.zh-CN.md) 配置服务器、沙箱、持久化存储和所选账号模式；只有机构模式需要邮件服务与邀请名单。仓库提供部署文件，不会自动发布到公网。本地 `make dev` 工作流保持不变，仍使用你自己的机器执行代码。

暂时没有服务器时，可以使用 [Cloudflare Tunnel + 本机 Docker](docs/cloudflare-tunnel.zh-CN.md)，把自己的域名接到电脑上的内测服务。电脑需要保持开机、联网和不休眠；账号和答题数据保存在本机的私有 Docker 数据卷中。

## 开始第一题

1. 在题库中选择论文，例如 **6 · DDPM**，再打开“前向加噪”模块。
2. 阅读模块目标，在右侧编辑器中补全标注为需要实现的函数；调度器等辅助函数已经提供。
3. 点击“运行全部测试”，等待判题结果自动显示。
4. 根据各层测试的失败原因、期望值与实际值修改代码，再次运行。

完整测试全部通过后，完成状态自动更新并解锁坑点。只跑部分测试不会解锁“全部通过”状态。本地模式的草稿和学习进度保存在当前浏览器中；托管模式中，用户登录后，草稿与判题结果会同步到账号。右上角可切换中文 / English，界面、题面和讲解随之切换。论文入口使用数字编号 1–46；模块编号 1–60 和内部目录名保持不变，兼容已有草稿和 CLI 命令。后续运行失败不会抹掉此前已通过的记录。旧草稿会保留；如需新版辅助代码，可先备份代码，再使用编辑器的重置功能。

顶部 **浅色 / 深色** 按钮用于切换外观。首次访问跟随系统主题，手动选择后会在当前浏览器记住。题库集中展示分类、搜索、筛选和模块进度；技术简介与可展开的练习范围放在论文详情页。

拖动工作区分隔条，可调整题面与代码区的左右宽度、代码与测试结果区的上下高度，比例会保存在当前浏览器。双击分隔条，或选中后按 Enter，可恢复默认比例；方向键可微调。运行按钮始终保留在底部，窄屏上题面和编辑器自动上下排列。

顶部 **精选合集** 通过彩色卡片呈现 LLM 基础、强化学习、扩散与 Flow、后训练、JEPA / 世界模型和视觉生成，卡片显示论文数与题目数，下方小标签用于分类筛选。合集以论文为单位，可以交叉收录；独立链接例如 `#/collections/llm-basics`，原有 `#/lists/...` 链接继续可用。切换合集会清空筛选，合集成员和双语卡片文案维护在 `web/src/paperCollections.ts`。

**随机一题** 从当前合集及筛选范围内打开一道未开始的模块题，排除有运行记录、已通过记录或已编辑草稿的题目；未修改的初始代码仍可被抽中。按钮会等待学习进度加载，没有符合条件的题目时禁用并提示。

托管模式的题目页展示公开汇总统计：做题人数、通过人数、提交总数、通过次数、未通过次数和通过率（通过次数 / 提交总数）。同一账号在每道题的人数统计中只计一次，多次完整提交则分别计数；局部测试和判题服务故障不计入。已有的服务器判题记录会自动纳入统计，数据随数据库持久保存，本地开发运行不参与统计。尚无提交时，通过率显示为“—”。

网页通过判题服务执行代码并自动显示结果，无需手动处理报告文件。本地服务监听 `127.0.0.1:8765`，Vite 代理 `/api` 请求；代码保存到 `workspace/`，由本机 Python/PyTorch 执行。这一启动方式面向个人本地使用。多人在线判题请使用独立的托管配置，不要直接公开开发服务。

## 社区与用户账号

在顶部 **社区 / Community** 浏览讨论、发帖与回复、点赞或转发，并关注其他研究者。个人公开主页展示帖子、回复、点赞、转发，以及关注与粉丝列表。机构模式的写入操作需要先验证 `institutions.yaml` 中已批准机构的邮箱；线上还需要管理员邀请。测试模式使用管理员分配的账号和访问码登录，个人主页标注“内测账号”，不宣称已通过机构邮箱认证。

个人主页展示去重后的累计通过和尝试题数、难度分布、年度活跃热力图、活跃天数和连续天数，以及最近通过的题目。日历使用查看者浏览器的时区，区分提交次数、每天尝试／通过的不同题数，以及首次通过题数。仅统计服务器判定的完整提交，局部测试和判题服务故障不计入；重复通过不会增加累计通过题数。社区统计包括帖子、回复、收到的点赞和转发，以及关注关系。

昵称和生成的头像始终公开。已验证机构、国家／地区分别提供展示开关，默认均隐藏；内测账号没有机构字段，完整邮箱始终私有。国家由用户选填，不查询 IP。在账号设置中关闭 **允许展示学习进度** 后，个人学习统计仅自己可见。代码、草稿和详细判题报告始终私有。本地练习仍保存在浏览器，不会自动上传至主页。题目总体统计仍只公开匿名汇总数字。

机构邮箱登录使用 Supabase Auth，需配置服务端环境；内测账号单独创建，登录不经过邮件服务。复制 `.env.example` 为 `.env` 并填写后，必须在启动服务的终端显式导出环境变量。OTP 邮件模板、SMTP、机构审核与隐私规则见 [社区配置指南（英文）](docs/community.md)。在线内测请使用 [部署指南](docs/hosting.zh-CN.md) 中的独立判题沙箱和邀请控制。

## Submit paper

**在线内测模式** 下，已登录用户可以提交论文提案并查看自己的提交，管理员可以查看全部提案。生成、验证和发布由维护者通过离线流程完成。线上判题范围固定为部署镜像中的 60 个精选模块。

**本地维护模式** 下，点击顶部 **Submit paper / 提交论文**，填写论文链接、可选的官方代码链接和希望练习的内容。模型读取来源，生成中英文题面、starter、参考实现、分层测试和错误变体；通过隔离验证后，进入人工评审。只有人工批准的当前版本才会分配数字模块编号并进入论文题库。导出时会根据 arXiv 标识或来源 URL，将同篇论文的模块合并到一个入口。目前生成器每次提交生成一个候选模块。生成时会寻找作者 toy 和论文图示，明确标注教学改编，并让 T3 通过有大小限制的 `artifact` fixture 输出实际测量的曲线和指标，供人工审核。

生成需要在启动服务的终端配置 `OPENAI_API_KEY` 和 `PAPERLEET_MODEL`。验证需要启动 Docker，并在仓库根目录构建镜像：

```bash
docker build -t paperleet-harness:local paperleet/submissions
```

未配置模型或 Docker 时仍可保存提交，网页会说明生成或验证尚缺的条件。候选材料与评审记录保存到 `.paperleet/submissions/`。本地评审由工作区所有者完成；托管模式不会向普通用户开放生成、验证或发布操作。完整流程、隔离边界与人工评审标准见 [harness 设计](docs/paper-submission-harness.md)。

## 维护者：CLI 与质量门

维护题目、调试判题或检查参考解时，可以使用 CLI：

```bash
paperleet list
paperleet init diffusion/param_conversion
# 编辑 workspace/diffusion/param_conversion.py
paperleet test diffusion/param_conversion
paperleet test diffusion/param_conversion --layers T0,T2 --report reports/my-result.json
paperleet test rl/gae --impl problems/rl/gae/solution.py
paperleet mutate diffusion/param_conversion
paperleet ci
```

`init` 默认不覆盖已有代码；确实要重置时使用 `--force`。

`paperleet ci` 检查参考解全部通过、每个 mutant 至少被一个真实失败抓到、每个 mutant 差异不超过 10 行、重复三次一致、T3 CPU 预算与总运行时间。扩展后的 60 模块目录总预算为 300 秒（可通过 `--budget` 设置），单个 T3 的时限仍由各题包声明。结果写入 `reports/ci.json`，每题完整参考报告也保存在 `reports/`。导入/语法错误不会被当作有效 mutant 检出。

## 论文与核心模块

| 论文编号 | 论文 | 核心范围 | 模块编号 |
| --- | --- | --- | --- |
| 1 | GAE | 实现 GAE 的反向递推和 episode 边界处理。 | 1 |
| 2 | PPO | 实现 clipped surrogate 及其梯度行为；价值损失、归一化和统计代码已提供。 | 2 |
| 3 | DPO | 实现偏好损失与隐式奖励；token 对齐和序列 log-prob 计算已提供。 | 3 |
| 4 | DeepSeekMath · GRPO | 选取论文的 GRPO 模块：组内相对优势、截断目标和回答级聚合。 | 4 |
| 5 | Dr. GRPO | 实现去除组内标准差缩放与固定生成长度归一化这两处修正。 | 5 |
| 6 | DDPM | 实现闭式前向加噪，以及高斯后验和 epsilon 参数化的反向一步；调度器与采样循环已提供。 | 6, 8 |
| 7 | Progressive Distillation | 先实现 v 参数化转换，再推导两步教师到一步学生的蒸馏目标并验证更新。 | 7, 26 |
| 8 | DDIM | 实现 DDIM 单步更新和干净端点；时间步选择、循环与反演探索代码已提供。 | 9 |
| 9 | Flow Matching | 实现条件路径、目标速度和 CFM 回归损失；时间采样、Euler 积分和训练辅助代码已提供。 | 10 |
| 10 | GSPO | 实现序列概率比与 GSPO 损失，验证掩码、截断区域和梯度。 | 11 |
| 11 | GKD · On-policy Distillation | 实现学生 rollout 和全词表散度；在小型自回归模型上验证。OPD 在此指 on-policy distillation。 | 12 |
| 12 | DAPO | 实现组筛选与 token 级目标；采样系统和分布式训练不在本模块范围。 | 13 |
| 13 | LoRA | 实现低秩增量和线性前向，验证冻结边界及小型回归学习。 | 14 |
| 14 | QLoRA | 实现分块 NF4 量化、反量化与适配器前向；不模拟双重量化或分页优化器。 | 15 |
| 15 | MeanFlow | 实现平均流的 JVP 目标和梯度边界，并运行小型生成实验。 | 16 |
| 16 | JiT | 实现 x-prediction 到速度的转换与去噪目标，使用论文启发的低维流形嵌入实验。 | 17 |
| 17 | Drifting Models | 实现漂移场与停止梯度损失，在小型二维分布上验证生成器行为。 | 18 |
| 18 | DMD | 实现 score 差目标与配对回归项，用解析双峰分布检查模式覆盖。 | 19 |
| 19 | DMD2 | 实现双时间尺度更新与辅助目标，在小型实验中验证参数更新边界。 | 20 |
| 20 | Consistency Models | 实现边界参数化和相邻时刻的一致性蒸馏损失。 | 21 |
| 21 | Transformer | 分五题实现缩放点积注意力、多头注意力、LayerNorm 与 Add & Norm、正弦位置编码、逐位置 FFN。 | 22, 27–30 |
| 22 | AdamW | 实现矩估计、偏差修正与解耦权重衰减的一步更新。 | 23 |
| 23 | FlashAttention | 实现分块递推与重缩放。这是 CPU 算法练习，不评价 GPU kernel 性能。 | 24 |
| 24 | CLIP | 实现特征归一化、温度缩放和双向对比损失，使用小型配对检索实验。 | 25 |
| 25 | GLM-5 | 分别实现 Muon Split 逐头正交化、异步 RL 的双边重要性采样修正。 | 31–32 |
| 26 | Kimi K3 | 分别实现有界 KDA 递推、块级注意力残差、SiTU-GLU，并重绘论文的门函数和激活曲线。 | 33–35 |
| 27 | OPSD | 双上下文因果对齐与逐词表项散度截断。 | 36, 37 |
| 28 | SDPO | 同题反馈路由、reverse KL 与 top-k 尾部概率聚合。 | 38, 39 |
| 29 | REPA | 逐 patch 表征对齐损失与梯度边界。 | 40 |
| 30 | Rotation Trick | 保持量化前向值的旋转梯度变换。 | 41 |
| 31 | Differential Transformer | 已投影张量上的差分注意力与 lambda 参数化。 | 42 |
| 32 | Block Diffusion | 拼接干净和含噪 token 的单次前向注意力掩码。 | 43 |
| 33 | DistiLLM-2 | 教师轨迹的 skew KL 与学生轨迹的 skew reverse KL。 | 44 |
| 34 | PKPO | 连续奖励变换、policy loss 与论文的一维 Gaussian toy。 | 45 |
| 35 | SVDQuant | 平滑、SVD 低秩分解与 4-bit 残差前向。 | 46 |
| 36 | MoGe | 截断加权 L1 的鲁棒最优估计。 | 47 |
| 37 | DAPS | 线性 Gaussian 后验 score、Langevin 与独立重新加噪。 | 48 |
| 38 | DiffusionNFT | 正负 beta 隐式速度、双分支目标与微型训练。 | 49 |
| 39 | VAR | 下一尺度条件输入与注意力掩码；多尺度残差量化与重建。 | 50–51 |
| 40 | FD-Loss | 总体矩的 EMA、可微 Fréchet 距离和停止梯度的损失归一化。 | 52–53 |
| 41 | VICReg | 方差、视图一致性与非对角协方差损失。 | 54 |
| 42 | LeJEPA · SIGReg | 特征函数高斯性统计与随机切片聚合。 | 55 |
| 43 | VISReg | 尺度、切片形状、中心与多视图一致性。 | 56 |
| 44 | I-JEPA | 教师目标块、预测损失与 EMA 更新。 | 57 |
| 45 | LeWorldModel | 联合预测目标与潜空间 rollout；终点代价与 CEM 精英重拟合。 | 58–59 |
| 46 | Causal-JEPA | 保留身份锚点的对象遮挡及历史、未来预测。 | 60 |

首页按七个领域分类。新增模块包含 CPU toy 实验、出处与改编说明，以及 T0–T3 验证；实验页区分参考曲线和用户代码生成的 T3 结果。练习覆盖精选核心算法，不要求完整模型训练或 GPU kernel 性能复现。

每个模块通过 `module.required_functions` 和 `module.provided_functions` 声明需要实现及已经提供的函数，starter 仅将必做函数留空。判题仍会使用辅助代码调用用户实现。每个模块均包含中英文题面和坑点、参考解、分层测试与错误变体。模块 6、7 无需 T3；模块 5 的长度趋势只作展示，不据此判分。

## 可视化与静态构建

仓库提供预生成数据，可直接启动。首页只读取轻量 `data/index.json`；打开某个模块时，再加载包含题面、参考解和可视化数据的 `data/modules/` 详情。详情文件名包含内容哈希，可跨访问缓存。完整 `data/catalog.json` 继续供 CLI 工具和旧客户端使用。修改题目或实验后重新导出：

```bash
paperleet vizdata all
paperleet export
cd web
npm run check
npm run build
```

`vizdata` 使用参考实现、固定种子和两条 PyTorch CPU 线程。每题独立数据文件限制为 500 KB。前端能直接计算的概念图随滑块实时更新。`npm run check` 验证全部维护模块的图表渲染、轨迹解码、报告图、完成状态，以及私有工作区隔离、草稿冲突和通过记录保留。

`web/dist/` 可作为阅读论文和查看可视化的静态预览。账号工作区、代码编辑、同步和判题需要配置 `/api` 服务。交互式内测请使用 [托管部署包](docs/hosting.zh-CN.md)；仓库不会自动部署公开服务。

## 目录与贡献

```text
paperleet/         CLI、pytest runner、本地服务、公共数学工具
papers.yaml        论文、核心贡献、练习范围与模块顺序
institutions.yaml  经审核的机构邮箱域名
problems/          独立核心模块包（强化学习、微调、生成模型、架构、优化、表示学习）
web/               Vite + React + TypeScript
tests/             判题框架的集成测试
workspace/         个人代码，忽略于 Git
reports/           判题和 CI 报告，忽略于 Git
.paperleet/        本地社区数据库与论文提交材料，忽略于 Git
```

出题流程见 [CONTRIBUTING.md](CONTRIBUTING.md)，详细验收依据见 [10 题试点规格](PaperLeet_第一步_10题试点规格.md)。GitHub Actions 运行框架测试、三次质量门、数据生成和前端构建，上传报告与静态站构建产物。

本次实现的实测结果与验证范围见 [试点验收记录](docs/validation/pilot.md)。

26 篇论文扩展版的数值验证、toy 来源、网页检查及 harness 验证见[验证记录](docs/validation/core-curriculum.md)。

本轮 Oral/Spotlight 扩展新增 12 篇论文、14 个模块。论文来源、toy 归属与验证见[候选证据](docs/research/paper-candidates-2026-10-13.md)和[验证记录](docs/validation/oral-spotlight-expansion.md)。

VAR 与 FD-Loss 新增四个核心模块及 CPU 小实验，详见[来源与验证记录](docs/validation/var-and-fd-loss.md)。

浏览器端判题原型现覆盖 Schedules、Attention、LoRA：三题均在本机 CPU/WASM 执行完整测试，Attention/LoRA 支持双精度、解析自动微分和 Adam 训练；不支持的 API 或运行时故障自动回退服务端。编辑器可切换自动、仅浏览器和服务端模式。运行时按需加载；线上账号进度仍由服务端确认。[实现与兼容性说明](docs/browser-judge.md)。
