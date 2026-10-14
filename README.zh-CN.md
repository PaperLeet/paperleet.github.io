# PaperLeet

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="web/public/brand/paperleet-logo-dark.svg">
  <img src="web/public/brand/paperleet-logo-light.svg" alt="PaperLeet" width="320">
</picture>

[English](README.md) | **简体中文**

读论文，写实现，用数学性质和真实行为验证理解。

当前试点包含 **49 篇论文、63 个核心模块**。先选择论文，了解核心贡献和本次练习范围，再进入模块的 LeetCode 风格分栏工作区。辅助代码直接提供，练习聚焦论文的关键机制。“论文完成”表示选定核心模块全部通过。论文组织保存在 `papers.yaml`，题面、实现、测试和可视化数据来自 `problems/`。

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

## 仓库边界

本公开仓库包含前端、论文题库、浏览器运行时、本机判题与离线维护工具。`make dev` 无需账号、邮件服务或私有仓库即可独立运行，草稿与进度保存在当前浏览器。

账号／社区页面及接口约定属于公开前端代码；其服务端实现、线上数据存储、在线判题编排和生产部署工具维护在独立的私有仓库 `PaperLeet/backend`。私有仓库以 Git 子模块固定引用本仓库的版本；开源版本不会安装或下载私有服务。

按项目决定保留 Git 历史。以前的提交包含原合并版后端，从当前目录移除并不会让这些旧代码变成秘密。此后的线上后端修改只进入私有仓库。

GitHub Pages 在 https://paperleet.github.io/ 提供浏览器练习。本地服务仅绑定 `127.0.0.1`，线上应用使用独立入口。

## 开始第一题

1. 在题库中选择论文，例如 **6 · DDPM**，再打开“前向加噪”模块。
2. 阅读模块目标，在右侧编辑器中补全标注为需要实现的函数；调度器等辅助函数已经提供。
3. 点击“运行全部测试”，等待判题结果自动显示。
4. 根据各层测试的失败原因、期望值与实际值修改代码，再次运行。

完整测试全部通过后，完成状态自动更新并解锁坑点。只跑部分测试不会解锁“全部通过”状态。本地模式的草稿和学习进度保存在当前浏览器中。右上角可切换中文 / English，界面、题面和讲解随之切换。论文入口使用数字编号 1–49；模块编号 1–63 和内部目录名保持不变，兼容已有草稿和 CLI 命令。后续运行失败不会抹掉此前已通过的记录。旧草稿会保留；如需新版辅助代码，可先备份代码，再使用编辑器的重置功能。

LLM 基础新增 RMSNorm、GQA 和 RoPE，包含独立梯度测试、完整 CPU/WASM 判题与教学实验。详见[公共核心验证](docs/validation/llm-foundations-public.md)。

顶部 **浅色 / 深色** 按钮用于切换外观。首次访问跟随系统主题，手动选择后会在当前浏览器记住。题库集中展示分类、搜索、筛选和模块进度；技术简介与可展开的练习范围放在论文详情页。

拖动工作区分隔条，可调整题面与代码区的左右宽度、代码与测试结果区的上下高度，比例会保存在当前浏览器。双击分隔条，或选中后按 Enter，可恢复默认比例；方向键可微调。运行按钮始终保留在底部，窄屏上题面和编辑器自动上下排列。

顶部 **精选合集** 通过彩色卡片呈现 LLM 基础、强化学习、扩散与 Flow、后训练、JEPA / 世界模型和视觉生成，卡片显示论文数与题目数，下方小标签用于分类筛选。合集以论文为单位，可以交叉收录；独立链接例如 `#/collections/llm-basics`，原有 `#/lists/...` 链接继续可用。切换合集会清空筛选，合集成员和双语卡片文案维护在 `web/src/paperCollections.ts`。

**随机一题** 从当前合集及筛选范围内打开一道未开始的模块题，排除有运行记录、已通过记录或已编辑草稿的题目；未修改的初始代码仍可被抽中。按钮会等待学习进度加载，没有符合条件的题目时禁用并提示。

网页通过判题服务执行代码并自动显示结果，无需手动处理报告文件。本地服务监听 `127.0.0.1:8765`，Vite 代理 `/api` 请求；代码保存到 `workspace/`，由本机 Python/PyTorch 执行。这一启动方式面向个人本地使用。线上应用使用独立私有服务入口。

## 线上服务

线上账号、草稿同步、个人主页和社区讨论由独立运营的应用提供。本地版隐藏这些入口，不请求账号 API；浏览器与本机练习均无需登录。

## Submit paper

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

`paperleet ci` 检查参考解全部通过、每个 mutant 至少被一个真实失败抓到、每个 mutant 差异不超过 10 行、重复三次一致、T3 CPU 预算与总运行时间。扩展后的 63 模块目录总预算为 300 秒（可通过 `--budget` 设置），单个 T3 的时限仍由各题包声明。结果写入 `reports/ci.json`，每题完整参考报告也保存在 `reports/`。导入/语法错误不会被当作有效 mutant 检出。

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
| 47 | RMSNorm | RMSNorm、前 k 特征统计、尺度关系与跨特征梯度。 | 61 |
| 48 | GQA | KV 投影组内均值转换、连续头共享、共享梯度与缓存容量。 | 62 |
| 49 | RoFormer / RoPE | 相邻特征对旋转、相对位置内积、解析梯度与缓存位置偏移。 | 63 |

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

GitHub Pages 使用 `VITE_PUBLIC_SITE=1` 和 `VITE_APP_EDITION=local` 构建浏览器练习版，编辑与浏览器判题无需 API。普通本地构建还支持回环地址上的本机判题；线上账号同步由独立私有应用提供。

## 目录与贡献

```text
paperleet/         CLI、pytest runner、本地服务、公共数学工具
papers.yaml        论文、核心贡献、练习范围与模块顺序
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

浏览器端判题现覆盖 44 个模块，包括 Schedules、Attention、LoRA、主要 RL 目标、JEPA/世界模型、OPSD/SDPO、谱分解，以及 GLM-5 双边重要性采样、OPD、Consistency Models、Progressive Distillation、DDPM 反向采样、DDIM 和 Flow Matching。全部 335 项维护测试在本机 CPU/WASM 执行，目录共维护 180 个错误实现，每题运行前先校准对应参考解与错误实现；保留原有精度、梯度、模型大小和训练轮数。单次分类采样保持正确分布，随机序列与原生 PyTorch 不要求逐项一致；不支持的 API 或运行时故障自动回退服务端。编辑器可切换自动、仅浏览器和服务端模式。运行时按需加载；线上账号进度仍由服务端确认。[实现与兼容性说明](docs/browser-judge.md)。
