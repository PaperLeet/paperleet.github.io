# PaperLeet

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="web/public/brand/paperleet-logo-dark.svg">
  <img src="web/public/brand/paperleet-logo-light.svg" alt="PaperLeet" width="320">
</picture>

[English](README.md) | **简体中文**

**网站源码：**[PaperLeet/paperleet.github.io](https://github.com/PaperLeet/paperleet.github.io) 包含本站前端、题库和浏览器／本机判题工具。

**独立离线版：**[Hongyang-Du/PaperLeet](https://github.com/Hongyang-Du/PaperLeet) 是重点维护与推广的开源入口，提供完整学习指南、本机 CPU 做题、本地解答保存和 MIT 许可证。网站顶部的 GitHub 图标指向此离线版仓库。欢迎在该仓库点 Star 支持项目。

读论文，写实现，用数学性质和真实行为验证理解。

当前题库包含 **56 篇论文、1 个入门专题、77 个核心模块**。先选择论文，了解核心贡献和本次练习范围，再进入模块的 LeetCode 风格分栏工作区。辅助代码直接提供，练习聚焦论文的关键机制。“论文完成”表示选定核心模块全部通过。论文组织保存在 `papers.yaml`，题面、实现、测试和可视化数据来自 `problems/`。

## 本地启动

产品名称为 **PaperLeet**，CLI 使用 `paperleet` 命令；此前的 `leetpaper` 别名继续兼容。

需要 Python 3.10+、Node.js 20.19+。在仓库根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install -e ./packages/stridetorch
paperleet export
npm --prefix web ci
make dev
```

打开终端显示的地址（默认 **http://127.0.0.1:5173**）。以后在根目录运行 **`make dev`** 即可同时启动网页与判题服务；按 Ctrl-C 会关闭两个进程。

## 仓库边界

本公开仓库包含前端、论文题库、浏览器运行时、本机判题与离线维护工具。`make dev` 无需账号、邮件服务或私有仓库即可独立运行。未登录时，练习代码和浏览器测试结果自动保存在当前浏览器，刷新或重新打开网站后可恢复。

基于 NumPy 的浏览器张量与自动求导内核已拆为可独立打包的 [StrideTorch 库](packages/stridetorch/README.md)。独立包只依赖 NumPy；现有 PaperLeet 安装会包含同一份库源码，原有接入入口继续可用。代码里保留普通的 `import torch`、`from torch import nn` 和 `import torch.nn.functional as F`，独立脚本用 `stridetorch train.py` 启动即可，无需改写导入语句；PaperLeet 的浏览器 Worker 会自动注册这些别名。

账号／社区页面及接口约定属于公开前端代码；其服务端实现、线上数据存储、在线判题编排和生产部署工具维护在独立的私有仓库 `PaperLeet/backend`。私有仓库以 Git 子模块固定引用本仓库的版本；开源版本不会安装或下载私有服务。

按项目决定保留 Git 历史。以前的提交包含原合并版后端，从当前目录移除并不会让这些旧代码变成秘密。此后的线上后端修改只进入私有仓库。

在线练习入口为 https://paperleet.com/，公开前端由 GitHub Pages 发布。本地服务仅绑定 `127.0.0.1`，线上应用使用独立入口。

## 开始第一题

入门可先做[训练与生成专题](https://paperleet.com/#/papers/llm-training-generation)：六道独立练习涵盖标签错位、SFT 掩码、采样、梯度累积与裁剪，均可直接使用浏览器 CPU/WASM 运行。

1. 在题库中选择论文，例如 **6 · DDPM**，再打开“前向加噪”模块。
2. 阅读模块目标，在右侧编辑器中补全标注为需要实现的函数；调度器等辅助函数已经提供。
3. 点击 **Run tests / 运行测试**，在浏览器中检查代码并等待结果显示。
4. 根据各层测试的失败原因、期望值与实际值修改代码，再次运行。
5. 在线上应用登录后，点击 **Submit / 提交**，由服务端判题，并将本次提交的代码与结果保存到账号。

浏览器完整测试全部通过后，本机学习进度更新并解锁坑点；账号完成记录以服务端提交通过为准。只跑部分测试不会解锁“全部通过”状态。右上角可切换中文 / English，界面、题面和讲解随之切换。论文与模块的数字编号、数字别名和内部目录名保持稳定，兼容已有草稿和 CLI 命令。后续提交失败不会抹掉此前已通过的记录。

LLM 基础新增 RMSNorm、GQA 和 RoPE，包含独立梯度测试、完整 CPU/WASM 判题与教学实验。详见[公共核心验证](docs/validation/llm-foundations-public.md)。

未登录的代码和浏览器测试结果保存在浏览器本地，与各个登录账号分开。缓存仅限同一设备、浏览器和网站域名；清除网站数据或结束无痕浏览会丢失缓存。登录不会自动导入游客草稿。登录后，账号草稿会自动同步到账号；点击 **Submit / 提交** 后，另行保存提交当时的完整源码与服务端结果，未通过的尝试也会保留。在当前题目的「我的提交」中，可查看、复制解答并加载更早的记录。后续编辑草稿不会改动已经提交的代码，提交历史仅对所属账号可见。

**Run tests / 运行测试** 只使用浏览器运行时；遇到不支持的 API 或运行失败时，不会自动向服务端提交代码。需要服务端判题并保存账号解答时，明确点击 **Submit / 提交**。已登录用户的浏览器测试结果与学习进度可以按账号保留在当前浏览器本地，不会成为正式的账号提交记录。

恢复初始代码会先要求确认，再替换当前草稿，不会自动下载备份；如需保留当前代码，请先复制或下载。账号中已经提交的解答和此前通过的记录继续保留，旧测试结果会标明对应旧代码。

代码编辑器加载失败时，普通文本模式仍可编辑和运行浏览器测试，已登录的草稿继续同步到账号。「重新加载编辑器」会等待账号草稿与本机学习记录保存完成后刷新页面；测试运行中、保存失败或存在草稿冲突时会禁用。刷新后可恢复已成功保存的游客草稿和浏览器测试结果、已同步的账号草稿，以及已保存到本地的登录用户浏览器测试结果。本地保存失败时，编辑器会显示提示；离开前可重试保存或下载代码。

语法错误和明确指向当前提交的异常会提供「定位代码第 N 行」，直接打开代码区并移动光标，普通文本模式也支持。代码已修改、版本无法核对，或错误只指向测试／参考实现时，不会提供误导的定位入口。

顶部 **浅色 / 深色** 按钮用于切换外观。首次访问跟随系统主题，手动选择后会在当前浏览器记住。题库集中展示分类、搜索、筛选和模块进度；技术简介与可展开的练习范围放在论文详情页。

拖动工作区分隔条，可调整题面与代码区的左右宽度、代码与测试结果区的上下高度，比例会保存在当前浏览器。双击分隔条，或选中后按 Enter，可恢复默认比例；方向键可微调。运行按钮始终保留在底部，窄屏上题面和编辑器自动上下排列。

顶部 **精选合集** 通过彩色卡片呈现 LLM 基础、强化学习、扩散与 Flow、后训练、JEPA / 世界模型和视觉生成，卡片显示论文数与题目数，下方小标签用于分类筛选。合集以论文为单位，可以交叉收录；独立链接例如 `#/collections/llm-basics`，原有 `#/lists/...` 链接继续可用。切换合集会清空筛选，合集成员和双语卡片文案维护在 `web/src/paperCollections.ts`。

**随机一题** 从当前合集及筛选范围内打开一道未开始的模块题，排除有运行记录、已通过记录或已编辑草稿的题目；未修改的初始代码仍可被抽中。按钮会等待学习进度加载，没有符合条件的题目时禁用并提示。

网页自动显示浏览器测试和服务端提交结果，无需手动处理报告文件。本机原生判题服务监听 `127.0.0.1:8765`，Vite 代理 `/api` 请求；本机原生运行会将代码保存到 `workspace/`，由 Python/PyTorch 执行。这一启动方式面向个人本地使用，线上应用使用独立私有服务入口。

## 线上服务

线上账号、草稿同步、个人主页和社区讨论由独立运营的应用提供。本地版隐藏这些入口，不请求账号 API；浏览器与本机练习均无需登录。

每个账号每天可提交 **1 篇论文提案**、发布 **5 篇讨论帖**、发布 **50 条回复**。三项额度分别计算，在 **UTC 00:00** 重置；删除已经发布的内容不会返还当天额度。

可以使用 AI 辅助起草和改进内容，发布前须自行审阅、核实观点与来源，并对最终内容负责。不允许自动批量发帖或回复。论文提案进入维护者审核流程，提交后不会自动生成已发布的练习题。

讨论和回复链接支持复制，分享链接不携带搜索条件或草稿。回复链接在打开和登录后定位、突出对应内容；目标回复不可用时仍可阅读完整讨论。自动复制失败时可手动复制链接。见[链接验证记录](docs/validation/discussion-links.md)。

帖子页支持原地刷新和重试读取，不清空回复草稿。服务端已确认发布的回复会立即显示，即使随后的读取失败，也可定位到新回复；重试只读取讨论，不再次提交回复。返回列表时保留题目、搜索和筛选条件。

帖子和回复在成功响应丢失后会保留待确认发布。重试相同内容不会重复创建；草稿已改动时，可明确选择发布新内容，或先重试此前发布并保留新草稿。此能力需要配套更新后的线上服务；标签页存储可用时支持同一标签页刷新恢复，关闭标签页或清理存储可能丢失待确认记录。

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

`paperleet ci` 检查参考解全部通过、每个 mutant 至少被一个真实失败抓到、每个 mutant 差异不超过 10 行、重复三次一致、T3 CPU 预算与总运行时间。CLI 默认总预算为 300 秒（可通过 `--budget` 设置）；发布 CI 对 77 模块目录使用 `paperleet ci --budget 600`，单个 T3 的时限仍由各题包声明。结果写入 `reports/ci.json`，每题完整参考报告也保存在 `reports/`。导入/语法错误不会被当作有效 mutant 检出。

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
| 50 | TIGER | 冻结残差量化、碰撞后缀、无约束 beam 解码与完整 ID 过滤。 | 64–65 |
| 51 | ColBERT | 逐 token 余弦 MaxSim、mask、成对排序与合成查询向量训练。 | 66 |
| 52 | SASRec | 因果／padding 历史可见性、逐位置负采样二元目标与依赖历史的合成训练。 | 67 |
| 53 | BPR | 用户级未观察候选、矩阵分解成对目标／全表先验与个性化合成训练。 | 68 |
| 54 | LightGCN | 对称二部图归一化、线性邻居传播、均匀层聚合与初始 embedding 真实训练。 | 69 |

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

独立静态站只需在 `web/` 运行 `VITE_PUBLIC_SITE=1 npm run build`，无需后端地址。
如需连接外部账号应用，设置 `VITE_HOSTED_APP_AVAILABLE=1` 和 HTTPS 来源
`VITE_HOSTED_APP_URL`。托管前端可用 `VITE_API_ORIGIN` 指定独立 HTTPS API 来源，
留空则使用同源请求。无效的已启用来源会在生成 JavaScript 前使构建失败；提示指出变量名，
不打印变量值。[独立部署验证记录](docs/validation/standalone-startup.md)。

## 目录与贡献

```text
paperleet/         CLI、pytest runner、本地服务、公共数学工具
papers.yaml        论文、核心贡献、练习范围与模块顺序
problems/          独立核心模块包（强化学习、微调、生成模型、架构、优化、表示学习）
web/               Vite + React + TypeScript
tests/             判题框架的集成测试
workspace/         个人代码，忽略于 Git
reports/           判题和 CI 报告，忽略于 Git
.paperleet/        本地判题队列与论文提交材料，忽略于 Git
```

出题流程见 [CONTRIBUTING.md](CONTRIBUTING.md)，详细验收依据见 [10 题试点规格](PaperLeet_第一步_10题试点规格.md)。GitHub Actions 运行框架测试、三次质量门、数据生成和前端构建，上传报告与静态站构建产物。

本次实现的实测结果与验证范围见 [试点验收记录](docs/validation/pilot.md)。

26 篇论文扩展版的数值验证、toy 来源、网页检查及 harness 验证见[验证记录](docs/validation/core-curriculum.md)。

本轮 Oral/Spotlight 扩展新增 12 篇论文、14 个模块。论文来源、toy 归属与验证见[候选证据](docs/research/paper-candidates-2026-10-13.md)和[验证记录](docs/validation/oral-spotlight-expansion.md)。

VAR 与 FD-Loss 新增四个核心模块及 CPU 小实验，详见[来源与验证记录](docs/validation/var-and-fd-loss.md)。

截至 2026-10-15，浏览器判题覆盖全部 **77 个维护模块**，包含 **608 项学习者测试**与 **336 个维护错误实现**，涵盖 Schedules、Attention、LoRA、检索与表征学习、强化学习、扩散模型，以及六道 LLM 入门题。完整测试在 CPU/WASM 执行，无需 GPU；保留原有精度、梯度、模型大小和训练轮数，例如 LoRA 的 300 步训练和 Flow Matching 的完整 2000 步训练。

**Run tests / 运行测试** 始终在浏览器执行。不支持的 API、超时和运行时故障会直接提示，不会自动把代码发送到服务端；登录后点击 **Submit / 提交** 才进行服务端判题并保存账号成绩。运行时按需加载，界面显示环境准备、参考解与错误实现校准、学习者测试进度，并可随时停止。CI 为每个导出模块启动独立的固定版本 Pyodide 进程，执行全部参考解和错误实现，并核对测试数量与代码身份；这项 WASM 检查与真实浏览器界面、设备兼容性验证分别进行。[实现与兼容性说明](docs/browser-judge.md)。
