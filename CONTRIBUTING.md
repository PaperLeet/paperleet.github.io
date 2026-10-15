# 贡献题目

先确定论文的核心贡献，再选择可以独立实现和验证的模块。一般每篇选取 1–3 个模块，不设固定数量；完整网络与训练流程可以留作拓展。新目录中的 `problem.yaml` 会被自动发现。维护者在根目录 `papers.yaml` 中定义论文的数字编号、贡献、练习范围和模块顺序；已审核的社区模块则按 arXiv 标识或来源 URL 自动归入论文。

## 1. 定义约定

为论文添加 arXiv 链接、作者、会议及核验过的公式位置。明确 tensor shape、时间方向、mask、归一化、梯度归属和端点行为。`id` 必须与 `problems/` 下的两级目录对应；`number` 是正整数题号，必须唯一且发布后不可重新编号；新题取当前最大题号加一。`short_id` 仅用于兼容旧的 CLI 别名。

论文编号与模块编号各自独立：首页展示论文编号，CLI 仍接收模块编号。维护模块声明 `paper_id`，对应 `papers.yaml` 的稳定 slug；同一论文的所有模块应引用相同的主要来源。`module.required_functions` 与 `module.provided_functions` 必须无重叠地覆盖 `functions`；每个必做模块至少有一个需要实现的函数。`goal` / `goal_en`、`scope` / `scope_en` 同步描述目标与范围，`optional: false` 表示计入论文核心模块进度。

新增社区模块会按来源归入已有论文；若未列入该论文 `papers.yaml` 的精选列表，默认作为可选拓展，不改变原有必做范围。要扩展必做核心模块，需明确更新精选列表。

```yaml
id: diffusion/example
number: 11
title_en: A paper-to-code exercise
title: A paper-to-code exercise
difficulty: M
type: implement
parent: null
paper:
  title: Original paper title
  authors: Authors
  venue: Conference and year
  arxiv: '0000.00000'
  anchors: ['Verified equation or section']
functions:
  - name: example
    signature: 'example(x: Tensor) -> Tensor'
conventions:
  layout: '[B, D]'
tests: {T0: 1, T1: 1, T2: 3, T3: 0}
t3_budget_s: 10
estimated_minutes: 20
viz: [example_visualization]
mutants:
  - id: m1_sign
    bug: 'Reverses the sign'
```

## 2. 写完整题目包

目录包含 `README.md`、`README.en.md`、`starter.py`、`solution.py`、`notes.md`、`notes.en.md`、`tests/test_*.py`、`mutants/*.py`、`viz/gen.py`。题面按规格写七节：问题、核心公式、公式到代码、本题约定、交互图、运行方式和延伸。starter 对必做函数保留类型签名、shape、契约和 `raise NotImplementedError`；对辅助函数给出实现。辅助代码调用当前文件中的待实现函数，不能导入参考解来绕过用户实现。测试应验证只补全必做函数就能通过，并确认错误核心实现会被抓到。

每个 mutant 是可独立加载的完整文件，首行说明唯一 bug，与 `solution.py` 的增删行总数不超过 10。合法的不同实现约定也可用作 mutant，但须在坑点中解释它仅不符合当前题目的契约。

## 3. 用性质检验代码

测试用 `@pytest.mark.t0` 到 `t3` 分层，每个测试 docstring 第一行说明性质的依据。通过 `impl` fixture 访问被测模块、`ref` fixture 访问参考解，禁止直接 import solution。参考数值对齐以外，还要有独立推导、不变量、边界、梯度和行为测试。

```python
@pytest.mark.t2
def test_identity(impl, artifact):
    """The declared identity must preserve every coordinate."""
    x = torch.randn(8, 2, dtype=torch.float64)
    got = impl.example(x)
    artifact('expected', x)
    artifact('got', got)
    assert_close_located(got, x, rtol=1e-8, atol=1e-10)
```

公共 fixture 固定 Python、NumPy、PyTorch 种子为 42，并限制 PyTorch 为两条线程。T0–T2 单测试硬超时 5 秒，T3 硬超时 15 秒、质量门预算 10 秒。使用 `artifact(name, value)` 保存数组、字典或 tensor，点云最多保留 2000 点。失败信息应包含位置、期望与实际值。

T3 必须使用传入的被测函数进行训练或采样；不能只验证公共工具或生成一条预设“训练曲线”。统计阈值先测参考解，再按规格标定，记录数据量、种子、模型、优化器、基线、阈值和测量环境。分布/公式约定改变时重新标定并检查 mutants。

## 4. 生成网页数据并运行质量门

**浏览器执行是所有公开题目的默认要求。** 新题必须在发布前适配 CPU/WASM 浏览器运行时，导出完整浏览器包，并使用原始 T0–T3 测试验证参考解通过、全部 mutants 被真实断言捕获。不能以“服务端可运行”代替浏览器适配，也不能删减测试或训练流程来制造通过结果。按 [浏览器判题接入说明](docs/browser-judge.md) 更新运行时注册与验证用例，再执行前后端质量检查。

`paperleet export` 和 `paperleet vizdata` 会拒绝发布尚未注册浏览器支持的题目；CI 与 Pages 发布流程还会检查题库、浏览器包注册和参考解／mutation 验证列表是否完整对应。候选题可继续在隔离环境开发和验证，完成浏览器适配后再进入公开题库。

`viz/gen.py` 提供 `generate() -> dict`，返回 JSON 可序列化数据。不得在 import 时启动训练。预计算轨迹应采样、降帧和压缩精度，输出单题不超过 500 KB。概念图在前端实时计算；报告图使用做题者实际 artifacts。

```bash
paperleet test diffusion/example --impl problems/diffusion/example/solution.py
paperleet mutate diffusion/example
paperleet ci --problem diffusion/example
paperleet vizdata diffusion/example
python -m pytest tests -q
cd web
npm run check
npm run build
```

最终运行完整 `paperleet ci`。所有参考解必须通过、全部 mutants 必须被真实断言捕获、三次重复的结论必须一致。CI 总预算为 300 秒（与 CLI 默认值一致）；单题调试可临时使用 `--repeats 1`，不能把该结果称为三次稳定性验收。

## 5. 盲测与评审

给新读者或独立 agent 只提供 starter 与题面，记录实现用时、约定不清处、所用可视化和遗漏的错误。评审不仅看覆盖率，还要确认测试对应论文主张、失败可定位，以及图与公式一致。不要提交个人 workspace、运行环境或大量临时报告。

## 通过网页提交论文

网页的 **Submit paper** 会把论文和官方代码来源送入候选题目流程。模型生成题面、参考实现和独立测试后，必须先通过隔离验证，再由人工完成论文依据、实现约定与测试质量评审，才能进入公开题库。候选材料保存在 `.paperleet/submissions/`，不会直接被题库扫描。详细设计与配置见 [Paper submission harness](docs/paper-submission-harness.md)。

## 维护面试自测提纲

`web/src/data/preparation-guides.json` 是编辑整理的学习内容，覆盖当前准备路径的知识点。每个知识点提供双语的「讲清原理、推到实现、检查边界」问题和可展开提示，依据对应 `README.md` 的实现约定与教学范围；`source` 和 `functions` 用于追溯题面和必做函数。

更新题目契约或准备路径时，同步检查提纲。不要把公司原题、面经频次或完整论文复现结论混入学习建议，也不要复制完整解答或把自测操作当作成绩、勋章的依据。提纲可在做题前使用；完整坑点仍遵守工作区已有的通过后解锁规则。

在 `web/` 目录运行 `node scripts/check-preparation-guides.mjs`，检查题目覆盖、双语内容、来源与函数契约，以及按需加载、快速切题、超时恢复和编辑器状态保留。完整的 `npm run check` 也包含这些检查。
