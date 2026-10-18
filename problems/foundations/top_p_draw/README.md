# Top-p：从候选概率到抽取 token

## 问题

实现 `sample_top_p(logits, p, temperature, generator)`，为每行返回一个词表 token 编号。提供的 `nucleus_probs` 已处理温度、稳定 softmax、核候选筛选和重新归一化；你来完成从 logits 到实际抽样结果的最后一步。

依据 [Holtzman 等，ICLR 2020，§3.1 公式 (2–3)](https://arxiv.org/abs/1904.09751)。论文定义核集合与重归一化分布；下面的逆累计分布算法和精确随机数流是本题教学约定，不声称论文指定这种实现。

## 核心公式

对原词表顺序的概率 q，计算 C_i = sum_{j<=i} q_j，再把所有 C_i 除以最后一项。抽取 u∈[0,1)，返回满足 C_i>u 的最小词表编号 i。每个 token 对应半开区间 [C_{i-1}, C_i)，零概率区间不能被选中。

## 公式到代码

调用提供的函数，沿最后一维累计求和，用累计和最后一项归一化，再为每行生成一个独立均匀随机数。CDF 中 <=u 的元素个数就是词表编号，不能返回按概率排序后的名次。

例：q=[0,.5,0,.5]，u=0 选 token 1，u=.5 选 token 3。提供的 helper 保留累计概率首次达到或超过 p 的最短非空前缀；并列概率优先较小词表编号。四项均匀 logits、p=.5 时只留下编号 0 和 1。

## 本题约定

- logits 为 CPU 上有限 float32/64 张量 `[...,V]`，V≥1，允许空前导维；p 有限且 0<p≤1，temperature 有限且>0。
- 输出为 CPU int64，shape 为 `logits.shape[:-1]`；单向量输入返回标量 tensor。不修改输入，不要求梯度。
- starter 已提供 `nucleus_probs`，返回原词表顺序的 float64 概率；保持该 helper 不变。
- 恰好调用一次 `torch.rand(logits.shape[:-1], dtype=torch.float64, generator=generator)`。使用调用方提供的 CPU Generator，不重设种子，不使用全局生成器，不让多行共享一个随机数。空 batch 不消耗随机数。
- float64 CDF 须除以最后一项，使终点精确为 1；选择严格满足 C_i>u 的位置，精确落在边界上也遵循此规则。
- 精确随机数流是教学约定；其他正确的分类抽样方法可能消耗不同的随机数。本题练习逆 CDF。
- EOS、多步生成、NaN/Inf logits、temperature=0 不在单 token 题目范围内。

## 交互图

<viz id="toy_experiment" />

你的函数以种子 42 从合成 logits 实际抽取 8192 个 token。图比较抽样频率与独立目标 [0,2/3,0,1/3]，被排除 token 必须从不出现。频率最大误差要求小于 .04；确定性边界测试使用精确判定。这里只检查合成抽样，不代表文本生成质量。

## 运行方式

```sh
paperleet test foundations/top_p_draw --impl problems/foundations/top_p_draw/solution.py
```

在工作区补全 starter。T0 检查类型与形状，T1 对照独立标量基准，T2 检查随机数流、边界和输入保护，T3 使用你的函数真实抽样。CPU/WASM 浏览器执行相同测试与 mutants。

## 延伸

结合已有的 #89 Top-p 候选概率题，解释为何找候选时需要排序，而抽样结果必须回到原词表编号。再想一想：把单步抽样接入生成循环后，怎样分别停止已遇到 EOS 的 batch 行？
