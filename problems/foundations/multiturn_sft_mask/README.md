# 多轮 SFT：回答标签与会话边界

## 1. 问题

一行中可能有多轮 assistant 回答，也可能拼接了不同会话。根据已经提供的 token 元数据，找出应该学习的目标位置。[TRL 的 assistant-only loss](https://huggingface.co/docs/trl/sft_trainer#train-on-assistant-messages-only) 与[标签错位说明](https://huggingface.co/docs/trl/sft_trainer#label-shifting-and-masking)是依据。本题不要求实现 tokenizer 或聊天模板。

实现：

```python
assistant_target_mask(role_ids: Tensor, conversation_ids: Tensor,
                      attention_mask: Tensor) -> Tensor
```

返回与输入同设备的 **bool Tensor `[B,T]`，标记目标 token 位置**。不是 attention 矩阵、错位后的掩码、token 编号或 loss。保留两个维度，不修改输入；整数与布尔元数据无需梯度。

## 2. 核心公式

用 `r`、`c`、`a` 表示角色、会话编号和有效性：

$$m_{b,0}=\mathrm{False},\qquad
m_{b,j}=[r_{b,j}=2]\land a_{b,j}\land a_{b,j-1}\land[c_{b,j}=c_{b,j-1}],\quad j\ge1.$$

前一个位置必须有效且属于同一会话，但**不要求前一个角色也是 assistant**：最后一个 user/上下文位置可以预测回答的首 token。输出不做错位；后续因果损失将 `logits[:, :-1]`、`tokens[:, 1:]` 和 `mask[:, 1:]` 对齐一次。

## 3. 从公式到代码

三个输入形状相同，均为 `[B,T]`。比较目标位置与紧邻前驱，组合角色、有效性与会话条件，并让第零列保持 false。B 或 T 可以为零；长度为一的维度不被压缩。

一行均有效的例子：

```text
角色：       [1, 2, 2, 1, 2, 2, 2, 2]
会话编号：   [7, 7, 7, 7, 7, 7, 9, 9]
目标掩码：   [F, T, T, F, T, T, F, T]
```

会话 7 的两轮回答都参与学习。位置 6 开始另一段 packed 会话，不能由位置 5 预测，即使两边都是 assistant token。

## 4. 本题约定

- CPU int64 的 `role_ids`、`conversation_ids`，bool 的 `attention_mask`，形状一致且 `B,T>=0`。输入保证符合约定，不考查输入校验；支持非连续视图。
- 有效位置的角色为 **0=system/角色头**、**1=user**、**2=assistant 内容**、**3=tool**。预处理将角色头和 BOS 标为 0。有效会话编号是非负 int64，不必连续，不同 batch 行可以复用编号。同一会话的所有轮次共享编号；相邻拼接的不同会话必须使用不同编号。
- `attention_mask=True` 表示真实保留的 token。padding 可以位于行首、行尾或中间，其角色/会话元数据可为任意 int64，不应影响输出。紧接 padding 的真实 token 没有有效的紧邻预测位置，因此不参与学习。
- **EOS 归属已提供**，不根据 token 编号推断。assistant 回答末尾的 EOS 标为角色 2，按同一规则参与；user/tool 的 EOS 保持相应角色。EOS 与 padding 即使使用同一 token 编号，也由有效性掩码区分。assistant EOS 不结束整段会话，也不会屏蔽后续轮次的回答。
- **截断由调用方先完成**。左截断后第零列没有保留的前驱，即使落在回答中间也不监督；后续可见回答 token 仍能参与。右截断保留末尾可见的合法目标，不补 EOS，也不因 EOS 被截掉而舍弃整段剩余回答。
- 角色编号、EOS 归属、会话编号布局与精确边界公式是 PaperLeet 明确的教学约定。[TRL 的 packing/截断文档](https://huggingface.co/docs/trl/reducing_memory_usage#packing)解释了保留序列边界的背景；本题不复现所有 TRL packing 策略。

这是**损失目标掩码**，不能阻止 attention 跨会话读取上下文。真正的 packed 模型还需要独立处理 attention 与位置编号；本题不实现这些机制，也不实现聊天模板解析、tokenizer 或完整语言模型训练。

## 5. 交互实验

<viz id="toy_experiment" />

配套实验使用两行原创的 12-token 数据、四类词表和 11 个手工标注的 assistant 目标，包含多轮回答、assistant EOS、packed 边界、左截断片段及 padding 间隙。你的掩码控制各位置独立 logits 的 80 次 SGD 更新。

图表展示真实训练 NLL、独立目标上的 NLL，以及一行目标掩码。正确选择应学会 11 个目标，同时让被排除的预测位置 logits 完全不变。初始四类均匀 NLL 为 `log(4)`；最终目标 NLL 须小于 `.08`。这是确定性的合成优化实验，不代表留出集语言质量或预训练模型指标。Lab 参考预览使用参考实现，提交结果调用你的函数。

## 6. 运行与验证

```bash
paperleet test foundations/multiturn_sft_mask --impl problems/foundations/multiturn_sft_mask/solution.py
paperleet mutate foundations/multiturn_sft_mask
paperleet ci --problem foundations/multiturn_sft_mask
```

T0 检查形状与空输入；T1 检查手工标注的多轮案例；T2 覆盖穷举局部真值表、截断、padding 元数据、非连续视图、会话编号/行独立性及下一词预测的解析梯度；T3 使用提交掩码训练。原生 Torch 与 CPU/WASM 使用相同测试、80 次更新和容差。

## 7. 延伸

把输出作为已有 `foundations/sft_loss_mask` 的 response mask，并传入原始有效性掩码；该损失内部只错位一次。进一步分别解释 tokenizer 如何标注角色、因果损失如何对齐标签、packed attention 如何隔离会话，避免把这三件事混为一谈。
