# 提示

1. 归一化 query，归一化已 detach 的 key/queue。
2. 正样本点积为第一列，后接 query 与字典的矩阵乘积；全部列都进入 logsumexp。
3. EMA 中 m 权重属于旧 key；返回 detached 独立副本。
4. 逐行写入队列 `(pointer+i)%K`，最后返回新指针；新 key 先归一化。
5. 当前 batch 的 key 在更新之前编码，并在 query SGD 与 key EMA 后入队。
