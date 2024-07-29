# FLOPs

## 概念
FLOPs（Floating Point Operations）既可以指“每秒浮点运算次数”，也可以单纯指一个计算过程中的“浮点运算次数”。在神经网络中，通常讨论的是后者，即计算一个前向传播或训练过程所需的总浮点运算次数。

计算FLOPs的主要目的是评估模型的计算复杂度和资源需求。这对于以下几个方面非常重要：
- 模型设计：在设计和选择模型时，通过比较FLOPs，可以选择在性能和效率之间达到最佳平衡的模型。
- 硬件适配：了解模型的FLOPs有助于选择合适的硬件平台。例如，高FLOPs的模型通常需要更强大的计算资源，如GPU或TPU。
- 优化和压缩：为了在资源受限的设备上运行模型（如移动设备），需要对模型进行优化和压缩。通过分析FLOPs，可以确定哪些部分可以优化以减少计算量。
- 比较和基准：FLOPs作为一个标准化的指标，可以用于比较不同模型的计算需求，帮助研究者和工程师进行更有依据的选择和调整。

每一层神经网络，如卷积层、全连接层等，都需要执行大量的浮点运算。通过计算每一层的FLOPs，可以得出整个网络的总FLOPs。

## 例子
以多头注意力层为例，计算FLOPs。

1. 记：
- 输入序列长度：$L$
- 输入特征维度：$d_{model}$
- 注意力头数：$h$
- 每个头的维度：$d_k = d_v = d_{model} / h$

已知两个形状为 $(a, b)$ 和 $(b, c)$ 的矩阵相乘所需的FLOPs数为 $abc$ 。

2. 线性变换
由于Q、K和V的计算是相同的，因此我们只需要计算一次，然后乘以3：
- 每个头的Q/K/V: $L \times d_{model} \times d_{model}$
- 总FLOPs: $ 3 \times L \times d_{model}^2$

3. 计算注意力权重

计算Q和K的点积，然后进行缩放和Softmax操作：

- 点积：$L \times d_{model} \times L$
- Softmax：要计算$L$次，每次$L$个操作，所以总数 $L \times L$
- 总FLOPs: $L^2 \times d_{model} + L^2$

4. 加权和
- 注意力权重矩阵乘以V：$L \times L \times d_{model}$

5. 线性变换
- $L \times d_{model} \times d_{model}$


6. 总FLOPs计算

将上述各部分的FLOPs相加，得到多头注意力层的总FLOPs：

$$
4 \times L \times d_{model}^2 + 2 \times L^2 \times d_{model} + L^2 
$$