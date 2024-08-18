# 训练时的显存如何计算？

## 概述

对于一个1.5B参数量的GPT-2模型，在16比特精度下需要3GB的内存来存储它的权重，但它却没法在具有32GB显存的一块GPU上进行训练。在训练阶段，绝大部分显存都是模型状态(model states)消耗的，即，由优化器状态、梯度、参数构成的张量。除此之外，剩下的显存由activations、临时缓冲区和fragmented memory消耗，统称为残留状态(residual states)。

注：下文的内存指的是GPU内存，即显存。

## 模型状态

### 模型参数

模型的内存 = 参数量 $\times$ 每个参数的内存

每个参数的内存，取决于精度。一个float32精度的参数，内存是4字节；int8则是1字节。

一个6B参数量的模型，如果是float32精度，需要24GB内存；如果是int8精度，需要6GB。

### 梯度内存

因为模型中梯度的数量通常等于中间变量的数量，所以memory_activations= memory_gradients。memory_activations的计算见下文。梯度内存的计算有时候也包含在优化器中。

### 优化器内存

优化器的状态通常会占到很大一部分的内存，这在混合精度训练中尤为明显。

不同优化器所存储的参数量不同。以Adam为例，它需要存储两部分的优化器状态：time averaged momentum和variance of the gradients。因此，在使用Adam进行模型训练的时候需要有足够的的内存空间来存储动量估计和梯度方差的复制值，除此之外还需要有足够的空间存储模型本身的梯度和权重。1.5B的GPT2，模型本身只需要3GB，但加上优化器之后，则需要24GB。

## 残留状态

### Activations

训练样例和模型参数进行矩阵运算，产生activations。activations是计算并存储在传播中的中间变量，在计算梯度时需要使用这些变量。

以llama为例。llama架构为hidden_size=4096, intermediate_size=1008, num_hidden_layers=32, content_length=2048。每个instance所需内存为(4096+11008) x 32 x 2048 = 990MB。

再比如，使用序列长度为1K和batch size为32训练的1.5B参数GPT-2模型，需要大约60GB的内存。

Activation checkpointing是一种常见的方法，通过牺牲33％的重新计算开销，从而将激活内存减少约总激活数量的平方根。这将该模型的激活内存消耗降低到约8GB。尽管有了activation checkpointing，对于更大的模型，激活内存仍可能会变得非常大。例如，类似GPT的模型具有1000亿个参数，即使使用激活检查点，对于批处理大小为32，也需要大约60GB的内存。

### 临时缓冲区

用于存储中间结果的临时缓冲区会消耗大量内存。像梯度all-reduce、梯度norm计算这样的操作，会把所有梯度融合到单个扁平的缓冲区(Operations such as gradient all-reduce, or gradient norm computation tend to fuse all the gradients into a single flattened buffer)。例如，对于一个包含1.5B个参数的模型，一个扁平的的fp32缓冲区将需要6GB的内存。

### 内存碎片

到目前为止，我们已经讨论了训练期间实际的内存消耗。此外，即使有大量可用内存，也有可能耗尽可用内存。这可能是由于内存碎片造成的。如果没有足够的连续内存来满足请求，申请内存将失败，即使总可用内存大于请求的内存量。我们观察到在训练非常大的模型时存在显着的内存碎片，导致在某些极端情况下即使仍有超过30％的内存可用，也会发生内存不足的问题。

*参考资料*

- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [有哪些省内存的大语言模型训练/微调/推理方法？](https://news.sohu.com/a/664956986_121119001)
