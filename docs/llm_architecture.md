# 各种大模型架构，傻傻分不清楚？

各种（基于Transformer的）大模型架构，什么encoder-decoder, encoder-only, decoder-only, prefix-decoder之类的，到底是什么意思？有什么区别？

其实没那么多概念，基本的思想是很简单的，本文解释几个最基本的概念，其他所有术语都是这些概念的变形和组合。

## 基础概念

**Auto-regressive** 训练语言模型时，当前的预测（或当前的编码），依赖且仅依赖于过去的信息。

**Causal** 几乎和auto-regressive同义。

**Encoder** 将文本（或其他模态的raw data）转换为向量的模块。

**Decoder** 将向量转换为文本的模块。

## 模型结构

**Encoder-Decoder** 原始Transforer结构。模型一部分参数只做编码，另一部分参数只做解码。比如，T5。

**Causal Decoder-only** 模型只有原始Transformer的decoder部分。语言建模时，仅依赖过去的序列，预测下一个token。比如，GPT2。

**Non-causal decoder-only** 模型只有原始Transformer的decoder部分。语言建模时，不是仅依赖过去的序列。也叫prefix language model，简称prefixLM。也叫prefix decoder。比如，GLM。

**Encoder-only** 模型只有原始Transformer的encoder部分，也叫autoencoding。比如，BERT。

## 预训练目标函数

模型结构不是孤立存在，要和目标函数共同完成模型建模的任务。

**Full language modeling** 从第一个token开始，一个一个地进行auto-regressive语言建模。

**Prefix language modeling** 允许先将文本的一部分定义为prefix，attention可以关注到全部prefix，也就是说在prefix内部是non-causal的。在prefix之外，基于之前的token，一个一个地预测之后的token。Encoder-decoder和Non-causal decoder-only的训练目标都是prefix language modeling。

**Masked language modeling** tokens或spans of tokens被[mask]替换，然后预测这替换的token，典型的如BERT。

## 大佬发言
这里列举一些大佬的观点，非常精辟，一针见血，有助于加深理解。

> What's called "encoder only" actually has an encoder and a decoder (just not an auto-regressive decoder). 
-- Yann LeCun

解释：以BERT为例，在将窗口转换为特征的时候，模型是encoder。在基于窗口向量，预测mask的时候，模型是decoder。因为预测的时候同时用了过去和将来的信息，所以不是auto-regressive 。

---

> What's called "encoder-decoder" really means "encoder with auto-regressive decoder"
-- Yann LeCun

解释：以原始Transformer为例，decoder解码的时候只依赖于原文和已经翻译了的译文，所以是auto-regressive decoder。 

---

> What's called "decoder only" really means "auto-regressive encoder-decoder"
-- Yann LeCun

解释：以GPT2为例，模型既是编码器，也是解码器，不论编码还是解码，都是auto-regressive。这句话非常精辟，要仔细品味。

---

> EncDec, PrefixLMs, Causal Dec-onlys are all autoregressive. 
-- Yi Tay 

解释：在生成阶段，都是autoregressive。

---

> All 3 archs are not that different. People somehow imagine that EncDec to be "not good at generation or chat". Not true. It's the objective function that matters.
-- Yi Tay 

解释：模型结构是表象，是外形，目标函数才决定了模型的本质。

---

> PrefixLM are causal decoders with non-causal input (but causal targets).
-- Yi Tay 

解释：refixLM在non-causel input(即prefix)的基础上，进行causal语言建模。

---

> Encoder-Decoders are prefixLMs with non-shared weights that connects two (enc/dec) stacks with cross attention. 
-- Yi Tay 

解释：在encoder-decoder中，prefix就是encoder所编码的文本，encoder和decode是两套参数，decoder解码的时候用了cross attension。

---

> Everything is technically seq2seq. Just whether it has a mask and whether the 'inputs' is empty.
-- Yi Tay 

解释：所有架构都联系到一起了，最高层面的一般化。



*参考资料*
- [https://deepgenerativemodels.github.io/notes/autoregressive/](https://deepgenerativemodels.github.io/notes/autoregressive/)
- [https://huggingface.co/docs/transformers/tasks/language_modeling](https://huggingface.co/docs/transformers/tasks/language_modeling)
- [UL2: Unifying Language Learning Paradigms](https://arxiv.org/pdf/2205.05131.pdf)
- [What Language Model Architecture and Pretraining
    Objective Work Best for Zero-Shot Generalization?](https://arxiv.org/pdf/2204.05832.pdf)
- A Survey of Large Language Models
- [https://twitter.com/ylecun/status/1651762787373428736](https://twitter.com/ylecun/status/1651762787373428736)
- [https://twitter.com/YiTayML/status/1651927473884655616](https://twitter.com/YiTayML/status/1651927473884655616)