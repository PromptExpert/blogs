# Llama3的模型结构

> Llama 3 uses a standard, dense Transformer architecture. It does not deviate significantly from Llama and Llama 2 in terms of model architecture; our performance gains are primarily driven by improvements in data quality and diversity as well as by increased training scale.

和原始Transformer的几点不同：
- Pre-normalization. To improve the training stability, we normalize the input of each transformer sub-layer, instead of normalizing the output. We use the RMSNorm normalizing function.
- SwiGLU activation function. We replace the ReLU non-linearity by the SwiGLU activation function to improve the performance. 
- RoPE。
- Grouped-Query Attention (GQA)。We use grouped query attention (GQA) with 8 key-value heads to improve inference speed and to reduce the size of key-value caches during decoding.

和Llama 2的几点不同：
- We use an attention mask that prevents self-attention between different documents within the same sequence. We find that this change had limited impact during in standard pre-training, but find it to be important in continued pre-training on very long sequences.
- We use a vocabulary with 128K tokens. Our token vocabulary combines 100K tokens from the tiktoken3 tokenizer with 28K additional tokens to better support non-English languages. Compared to the Llama 2 tokenizer, our new tokenizer improves compression rates on a sample of English data from 3.17 to 3.94 characters per token. This enables the model to “read” more text for the same amount of training compute. We also found that adding 28K tokens from select non-English languages improved both compression ratios and downstream performance, with no impact on English tokenization.
- We increase the RoPE base frequency hyperparameter to 500,000. This enables us to better support longer contexts; Xiong et al. (2023) showed this value to be effective for context lengths up to 32,768.


Llama 3 405B uses an architecture with 126 layers, a token representation dimension of 16,384, and 128 attention heads; see Table 3 for details. This leads to a model size that is approximately compute-optimal according to scaling laws on our data for our training budget of $3.8 \times 10^{25}$ FLOPs.

![](../images/llama3_table3.png)

实现: https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/api/model.py

*参考资料*
- LLaMA- Open and Efficient Foundation Language Models
- Llama 2- Open Foundation and Fine-Tuned Chat Models
- The Llama 3 Herd of Models

