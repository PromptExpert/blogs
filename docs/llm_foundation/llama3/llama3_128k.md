# Llama3如何实现128K的上下文长度

> We use a lower batch size early in training to improve training stability, and increase it subsequently to improve efficiency. Specifically, we use an initial batch size of 4M tokens and sequences of length 4,096, and double these values to a batch size of 8M sequences of 8,192 tokens after pre-training 252M tokens. We double the batch size again to 16M after pre-training on 2.87T tokens.

> In the final stages of pre-training, we train on long sequences to support context windows of up to 128K tokens. We increase the supported context length in increments, pre-training until the model has successfully adapted to the increased context length. We assess successful adaptation by measuring whether (1) model performance on short-context evaluations has recovered completely and (2) the model perfectly solves “needle in a haystack” tasks up to that length. In Llama 3 405B pre-training, we increased context length gradually in six stages, starting from the original 8K context window and ending in the final 128K context window. This long-context pre-training stage was performed using approximately 800B training tokens.

总结：
1. 在4M tokens的batch size上训练4K长度。
2. 252M tokens之后，batch size增加到8M，长度增加到8K。
3. 逐渐提高上下文长度，从8K到128K，共经历6个提高阶段，整个过程持续约800B tokens。
4. 在这个扩展上下文的过程中，评测：
    - 在短上下文评估中的表现是否完全恢复。
    - 模型是否能够完美地解决长度范围内的“needle in a haystack”任务。“Needle in a haystack”任务通常涉及在大量无关信息中找到特定且稀有的目标信息，就像在一堆干草中找到一根针。
