# Llama3 Training Recipe

> The recipe used to pre-train Llama 3 405B consists of three main stages: (1) initial pre-training, (2) long-context pre-training, and (3) annealing. 

## Initial Pre-Training
> We pre-train Llama 3 405B using a cosine learning rate schedule, with a peak learning rate of $8 \times 10^{-5} , a linear warm up of 8,000 steps, and a decay to $8 \times 10^{-7}$ over 1,200,000 training steps. We use a lower batch size early in training to improve training stability, and increase it subsequently to improve efficiency. We found this training recipe to be very stable: we observed few loss spikes and did not require interventions to correct for model training divergence.

小的batch size在训练初期能够引入更多的梯度噪声，从而起到类似正则化的效果，帮助模型跳出局部最优解，增强训练稳定性和泛化能力，同时由于参数更新更频繁，有助于更快速地找到较好的优化路径。

## Long Context Pre-Training
详见 [Llama3如何实现128K的上下文长度](llm_foundation/llama3/llama3_128k.md)

## Annealing
> During pre-training on the final 40M tokens, we linearly annealed the learning rate to 0, maintaining a context length of 128K tokens. During this annealing phase, we also adjusted the data mix to upsample data sources of very high quality; Finally, we compute the average of model checkpoints during annealing to produce the final pre-trained model.