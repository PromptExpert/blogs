# Llama 3 Post-Training

Post-Training是Llama 3训练的重头戏，决定了模型的上限。

> We produce the aligned Llama 3 models by applying several rounds of post-training, or aligning the model with human feedback on top of a pre-trained checkpoint. Each round of post-training involves supervised finetuning (SFT) followed by Direct Preference Optimization (DPO) on examples collected either via human annotations or generated synthetically.

> The backbone of our post-training strategy is a reward model and a language model. We first train a reward model on top of the pre-trained checkpoint using human-annotated preference data. We then finetune pre-trained checkpoints with SFT, and further align the checkpoints with DPO.

## Reward Modeling
The training objective is the same as Llama 2 except that we remove the margin term in the loss, as we observe diminishing improvements after data scaling. 

The reward model takes a model response and its corresponding prompt (including contexts from previous turns) as inputs and outputs a scalar score to indicate the quality (e.g., helpfulness and safety) of the model generation. 

有用和安全需要权衡，于是训练了两个RM，一个是Helpfulness RM， 一个是Safety RM。

RM初始化为预训练后的模型。模型的结构和超参数都和预训练模型相同，except that the classification head for next-token prediction is replaced with a regression head for outputting a scalar reward.

To train the reward model, we convert our collected pairwise human preference data into a binary ranking label format (i.e., chosen & rejected) and enforce the chosen response to have a higher score than its counterpart. We used a binary ranking loss:

$$
L_{\text{ranking}} = -\log(\sigma(r_{\theta}(x, y_c) - r_{\theta}(x, y_r)))
$$

where $r_{\theta}(x, y)$ is the scalar score output for prompt $x$ and completion $y$ with model weights $\theta$. $y_c$ is the preferred response that annotators choose and $y_r$ is the rejected counterpart.

In addition to standard preference pair of (chosen, rejected) response, annotations also create a third “edited response” for some prompts, where the chosen response from the pair is further edited for improvement. Hence, each preference ranking sample has two or three responses with clear ranking (edited > chosen > rejected). We concatenate the prompt and multiple responses into a single row during training with responses randomly shuffled. This is an approximation to the standard scenario of putting the responses in separate rows and computing the scores, but in our ablations, this approach improves training efficiency without a loss in accuracy.

数据构造略。

原文这里有地方没说清楚。前面说RM takes a model response and its corresponding prompt (including contexts from previous turns) as inputs and outputs a scalar score， 后面又说 We concatenate the prompt and multiple responses into a single row during training with responses randomly shuffled，那么优化目标是什么呢？还是上面那个公式吗？还有，edited response是怎么利用的？

## Supervised Finetuning

Our preference data annotation process is similar to Llama 2.

