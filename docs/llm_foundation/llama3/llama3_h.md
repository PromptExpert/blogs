# Llama 3 Post-Training

Post-Training是Llama 3训练的重头戏，决定了模型的上限。

> We produce the aligned Llama 3 models by applying several rounds of post-training, or aligning the model with human feedback on top of a pre-trained checkpoint. Each round of post-training involves supervised finetuning (SFT) followed by Direct Preference Optimization (DPO) on examples collected either via human annotations or generated synthetically.

## Modeling
> The backbone of our post-training strategy is a reward model and a language model. We first train a reward model on top of the pre-trained checkpoint using human-annotated preference data. We then finetune pre-trained checkpoints with SFT, and further align the checkpoints with DPO.

### Reward Modeling
