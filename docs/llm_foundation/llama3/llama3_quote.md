# Llama 3 金句摘抄
> The development of modern foundation models consists of two main stages: (1) a pre-training stage in which the model is trained at massive scale using straightforward tasks such as next-word prediction or captioning and (2) a post-training stage in which the model is tuned to follow instructions, align with human preferences, and improve specific capabilities (for example, coding and reasoning).

> We believe there are three key levers in the development of high-quality foundation models: data, scale, and managing complexity.

> Language model pre-training involves: (1) the curation and filtering of a large-scale training corpus, (2) the development of a model architecture and corresponding scaling laws for determining model size, (3) the development of techniques for efficient pre-training at large scale, and (4) the development of a pre-training recipe.

> To obtain a high-quality language model, it is essential to carefully determine the proportion of different data sources in the pre-training data mix. Our main tools in determining this data mix are knowledge classification and scaling law experiments.

> Our performance gains are primarily driven by improvements in data quality and diversity as well as by increased training scale.

> In addition to determining the optimal model size, a major challenge is to forecast the flagship model’s performance on downstream benchmark tasks, due to a couple of issues: (1) Existing scaling laws typically predict only next-token prediction loss rather than specific benchmark performance. (2) Scaling laws can be noisy and unreliable because they are developed based on pre-training runs conducted with small compute budgets.

> The recipe used to pre-train Llama 3 405B consists of three main stages: (1) initial pre-training, (2) long-context pre-training, and (3) annealing.

> We produce the aligned Llama 3 models by applying several rounds of post-training,6 or aligning the model with human feedback on top of a pre-trained checkpoint. Each round of post-training involves supervised finetuning (SFT) followed by Direct Preference Optimization (DPO) on examples collected either via human annotations or generated synthetically.

> We define reasoning as the ability to perform multi-step computations and arrive at the correct final answer.

> Steerability is the ability to direct the model’s actions and outcomes to meet developer and user specifications.

> In many ways, the development of high-quality foundation models is still in its infancy. Our experience in developing Llama 3 suggests that substantial further improvements of these models are on the horizon.

> Throughout the development of the Llama 3 model family, we found that a strong focus on high-quality data, scale, and simplicity consistently yielded the best results. In preliminary experiments, we explored more complex model architectures and training recipes but did not find the benefits of such approaches to outweigh the additional complexity they introduce in model development.

> Developing a flagship foundation model such as Llama 3 involves overcoming a plethora of deep technical problems but also requires clever organizational decisions.

> We believe that the public release of foundation models plays a key role in the responsible development of such models, and we hope that the release of Llama 3 encourages the industry to embrace the open, responsible development of AGI.