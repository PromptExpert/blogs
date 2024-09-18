Comments:

- The o1 model series is trained with large-scale reinforcement learning to reason using chain of thought.[^1]
- o1 is a model that thinks before giving the final answer. In my own words, here are the biggest updates to the field of AI.[^2]
- Don’t do chain of thought purely via prompting, train models to do better chain of thought using RL.[^2]
- In the history of deep learning we have always tried to scale training compute, but chain of thought is a form of adaptive compute that can also be scaled at inference time. 在深度学习和人工智能的背景下，"adaptive compute"（自适应计算）指的是一种能够根据任务需求动态调整计算资源或计算策略的技术。这种方法与传统的固定计算不同，传统方法通常在训练和推理阶段使用相同的计算资源和策略，而adaptive compute则允许在推理阶段灵活调整计算量和计算路径。[^2]
- Our large-scale reinforcement learning algorithm teaches the model how to think productively using its chain of thought in a highly data-efficient training process. [^3]
- Through reinforcement learning, o1 learns to hone its chain of thought and refine the strategies it uses. It learns to recognize and correct its mistakes. It learns to break down tricky steps into simpler ones. It learns to try a different approach when the current one isn’t working. This process dramatically improves the model’s ability to reason.[^3]
- We're no longer limited by pretraining paradigm; now, we can scale through inference compute, opening up new possibilities for capabilities and alignment.[^4]
- The interesting update from Strawberry is that OpenAI has found a way to add a new dimension on which to improve performance: compute during inference. The company has found that when Strawberry takes longer to respond to a prompt—in other words, when it’s given more time to think—it generally responds more accurately.[^5]



Reading List:

- [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314)
- https://github.com/hijkzzz/Awesome-LLM-Strawberry
- [A Survey on Self-play Methods in Reinforcement Learning](https://arxiv.org/abs/2408.01072) 



References:

[^1]: OpenAI o1 System Card </br>
[^2]: Jason Wei </br> 
[^3]: https://openai.com/index/learning-to-reason-with-llms/ </br>
[^4]: Mira Murati </br>
[^5]: https://every.to/chain-of-thought/openai-s-o1-model-explained