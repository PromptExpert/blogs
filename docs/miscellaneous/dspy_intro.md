# 用最短的篇幅说清楚，DSPy是干吗的

你手写过决策树吗？我写过。那是一个由无数个if else组成的程序，用来检测一张图片是否含有某种模式。

然而，决策树不必手写的。你只要设定一个优化目标，给定一些数据，训练模型就好了。

我们现在做的很多prompt工程，和手写决策树有相似之处：想一个prompt模板或pipeline，调prompt，分析case，再调prompt。这个工作，可不可以也通过“训练”完成呢？

答案是可以，dspy就是做这件事的，它是一个自动调试prompt pipeline的框架。dspy类比于xgboost, pytorch，他们做的都是同样的事：给定一个任务，一个指标，一些训练集，然后训练出想要的模型。

和pytorch不同的是，dspy优化的“参数”，不是神经网络，而是prompt和prompt pipeline。


