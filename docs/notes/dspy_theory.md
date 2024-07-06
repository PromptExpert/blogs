# DSPy理论详解

这两篇论文有很多计算机科学术语，而且同一个概念可能在不同的地方用不同的术语表达，容易引起困惑。

[官方github](https://github.com/stanfordnlp/dspy)给出的定义是：
> DSPy: The framework for programming—not prompting—foundation models

随着下面的分析，我们会逐渐理解这个定义。

> DSPy pushes building new LM pipelines away from manipulating free-form strings and closer to programming (composing modular operators to build text transformation graphs) where a compiler automatically generates optimized LM invocation strategies and prompts from a program. 

这里programming的含义不是大家常规里的那样（一连串的指令，执行条件分支、循环和表达式计算），而是计算机科学里更一般化的定义（其实两者是等价的，都是图灵完备）。这里的定义和lambda演算类似，lambda演算是一种抽象的计算模型，它使用函数来表达计算，lambda演算中有三个基本概念：变量、抽象（表示函数的定义）、应用（表示函数的调用）。这里modular operators就是函数，text是变量，transformation graphs是一连串的函数。

---


> We first translate string-based prompting techniques, including complex and task-dependent ones like Chain of Thoughtand ReAct, into declarative modules that carry natural-language typed signatures. 

declarative modules可以理解为所定义的函数、所声明的函数，就好像Python中的`def foo(x,y)`。
typed signatures是函数的输入类型和返回类型，例如，在Haskell中，`add :: Int -> Int -> Int`就是一个签名，它的意思是，add函数输入两个Int，返回一个Int。

---

> DSPy modules are task-adaptive components—akin to neural network layers—that abstract any particular text transformation, like answering a question or summarizing a paper. 

modules, transformation, modular operators, 说的都是一回事，都是函数，输入若干个变量，返回若干个变量。


> Inspired directly by PyTorch abstractions, DSPy modules are used via expressive define-by-run computational graphs. 

“Define-by-run” 是指在运行时动态定义计算图的编程范式，也就是动态图。在computational graph中，节点是module，边是有向的，表示文本传递的方向。


> Pipelines are expressed by (1) declaring the modules needed and (2) using these modules in any logical control flow (e.g., if statements, for loops, exceptions, etc.) to logically connect the modules.

对Pipelines做出来定义，跟上面的computational graphs是一个意思，


> We then develop the DSPy compiler, which optimizes any DSPy program to improve quality or cost. The compiler inputs are the program, a few training inputs with optional labels, and a validation metric. The compiler simulates versions of the program on the inputs and bootstraps example traces of each module for self-improvement, using them to construct effective few-shot prompts or finetuning small LMs for steps of the pipeline. 

> Optimization in DSPy is highly modular: it is conducted by teleprompters, which are general-purpose optimization strategies that determine how the modules should learn from data. In this way, the compilerautomatically maps the declarative modules to high-quality compositions of prompting, finetuning, reasoning, and augmentation.


> DSPy programs are expressed in Python: each program takes the task input (e.g., a question to answer or a paper to summarize) and returns the output (e.g., an answer or a summary) after a series of steps. DSPy contributes three abstractions toward automatic optimization: signatures, modules, and teleprompters. Signatures abstract the input/output behavior of a module; modules replace existing hand-prompting techniques and can be composed in arbitrary pipelines; and teleprompters optimize all modules in the pipeline to maximize a metric.

---

> Instead of free-form string prompts, DSPy programs use natural language signatures to assign work to the LM. A DSPy signature is natural-language typed declaration of a function: a short declarative spec that tells DSPy what a text transformation needs to do (e.g., “consume questions and return answers”), rather than how a specific LM should be prompted to implement that behavior. 

这里的signature借用了编程语言中的术语，含义是类似的。在编程语言中，signature是函数的输入类型和返回类型，例如，在Haskell中，`add :: Int -> Int -> Int`是一个签名，它的意思是，add函数输入两个Int，返回一个Int。DSPy签名告诉LM需要做什么（而不是怎么做）。

---

> Akin to type signatures in programming languages, DSPy signatures simply define an interface and provide type-like hints on the expected behavior. To use a signature, we must declare a module with that signature, like we instantiated a Predict module above. A module declaration like this returns a function having that signature.

用编程语言的术语类比，DSPy signatures就像接口，只定义方法，不提供实现细节。module declaration则是实现。

---

>  Like layers in PyTorch, the instantiated module behaves as a callable function: it takes in keyword arguments corresponding to the signature input fields (e.g., question), formats a prompt to implement the signature and includes the appropriate demonstrations, calls the LM, and parses the output fields.

Field就是signature中的一个字段，类比于函数定义中的类型。

---

> Parameterization Uniquely, DSPy parameterizes these prompting techniques. To understand this parameterization, observe that any LM call seeking to implement a particular signature needs to specify parameters that include: (1) the specific LM to call, (2) the prompt instructions and the string prefix of each signature field and, most importantly, (3) the demonstrations used as few-shot prompts (for frozen LMs) or as training data (for finetuning). We focus primarily on automatically generating and selecting useful demonstrations. In our case studies, we find that bootstrapping good demonstrations gives us a powerful way to teach sophisticated pipelines of LMs new behaviors systematically.

在DSPy中，参数有（1）所调用的语言模型。（2）prompt instructions and string prefix of each signature field。（3）示例，用作few shot或训练数据。

---

> DSPy programs may use tools, which are modules that execute computation. 

和Agent中的tools是一个概念。

---

> Programs: DSPy modules can be composed in arbitrary pipelines in a define-by-run interface. Inspired directly by PyTorch and Chainer, one first declares the modules needed at initialization, allowing DSPy to keep track of them for optimization, and then one expresses the pipeline with arbitrary code that calls the modules in a forward method. 

“Define-by-run” 是指在运行时动态定义计算图的编程范式，也就是动态图。Programs也是modules，就像torch中，一些小的module组成一个最终的大的module。

---

> When compiling a DSPy program, we generally invoke a teleprompter, which is an optimizer that takes the program, a training set, and a metric—and returns a new optimized program. Different teleprompters apply different strategies for optimization.

teleprompter类比于torch中的优化器。

---

> A key source of DSPy’s expressive power is its ability to compile—or automatically optimize—any program in this programming model. Compiling relies on a teleprompter, which is an optimizer for DSPy programs that improves the quality (or cost) of modules via prompting or finetuning, which are unified in DSPy. While DSPy does not enforce this when creating new teleprompters, typical teleprompters go through three stages.

compile类比于torch中的train。

---















