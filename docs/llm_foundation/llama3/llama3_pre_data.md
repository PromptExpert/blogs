# Llama3 预训练数据

> Compared to prior versions of Llama, we improved both the quantity and quality of the data we use for pre-training and post-training. These improvements include the development of more careful pre-processing and curation pipelines for pre-training data and the development of more rigorous quality assurance and filtering approaches for post-training data. We pre-train Llama 3 on a corpus of about 15T multilingual tokens, compared to 1.8T tokens for Llama 2.

> We create our dataset for language model pre-training from a variety of data sources containing knowledge until the end of 2023. We apply several de-duplication methods and data cleaning mechanisms on each data source to obtain high-quality tokens. We remove domains that contain large amounts of personally identifiable information (PII), and domains with known adult content.

## Web Data Curation

> Much of the data we utilize is obtained from the web and we describe our cleaning process below.

### 过滤掉有害信息和PII信息。
如题。

### HTML抽取和清洗
- 构建了一个parse，能够去除广告、导航栏、版权声明等信息，保留主体信息，比第三方的好。
- 小心处理公式和代码，保留原来的结构。
- markdown对模型表现有害，删除了。

### 去重
- URL级别去重。每个URL，只保留最新的版本。
- 文档级别去重。MinHash去重。
- 行级别去重。和ccNet类似。实践证明有较大提升。

### Heuristic去重
用各种规则和pattern，删除各类重复。

### 基于模型的质量过滤
- fasttext分类，判断文档是否会被wikipedia引用。
- 微调Llama 2，给定一些质量要求，让它判断是否满足质量要求。以微调后的Llama 2生成的数据为训练数据，训练DistilRoberta，为文档输出一个质量分。

### 代码和推理数据
为代码和数学相关的文档特别处理。特别微调Llama 2，特别解析。

### 多语言数据
- 用fasttext分类语种。
- 每个语言内部，进行文档级别和行级别去重。
- 每个语言分别过滤低质量。
- 实验选择多语言的token量，平衡英语和多语言评测。

## Determining the Data Mix
> To obtain a high-quality language model, it is essential to carefully determine the proportion of different data sources in the pre-training data mix. Our main tools in determining this data mix are knowledge classification and scaling law experiments.

### Knowledge classification
训练一个知识领域分类器。

### Scaling laws for data mix
> To determine the best data mix, we perform scaling law experiments in which we train several small models on a data mix and use that to predict the performance of a large model on that mix. We repeat this process multiple times for different data mixes to select a new data mix candidate. Subsequently, we train a larger model on this candidate data mix and evaluate the performance of
that model on several key benchmarks.

### Data mix summary
最终的数据集，包含50%的通用领域，25%的数学和推理，17%的代码，8%的多语言。

## Annealing Data
Annealing阶段是学习率降低、训练趋于收敛的阶段。

在退火阶段，用少量高质量代码和数学数据可提高模型表现。
