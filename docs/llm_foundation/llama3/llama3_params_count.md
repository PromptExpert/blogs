# Llama3的参数量计算

以Llama-3.1-8B为例，结合[llama3的实现](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/api/model.py)，计算llama3的参数量。

llama3的参数量是以下部分之和：
- Embedding层
- Transformer Block * Block层数
- Output层

## Config
```json
{
    "dim": 4096, 
    "ffn_dim_multiplier": 1.3, 
    "multiple_of": 1024, 
    "n_heads": 32, 
    "n_kv_heads": 8, 
    "n_layers": 32, 
    "norm_eps": 1e-05, 
    "rope_theta": 500000.0, 
    "use_scaled_rope": true, 
    "vocab_size": 128256
}
```

## Embedding层

```python
self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
```

计算公式：embed_params = vocab_size * dim

参数量：0.53B

## Transformer Block
### RMSNorm
```python
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
```

计算公式：rms_params = dim + dim # each Transformer block has two RMSNorm layers
计算结果：0.00B

### Attention
```python
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
```

计算公式：
```
head_dim = dim / n_heads
query_proj_params = dim*(n_heads*head_dim) 
key_proj_params = dim*(n_kv_heads*head_dim)
value_proj_params = dim*(n_kv_heads*head_dim)
out_proj_params = (n_heads*head_dim)*dim
attn_params = query_proj_params+key_proj_params+value_proj_params+out_proj_params
``` 
计算结果：0.04B

### FFN
```python
class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
```

计算公式：
```
hidden_dim = 4 * args.dim
hidden_dim = int(2 * dim*4 / 3)
hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
hidden_dim = int(ffn_dim_multiplier * hidden_dim)
w1_params = dim * hidden_dim
w2_params = dim * hidden_dim
w3_params = dim * hidden_dim
ffn_params = w1_params+w2_params+w3_params
```

计算结果：0.18B

### Single Transformer Block Sum
计算公式：transformer_block_params = rms_params+attn_params+ffn_params

计算结果：0.22B

## All Transformer Blocks Sum
计算公式：transformer_params = transformer_block_params*n_layers

计算结果：7.10B

## Output
```python
self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
```
计算公式：output_params = dim * vocab_size

计算结果：0.53B

## Total
```python
class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
```
计算公式：total_params = embed_params + transformer_params + output_params

计算结果：8.00B

## 计算公式汇总
```python
# config
dim = 4096
n_layers = 32
n_heads = 32
n_kv_heads = 8
ffn_dim = 6144
vocab_size = 128256
multiple_of = 1024  # make SwiGLU hidden layer size multiple of large power of 2
ffn_dim_multiplier = 1.3
norm_eps = 1e-5
rope_theta = 500000
max_seq_len = 8192

# Emebdding
embed_params = vocab_size * dim
print (f'Parameters of Embebdding layer = {embed_params/10**9:.2f}B')

# RMSNorm
rms_params = dim + dim # each Transformer layer has two RMSNorm layers
print (f'Parameters of RMSNorm layer = {rms_params/10**9:.2f}B')

# Attention
head_dim = dim / n_heads
query_proj_params = dim*(n_heads*head_dim)
key_proj_params = dim*(n_kv_heads*head_dim)
value_proj_params = dim*(n_kv_heads*head_dim)
out_proj_params = (n_heads*head_dim)*dim
attn_params = query_proj_params+key_proj_params+value_proj_params+out_proj_params
print (f'Parameters of Attention layer = {attn_params/10**9:.2f}B')

# FFN
hidden_dim = int(2 * dim*4 / 3)
hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
hidden_dim = int(ffn_dim_multiplier * hidden_dim)
print (f'FFN hidden_dim = {hidden_dim}')
w1_params = dim * hidden_dim
w2_params = dim * hidden_dim
w3_params = dim * hidden_dim
ffn_params = w1_params+w2_params+w3_params
print (f'Parameters of FFN layer = {ffn_params/10**9:.2f}B')

# Transformer Block
transformer_block_params = rms_params+attn_params+ffn_params
print (f'Parameters of Transformer block = {transformer_block_params/10**9:.2f}B')

# Transformer
transformer_params = transformer_block_params*n_layers
print (f'Parameters of Transformer = {transformer_params/10**9:.2f}B')

# Output
output_params = dim * vocab_size
print (f'Parameters of output layer = {output_params/10**9:.2f}B')

# Total
total_params = embed_params + transformer_params + output_params
print (f'Parameters total = {total_params//10**9:.2f}B')

#Parameters of Embebdding layer = 0.53B
#Parameters of RMSNorm layer = 0.00B
#Parameters of Attention layer = 0.04B
#FFN hidden_dim = 14643
#Parameters of FFN layer = 0.18B
#Parameters of Transformer block = 0.22B
#Parameters of Transformer = 7.10B
#Parameters of output layer = 0.53B
#Parameters total = 8.00B
```