# ass1

```python


def merge_token(token_tuple, pair):
    #合并后的token样式,这里用的简单匹配，用kmp应该会快一点? 这个想法是错误的，因为匹配长度固定为2，KMP只会增加开销
    # token_tuple是指合并几次后的序列，里面的元素可能不再是单个字
    new_token = []
    i = 0

    while i < len(token_tuple):
        if (i <= len(token_tuple) - len(pair)):
            token=False
            for j in range( len(pair)):

                if token_tuple[i+j]!=pair[j]:
                    token=True
                    break
            if token==True:
                new_token.append(pair[0] + pair[1])
                #合并了相邻的两个token,所以i要加2
                i += 2
            else:
                new_token.append(token_tuple[i])
                i += 1     

        else:
            #用于退出死循环
            new_token.append(token_tuple[i])
            i += 1
    
    return tuple(new_token)
```

gpt-2 类型的BPE的合并是全文用正则预分词 ， 然后在每个片段内部单独merge，**merge过程不会跨片段**
最基础的原版的BPE允许跨pair合并

按单词做拆分是为了防止

BPE training 每一轮是：

```
1. 对整个语料的 pretoken_counts 统计所有 pair 频率
2. 选出全局最高频 pair
3. 把这个 pair 应用到所有 pre-token 里
4. 得到新的 pretoken_counts
5. 进入下一轮
```

也就是说：

```
“选 pair” 是全局的；
“执行 merge” 是遍历每个 pre-token 局部完成的；
“pretoken_counts dict” 是整篇文章 / 整个语料的压缩表示。
```

你的代码里应该这样做：

```
new_pretokens = defaultdict(int)

for token_tuple, count in pretokens.items():
    merged_tuple = merge_token(token_tuple, pair)
    new_pretokens[merged_tuple] += count
```

#### BPE的性能瓶颈和并发修改

```python
    #性能瓶颈,最好考虑并发实现
    for part in tqdm(parts, desc="正则预分词", unit="片段"):
        for match in re.finditer(PAT, part):
            pretoken = match.group()
            bs = pretoken.encode("utf-8")
            pretoken_bytes = tuple(bytes([b]) for b in bs)
            pretoken_counts[pretoken_bytes] += 1
```

最大的性能瓶颈在这里，即re.finditer(PAT, part)这部分正则实现，而这部分实际上是可以实现并发的，

我在文件中做了并行化处理，具体看代码吧，不赘述

还有个点，并行化越高收益越低，单线程越240s ， 8线程68s，16线程39s，还是能看出来收益衰减的

在训练时由于我的数据量很大，且词表每做很大，导致下面的代码我只做了num_workers的优化，但speed test要求merge的速度，因此下面`pair_count=countPairs(pretoken)`每次全量更新的方法导致时间极大，增量更新pair_count这个过程需要在merge中间做处理，即在`merge_token(token_tuple, pair):`过程

````python
def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    *,
    num_workers: int = 4,
) -> tuple[
    dict[int, bytes],
    list[tuple[bytes, bytes]],
]:
    #默认for 循环过不了 speed test,问题出在BPE merge上   
    vocab=init_vocab(special_tokens)

    start = time.perf_counter()
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    pretoken=count_vocab_parallel(special_tokens, text,num_workers=num_workers)

    print(f"countVocab耗时：{time.perf_counter() - start:.3f} 秒")
    merge=[]

    bpe_start = time.perf_counter()
    num_tokens_to_add = vocab_size - len(vocab)
    pair_count=countPairs(pretoken)
    with tqdm(total=vocab_size-len(vocab), desc="训练 BPE") as progress:
        while len(vocab)<vocab_size:
            #每次都执行countPairs导致复杂度超了
            # pair_count=countPairs(pretoken)
            if not pair_count:
                break

            pair=get_best_pair(pair_count) 
            vocab[len(vocab)]=pair[0]+pair[1]
            merge.append((pair[0],pair[1]))

            #循环遍历所有pair
            new_pretokens=defaultdict(int)
            for token_tuple, count in pretoken.items():
                merged_tuple=merge_token(token_tuple,pair)
                new_pretokens[merged_tuple]+=count
            pretoken=new_pretokens
            # print(new_pretokens)
            progress.update(1)
            # 更新pair_count
            pair_count=countPairs(pretoken)

    elapsed = time.perf_counter() - bpe_start
    print(f"产生{num_tokens_to_add}个新token共计耗时：{elapsed:.3f} 秒")
    # #合并后的词表，最终的tuple token,训练过程
    # print(pretoken)
    return vocab,merge
````

但是上面的耗时约为2.36s，远超作业要求的1.5s

### 增量更新

现在慢的根本原因是 `for token_tuple, count in pretoken.items():` 每一轮都要遍历所有pre-token , 但我们可以`建立一个索引，记录某个 pair 出现在哪些 token_tuple 里` ,然后再使用增量更新 ， 从而在修改时不用每次都全局遍历一次

这部分先跳过吧，bpe的价值有点低了

直接进transformer部分

## 1.2transformer

第一个点 : 在torch中通常使用 $W \in \mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}$，行向量 $$x \in \mathbb{R}^{batch\times d_{\text{in}}}$$  即将batch放在前半部分，即使用行向量，使得$$y = xW^\top,$$ 但在matlab等其他工具包常使用$$y = Wx$$的转置版本

### 1.2.3.3 基本构建块：Linear 与 Embedding 模块

这里我们将不依赖 torch.nn中的现成函数，手工构建 Linear 与 Embedding 模块

对于初始化参数，Pre-norm Transformers 对初始化十分鲁棒，但初始化仍然会显著影响训练速度和收敛。我们这里使用一个近似的鲁棒初始化

- Linear weights：$\mathcal{N}\left(\mu=0,\sigma^2=\frac{2}{d_{\text{in}}+d_{\text{out}}}\right)$，在 $[-3\sigma, 3\sigma]$ 截断。
- Embedding：$\mathcal{N}(\mu=0,\sigma^2=1)$，在 $[-3, 3]$ 截断。
- RMSNorm：1。

这里应使用 `torch.nn.init.trunc_normal_` 初始化截断正态权重。

```python
class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features=in_features
        self.out_features=out_features
        std = (2 / (in_features + out_features)) ** 0.5
        self.weight=nn.Parameter(torch.randn(self.out_features,self.in_features,device=device,dtype=dtype))
        #随机初始化
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #这里可以不写，因为矩阵乘法本身就会在维度错误时抛出异常。
        if x.shape[-1]!=self.in_features:
            print("维度不匹配")
        return x@self.weight.T
```



Embedding部分还多了一个点：怎么从int值映射到维度为dim的向量,使用for循环会很慢，我们需要Tensor高级索引，即 `output=self.weight[x]`

```python
class Embedding(nn.Module):
    def __init__(self, num_embeddings,d_model,device=None, dtype=None):
        super().__init__()
        self.weight=nn.Parameter(torch.empty(num_embeddings,d_model,device=device,dtype=dtype))
        
        std = (2 / (num_embeddings+d_model)) ** 0.5
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        output=self.weight[x]
        return output
if __name__=='__main__':
    vocab,dim=1000,768
    emb=Embedding(vocab,dim)
    batch,seq_len=32,300
    x=torch.randint(0,1000,(batch,seq_len))
    y=emb(x)
    print(x.shape)
    print(y.shape)
```

### 1.2.3.4 [Transformer block ](assignment1-basics-main/code/transformer/transformer_block.py)

Transformer block 有两个 sub-layers：一个 multi-head self-attention mechanism 和一个 position wise feed-forward network

接下来我们构建这两个sub-layer,并将其合并为完整的Transformer block

#### RMSNorm 

$$
\mathrm{RMSNorm(a_{i})}=\frac{\mathrm{a_{i}}}{\mathrm{RMS(a)}}g_i \\
其中\begin{array}{r}{\mathsf{RMS}(a)=\sqrt{\frac{1}{\mathsf{d}_{m o d e l}}\sum_{i=1}^{d_{m o d e l}}a_{i}^{2}+\epsilon}}\end{array}\\
也可以写成RMS(x)=mean(x^2)+ϵ
	​
$$

注意，这里不是求全局平均，而是**对最后一维求平均**，从而得到一个标量，详见代码

```python
temp = torch.mean(x * x, dim=-1, keepdim=True)
rms = torch.sqrt(temp + self.eps)
```





#### SWiGLU

公式如下，实现在代码中有

$ \text{SwiGLU}(x, W_1, W_2, W_3) = W_2 \big(\text{SiLU}(W_1 x) \odot W_3 x\big) $

#### ROPE

公式如下

$ R = \begin{bmatrix} \cos\alpha & \sin\alpha \\ -\sin\alpha & \cos\alpha \end{bmatrix} \\且 Y= R\begin{bmatrix}x_0 \\ x_1 \end{bmatrix}$

即$$y_0 = x_0 \cos\alpha - x_1 \sin\alpha,\\ y_1 = x_0 \sin\alpha + x_1 \cos\alpha$$

代码中的：

```
output = torch.stack(
    [x_rotated_even, x_rotated_odd],
    dim=-1
)
```

结果是：

```
[
    [e0, o0],
    [e1, o1],
    [e2, o2],
    [e3, o3],
]
```

因此后续直接flatten(-2)即可恢复正确顺序





#### scaled_dot_product_attention

$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$

且使用mask 用于掩盖后续的seq，从而避免attention得到不应得到的信息

#### MultiHeadSelfAttentionWithRope

forward时带上rope，只有QK要做RoPE处理，下三角矩阵`torch.tril`生成mask

#### cross_entropy

计算交叉熵，可以看看公式理解一下，这里用和safe_softmax一样的-max方法防止溢出

### AdamW

optimizer的更新，具体详情看代码

#### get_lr_cosine_schedule

余弦退火lr

#### gradient_clipping

clip 梯度范数，避免梯度过大

#### save/load_checkpoint

需要加载的有三部分，即：` "model": 模型状态,    "optimizer": 优化器状态,    "iteration": 当前迭代次数` , 通过这三部分的保存和加载实现断点续连
