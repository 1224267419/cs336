import torch
import torch.nn as nn
from einops import rearrange
try:
    from .linear import Linear,Embedding
except ImportError:
    from linear import Linear
import math
import numpy as np

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model=d_model
        self.eps=eps
        self.weight=torch.nn.Parameter(torch.ones(d_model,device=device, dtype=dtype))
        
    def forward(self, x: torch.Tensor)-> torch.Tensor :
        in_dtype=x.dtype
        x=x.to(torch.float32)
        # temp=x*x/self.d_model
        temp=torch.mean(x**2, dim=-1, keepdim=True)
        rms=(temp+self.eps)**0.5
        result=x*self.weight/rms
        return  result.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self,d_model:int,d_ff:int,device=None,dtype=None):
        super().__init__()
        self.d_model=d_model
        self.weight1=torch.nn.Parameter(torch.randn(d_ff,d_model,device=device,dtype=dtype))
        self.weight2=torch.nn.Parameter(torch.randn(d_model,d_ff,device=device,dtype=dtype))
        self.weight3=torch.nn.Parameter(torch.randn(d_ff,d_model,device=device,dtype=dtype))
    
    def silu(self,x:torch.Tensor,)->torch.Tensor:
        return x*torch.sigmoid(x)

    def GLU(self,x:torch.Tensor,weight1:torch.Tensor,weight2:torch.Tensor)->torch.Tensor:
        #内部是矩阵乘法，两个unit间是逐元素乘法(gate)
        return self.silu(x@weight1.T)*(x@weight2.T)

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        return (self.silu(x@self.weight1.T)*(x@self.weight3.T))@self.weight2.T

class RoPE(nn.Module):
    def __init__(
        self,
        d_k: int,
        theta: float,
        max_seq_len: int,
        device=None,
        dtype=None,
    ):
        super().__init__()

        if d_k % 2 != 0:
            raise ValueError("d_k 必须是偶数")

        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len

        # 所有 token 位置：
        # [0, 1, 2, ..., max_seq_len - 1]
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)

        # 每组二维特征对应的索引：
        # [0, 2, 4, ..., d_k - 2]
        dimension_indices = torch.arange(
            0,
            d_k,
            2,
            device=device,
            dtype=torch.float32,
        )

        # 每组二维特征的旋转频率：
        # 1 / theta^(0/d_k)
        # 1 / theta^(2/d_k)
        # 1 / theta^(4/d_k)
        # ...
        inverse_frequencies = theta ** (
            -dimension_indices / d_k
        )

        # 每个位置、每组维度的旋转角度
        #
        # angles[p, k]
        # = position[p] * inverse_frequency[k]
        angles = torch.outer(
            positions,
            inverse_frequencies,
        )

        # 提前缓存旋转需要的 sin 和 cos
        cos_cache = torch.cos(angles)
        sin_cache = torch.sin(angles)

        self.register_buffer(
            "cos_cache",
            cos_cache,
        )

        self.register_buffer(
            "sin_cache",
            sin_cache,
        )

    def forward(self, x: torch.Tensor,token_positions:torch.Tensor) -> torch.Tensor:
        if token_positions.numel() == 0:
            return x
        if token_positions.min() < 0 or token_positions.max() >= self.max_seq_len:
            raise ValueError(
                f"token_positions 必须位于 [0, {self.max_seq_len}) 内"
            )

        token_positions = token_positions.to(
            device=self.cos_cache.device,
            dtype=torch.long)
        cos=self.cos_cache[token_positions].to(dtype=x.dtype)
        sin=self.sin_cache[token_positions].to(dtype=x.dtype)

        #取奇偶
        x_even=x[...,::2]
        x_odd=x[...,1::2]

        #应用旋转矩阵

        x_rotated_even=x_even*cos-x_odd*sin
        x_rotated_odd=x_even*sin+x_odd*cos


        output=torch.stack([x_rotated_even,x_rotated_odd],dim=-1)
        #最后两个维度展平回到一维
        output=output.flatten(-2)
        return output

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # 计算 softmax
    exp_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)
    return exp_x/torch.sum(exp_x, dim=dim, keepdim=True)

def scaled_dot_product_attention(
    Q,
    K,
    V,
    mask = None,
):
    
    score=(Q @ K.transpose(-2, -1)) / Q.shape[-1]**0.5
    # 先 mask 再 softmax，被 mask 的位置的权重为 0。
    if mask is not None:
        score = score.masked_fill(~mask.to(device=score.device, dtype=torch.bool), float("-inf"))
    #softmax使得每个维度上的attention score 和为1
    score=softmax(score,dim=-1)
    
    return score@V

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float = 10000.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        # 检查 d_model 能否整除 num_heads
        if d_model%num_heads!=0:
            raise ValueError("d_model 必须能被 num_heads 整除")
        # 保存 d_model、num_heads、head_dim
        self.d_model=d_model
        self.num_heads=num_heads
        head_dim=d_model//num_heads
        self.head_dim=head_dim
        
        # 创建 q_proj、k_proj、v_proj、output_proj
        # Q=xWq...
        self.q_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        self.k_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        self.v_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        self.output_proj=Linear(d_model,d_model,device=device,dtype=dtype)


    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        #用前导batch_like ，兼容各种batch
        *leading_dims, seq_len, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError(f"输入最后一维应为 {self.d_model}，实际为 {d_model}")
        # 2. 计算 Q、K、V
        #    (..., seq, d_model)
        Q=self.q_proj(x)
        K=self.k_proj(x)
        V=self.v_proj(x)



        # 3. 拆分 head
        #    (..., heads, seq, head_dim)
        Q = Q.reshape(*leading_dims, seq_len, self.num_heads, self.head_dim).transpose(-3, -2)
        K = K.reshape(*leading_dims, seq_len, self.num_heads, self.head_dim).transpose(-3, -2)
        V = V.reshape(*leading_dims, seq_len, self.num_heads, self.head_dim).transpose(-3, -2)


        # 4. 创建 causal mask
        #    (seq, seq)
        #下三角矩阵，上三角全为0，用diagonal可以调整对角线位置
        mask=torch.tril(torch.ones(seq_len,seq_len,dtype=torch.bool,device=x.device))

        # 5. 调用 scaled_dot_product_attention
        res=scaled_dot_product_attention(Q,K,V,mask)
        # 6. 把 heads 拼回 d_model
        res = res.transpose(-3, -2).reshape(*leading_dims, seq_len, d_model)
        # 7.返回 output projection
        return self.output_proj(res)

class MultiHeadSelfAttentionWithRope(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float = 10000.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        # 检查 d_model 能否整除 num_heads
        if d_model%num_heads!=0:
            raise ValueError("d_model 必须能被 num_heads 整除")
        # 保存 d_model、num_heads、head_dim
        self.d_model=d_model
        self.num_heads=num_heads
        head_dim=d_model//num_heads
        self.head_dim=head_dim
        
        # 创建 q_proj、k_proj、v_proj、output_proj
        # Q=xWq...
        self.q_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        self.k_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        self.v_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        self.output_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        if max_seq_len is None:
            raise ValueError("使用 RoPE 时必须提供 max_seq_len")
        self.rope = RoPE(
            d_k=self.head_dim,
            theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None, 
        #在增量生成或 KV cache 中，当前输入虽然可能只有一个 token，但它的真实位置不一定是 0
    ) -> torch.Tensor:
        #用前导batch_like ，兼容各种batch
        *leading_dims, seq_len, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError(f"输入最后一维应为 {self.d_model}，实际为 {d_model}")
        # 2. 计算 Q、K、V
        #    (..., seq, d_model)
        Q=self.q_proj(x)
        K=self.k_proj(x)
        V=self.v_proj(x)



        # 3. 拆分 head
        #    (..., heads, seq, head_dim)
        Q = Q.reshape(*leading_dims, seq_len, self.num_heads, self.head_dim).transpose(-3, -2)
        K = K.reshape(*leading_dims, seq_len, self.num_heads, self.head_dim).transpose(-3, -2)
        V = V.reshape(*leading_dims, seq_len, self.num_heads, self.head_dim).transpose(-3, -2)
        # 初始一般是(1,len)，为了兼容batch
        # 这里unsqueeze(-2)是为了在head处也做广播
        if token_positions is None:
            rope_positions = torch.arange(
                seq_len,
                device=x.device,
                dtype=torch.long,
            )
        else:
            rope_positions = token_positions.to(
                device=x.device,
                dtype=torch.long,
            )
        rope_positions=rope_positions.unsqueeze(-2)
        Q=self.rope(Q,rope_positions)
        K=self.rope(K,rope_positions)

        # 4. 创建 causal mask
        #    (seq, seq)
        #下三角矩阵，上三角全为0，用diagonal可以调整对角线位置
        mask=torch.tril(torch.ones(seq_len,seq_len,dtype=torch.bool,device=x.device))

        # 5. 调用 scaled_dot_product_attention
        res=scaled_dot_product_attention(Q,K,V,mask)
        # 6. 把 heads 拼回 d_model
        res = res.transpose(-3, -2).reshape(*leading_dims, seq_len, d_model)
        # 7.返回 output projection
        return self.output_proj(res)
    
class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int | None = None,
        theta: float = 10000.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.Attention=MultiHeadSelfAttentionWithRope(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )
        self.SwiGLU=SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype
        )
        #两个ln的权重不同
        self.ln1=RMSNorm(d_model=d_model,
                             device=device,
                             dtype=dtype)
        self.ln2=RMSNorm(d_model=d_model,
                             device=device,
                             dtype=dtype)
    def forward(self,x:torch.Tensor,token_positions
    ):
        y=x+self.Attention(self.ln1(x),token_positions)
        z=y+self.SwiGLU(self.ln2(y))
        return z

class TransformerLM(nn.Module):
    # Token Embedding ->transformer block *n ->RMSNorm ->logits 
    def __init__(  
            self,  
            vocab_size: int,
            context_length: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            rope_theta: float,
            device=None,
            dtype=None,
            ):
        super().__init__()
        self.embed=Embedding(vocab_size,d_model)
    #   #可以这样子定义，但是管理会比较麻烦，用
    #   # nn.ModuleList更好
    #     for i in range(num_layers):
    #             setattr(
    #     self,
    #     f"tfblock{i}",
    #     TransformerBlock(
    #         d_model=d_model,
    #         num_heads=num_heads,
    #     ),
    # )
        self.layers=nn.ModuleList(
            [TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                device=device,
                dtype=dtype
            ) 
            for _ in range(num_layers)]
        )
        self.rmsn=RMSNorm(d_model=d_model,device=device,dtype=dtype)
        self.lm=Linear(d_model,vocab_size)
    def forward(self,
                x:torch.Tensor,
                
                ):
        *batch,lenth=x.shape
        token_positions=torch.arange(0,lenth,device=x.device,dtype=torch.long)
        x=self.embed(x)
        for layer in self.layers:
            x=layer(x,
                    token_positions=token_positions,)
        x=self.rmsn(x)
        x=self.lm(x)
        return x
        # return torch.logit(x)

def cross_entropy(logits,targets):
        max_values, indices=torch.max(logits,dim=-1,keepdim=True)

        shifted_logits=logits-max_values
        e_logits=torch.exp(shifted_logits)
        e_logits_sum=torch.sum(e_logits,dim=-1,keepdim=True)
        #扩张-1 ，用于gather
        z_y=shifted_logits.gather(
            dim=-1,
            index=targets.unsqueeze(-1),
        )
        
        loss=torch.log(e_logits_sum)-z_y

        return loss.mean()

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    ):
        # 检查：
        # lr 是否非负
        # betas 是否位于 [0, 1)
        # eps 是否非负
        # weight_decay 是否非负

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }

        super().__init__(params, defaults)
        

    def step(self,closure=None):
        loss =None
        if closure is not None:
            loss = closure()
        # 读取这一组的超参数
            
        for group in self.param_groups:
            # AdamW 只需要长期保存step,m，v ， 其余都能通过计算得到
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                # 跳过没有梯度的参数
                if p.grad is None:
                    continue
                # 取得 p.grad
                grad=p.grad
                # 取得或初始化 self.state[p]
                state = self.state[p]
                
                if len(state)==0:
                        state["step"] = 0
                        state["m"] = torch.zeros_like(p)
                        state["v"] = torch.zeros_like(p)
                
                m = state["m"]
                v = state["v"]
                # 更新 step
                state['step']+=1        
                t = state["step"]

                # # 更新 m,v
                # state["m"]=beta1*m+(1-beta1)* grad
                # state["v"]=v*beta2+(1-beta2)*grad**2

                #原地更新m,v ,避免创建新的变量
                m.mul_(beta1)
                m.add_(grad, alpha=1 - beta1)

                v.mul_(beta2)
                v.addcmul_(grad, grad, value=1 - beta2)

                
                # 计算 bias correction 后的学习率,
                adjust_lr=lr*math.sqrt(1-beta2**t)/(1-beta1**t)

                # #错误版本，没有no_gard
                # # weight decay
                # p-=lr * weight_decay * p
                # # 参数更新,而且这里的m和v都不是tensor
                # p-= adjust_lr*state["m"]/(torch.sqrt(state["v"])+eps)
                with torch.no_grad():
                    # weight decay
                    p.mul_(1 - lr * weight_decay)
                    # 参数更新
                    p.add_(-( adjust_lr*state["m"]/(torch.sqrt(state["v"])+eps)))

        return loss
def get_lr_cosine_schedule(
    it,
    max_learning_rate,
    min_learning_rate,
    warmup_iters,
    cosine_cycle_iters,
):
    if it < warmup_iters:
        return  it / warmup_iters * max_learning_rate

    elif it <= cosine_cycle_iters:
        progress = (it-warmup_iters)/(cosine_cycle_iters-warmup_iters)
        cosine_factor =  0.5 * (1 + math.cos(math.pi * progress))

        return  min_learning_rate + cosine_factor * (max_learning_rate - min_learning_rate)

    else:
        return  min_learning_rate
    
def gradient_clipping(parameters, max_norm,eps=1e-6):
    # parameters通常是一次性迭代器，需要转成list
    parameters = list(parameters)
    total_norm=0
    #计算梯度范数
    for p in parameters:
        if p.grad is None:
            continue
        total_norm+=torch.sum(p.grad**2)
    total_norm.sqrt_()
    if total_norm>max_norm:
        scale=max_norm/(eps+total_norm)
        for p in parameters:
            if p.grad is None:
                continue
            p.grad*=scale
    
def get_batch(
    dataset, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    # 1. 随机生成 batch_size 个有效起点
    starts=torch.randint(0,len(dataset)-context_length,(batch_size,))
    # 2. 对每个起点 start：
    #    input 取 dataset[start:start + context_length]
    #    target 取 dataset[start + 1:start + context_length + 1]
    batch_data=[]
    batch_target=[]
    for start in starts:
        start=start.item()
        batch_data.append(dataset[start:start+context_length]  )
        batch_target.append(dataset[start + 1:start + context_length + 1])

    # 3. 将各段组合成二维数组
    # 4. 转成 torch.long, 移到指定 device
    # stack适合将list中的多个元素拼接并产生新的维度，且可以用dim控制
    # (RoPE中也有过类似的操作)

    batch_data = torch.from_numpy(
        np.stack(batch_data)
    ).to(
        device=device,
        dtype=torch.long,
    )

    batch_target = torch.from_numpy(
        np.stack(batch_target)
    ).to(
        device=device,
        dtype=torch.long,
    )


    # 6. 返回 inputs, targets
    return batch_data,batch_target

def save_checkpoint(model:nn.Module, optimizer:torch.optim.Optimizer, iteration:int, out:str):
    model_state=model.state_dict()
    optimizer_state=optimizer.state_dict()
    checkpoint={
        "model":model_state,
        "optimizer":optimizer_state,
        "iteration": iteration
    }
    torch.save(checkpoint,out)
    
def load_checkpoint(src,model:nn.Module, optimizer:torch.optim.Optimizer):
    checkpoint=torch.load(src)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint["iteration"]

if __name__=="__main__":

    batch_size,sequence_length, d_model=10,30,768

    # # RMSNorm测试
    # rmsn=RMSNorm(d_model)
    # x=torch.ones(batch_size,sequence_length, d_model)
    # print(x.shape)
    # print(rmsn(x).shape)

    #SwiGLU测试
    # d_ff=2048
    # swiglu=SwiGLU(d_model,d_ff)
    # x=torch.ones(batch_size,sequence_length, d_model)
    # print(x.shape)
    # print(swiglu(x).shape)

    # num_heads=8
    # mu=MultiHeadSelfAttention(d_model,num_heads)
    # x=torch.randn(batch_size,sequence_length,d_model)
    # print(x.shape)
    # y=mu(x)
    # print(y.shape)

    # mur=MultiHeadSelfAttentionWithRope(d_model,num_heads,100)
    # print(x.shape)
    # y=mur(x,torch.arange(0,30))
    # print(y.shape)
    dataset=torch.randint(0,100,(100,))
    batch_size=10
    context_length=5
    batch_data,batch_target=get_batch(dataset,batch_size,context_length,"cpu")
    print(batch_data.shape)
    print(batch_target.shape)
    print(batch_data.dtype)
    print(batch_target.dtype)
    print(batch_data.device)