import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features=in_features
        self.out_features=out_features
        self.weight=nn.Parameter(torch.empty(self.out_features,self.in_features,device=device,dtype=dtype))
        #随机初始化
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=1,
            a=-3 ,
            b=3 ,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #这里可以不写，因为矩阵乘法本身就会在维度错误时抛出异常。
        if x.shape[-1]!=self.in_features:
            print("维度不匹配")
        return x@self.weight.T
    
class Embedding(nn.Module):
    def __init__(self, embedding_dim,d_model,device=None, dtype=None):
        super().__init__()
        self.weight=nn.Parameter(torch.empty(embedding_dim,d_model,device=device,dtype=dtype))
        self.num_embeddings = d_model
        self.embedding_dim = embedding_dim
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=1,
            a=-3 ,
            b=3 ,
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
