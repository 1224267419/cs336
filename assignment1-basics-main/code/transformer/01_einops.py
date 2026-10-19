import torch
from einops import rearrange, einsum

batch, sequence, d_in, d_out,=3,4,5,6
D=torch.ones(3,4,5)
A=torch.ones(4,5)
Y1=D@A.T
Y2=einsum(D,A,"b s i , o i -> b s o")
print(torch.all(Y1==Y2))
x=torch.tensor([i for i in range(batch*sequence*d_in)])
print(x)
x=rearrange(x,
            "(batch lenth dim)-> batch lenth dim",
            batch=batch,
            sequence=sequence,
            )
print(x)

#假设要让图片变暗
batch,hight,weight,channel=3,480,960,3

image=torch.randn(batch,hight,weight,channel)
light=torch.linspace(0,1,10)
new_light=rearrange(light,
                "l -> 1 l 1 1 1",
                )
new_image=rearrange(
    image,
    "batch hight weight channal ->batch 1 hight weight channal"
)
#rearrange以后广播
new_image1=new_light*new_image
#直接用einsum，免去之前的计算部分
new_image2=einsum(image,light,
                 'batch 1 hight weight channel ,1 l 1 1 1 ->batch l hight weight channel'
                 )
print(torch.all(new_image1=new_image2))