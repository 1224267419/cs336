import torch 
from einops import rearrange,einops

batch ,sequence, d_in, d_out=3,4,5,6
D=torch.randn(batch, sequence, d_in)
A=torch.randn(d_in, d_out)
y1=D@A
y2=torch.einsum('bsd,df->bsf',D,A)
print(torch.allclose(y1,y2))


batch,height,width,channels=64,128,128,3
images=torch.randn(batch, height, width, channels)
#0-1 10个step
dim_by=torch.linspace(start=0, end=1.0, steps=10)
#多个dim做广播
dim_value=rearrange(dim_by,"dim ->1 dim 1 1 1")
images1=rearrange(images,"b h w c -> b 1 h w c")
y1=images1*dim_value

y2=torch.einsum(
    "b h w c,d -> b d h w c",
    images,dim_by,
    )

print(torch.allclose(y1,y2))