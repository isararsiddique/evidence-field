"""Larger from-scratch Transformer control; excluded from the EF architecture."""
import torch
from torch import nn

class WideViT(nn.Module):
    def __init__(self,dim,channels,size,classes,width=128,depth=4):
        super().__init__();self.dim=dim;self.patch=8;self.tokens=(size//8)**dim
        self.embed=nn.Linear(channels*8**dim,width);self.cls=nn.Parameter(torch.zeros(1,1,width))
        self.pos=nn.Parameter(torch.randn(1,self.tokens+1,width)*.02)
        layer=nn.TransformerEncoderLayer(width,8,2*width,dropout=.1,activation='gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,depth,enable_nested_tensor=False);self.norm=nn.LayerNorm(width);self.head=nn.Linear(width,classes)
    def forward(self,x):
        z=x
        for axis in range(2,2+self.dim):z=z.unfold(axis,8,8)
        order=[0,*range(2,2+self.dim),1,*range(2+self.dim,2+2*self.dim)]
        z=z.permute(order).reshape(len(x),self.tokens,-1);z=self.embed(z)
        z=torch.cat([self.cls.expand(len(x),-1,-1),z],1)+self.pos
        return self.head(self.norm(self.encoder(z)[:,0]))
