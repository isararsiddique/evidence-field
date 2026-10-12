"""Conventional raw-image CNN/ViT controls, for 2D and 3D inputs."""
import torch
from torch import nn

class CNN(nn.Module):
    def __init__(self,dim,channels,classes):
        super().__init__();conv=nn.Conv2d if dim==2 else nn.Conv3d;blocks=[];a=channels
        for b in [16,32,48]:
            blocks.append(nn.Sequential(conv(a,b,3,2,1,bias=False),nn.GroupNorm(8,b),nn.GELU(),conv(b,b,3,1,1,bias=False),nn.GroupNorm(8,b),nn.GELU()));a=b
        self.blocks=nn.Sequential(*blocks);self.head=nn.Linear(48,classes)
    def forward(self,x):return self.head(self.blocks(x).flatten(2).mean(-1))

class ViT(nn.Module):
    def __init__(self,dim,channels,size,classes):
        super().__init__();self.dim=dim;self.patch=7 if size==28 else 8;self.tokens=(size//self.patch)**dim
        self.embed=nn.Linear(channels*self.patch**dim,64);self.cls=nn.Parameter(torch.zeros(1,1,64));self.pos=nn.Parameter(torch.randn(1,self.tokens+1,64)*.02)
        layer=nn.TransformerEncoderLayer(64,4,128,dropout=.1,activation='gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,3,enable_nested_tensor=False);self.norm=nn.LayerNorm(64);self.head=nn.Linear(64,classes)
    def forward(self,x):
        z=x
        for axis in range(2,2+self.dim):z=z.unfold(axis,self.patch,self.patch)
        order=[0,*range(2,2+self.dim),1,*range(2+self.dim,2+2*self.dim)]
        z=z.permute(order).reshape(len(x),self.tokens,-1)
        z=self.embed(z);z=torch.cat([self.cls.expand(len(x),-1,-1),z],1)+self.pos
        return self.head(self.norm(self.encoder(z)[:,0]))
