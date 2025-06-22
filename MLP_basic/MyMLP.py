import torch 
import torch.nn as nn 
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(MLP,self).__init__()
        self.input_size = input_size
        self.f1 = nn.Linear(input_size,hidden_size)
        self.active_layer = nn.GELU() 
        self.f2 = nn.Linear(hidden_size,hidden_size)
        self.f3 = nn.Linear(hidden_size,num_classes)

    def forward(self,x):
        assert x.shape[-1] == self.input_size, f'dim of x is {x.shape[-1]} in wrong dim'
        x = self.f1(x)
        x = self.active_layer(x) 
        x = self.f2(x) 
        x = self.active_layer(x) 
        return self.f3(x)
        



    