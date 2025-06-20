import torch 
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from collections import OrderedDict
from typing import Optional

def clones(layer,N):
    r'''
    Create N identical layers

    Args:
        layer (nn.Module): The layer to be cloned.
        N (int): Number of clones to create.

    Returns:
        list: List of cloned layers.
    '''
    return nn.ModuleList([layer for _ in range(N)])

def drop_path(x, drop_prob: float = 0., training: bool = False):
    r'''
    Apply Drop Path regularization

    Args:
        x (torch.Tensor): Input tensor.
        drop_prob (float): Probability of dropping paths.
        training (bool, optional): Whether the model is in training mode. Defaults to False.

    Returns:
        torch.Tensor: Output tensor after applying Drop Path.
    '''
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1) # -> [batch_size, 1, 1, ..., 1]
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # round down
    output = x.div(keep_prob) * random_tensor  # keep the same mean
    return output

class DropPath(nn.Module):
    r'''
    Drop Path Layer for regularization

    Args:
        drop_prob (float): Probability of dropping paths.
        training (bool, optional): Whether the model is in training mode. Defaults to False.

    Returns:
        torch.Tensor: Output tensor after applying Drop Path.
    '''
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    





class PatchEmbedding(nn.Module):
    r'''
    Patch Embedding Layer for Vision Transformer (ViT)

    Args:
        img_size (int): Size of the input image (assumed square).   
        in_channels (int): Number of input channels (e.g., 3 for RGB images).
        patchsize (int): Size of the patches to be extracted from the image.
        embed_dim (int): Dimension of the embedding for each patch.
        normlayer (nn.Module, optional): Normalization layer to apply after projection. Defaults to None.

    Returns:
        torch.Tensor: Embedded patches of shape (batch_size, num_patches, embed_dim).
    '''
    def __init__(self,img_size,in_channels,patchsize,embed_dim,normlayer=None):
        super(PatchEmbedding, self).__init__()
        self.patchsize = (patchsize,patchsize)
        self.img_size = (img_size,img_size)
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patchsize, stride=patchsize)
        self.grid_size = (self.img_size[0] // self.patchsize[0], self.img_size[1] // self.patchsize[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.norm = normlayer(embed_dim) if normlayer is not None else nn.Identity()

    def forward(self, x):
        assert x.shape[2]== self.img_size[0] and x.shape[3] == self.img_size[1], \
            f"Input image size ({x.shape[2]}, {x.shape[3]}) does not match expected size ({self.img_size[0]}, {self.img_size[1]})"
        x = self.proj(x).flatten(2).transpose(1,2)
        x = self.norm(x)
        return x
    

class PositionalEncoding(nn.Module):
    r'''
    Positional Encoding Layer for Vision Transformer (ViT)

    Args:
        embed_dim (int): Dimension of the input embeddings.
        max_len (int, optional): Maximum length of the sequence. Defaults to 5000.

    Returns:
        torch.Tensor: Positional encoding tensor of shape (1, max_len, embed_dim).
    '''
    def __init__(self,embed_dim,max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / embed_dim))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

def attention(query,key,value,mask=None,dropout=None):
    r'''
    Scaled Dot-Product Attention

    Args:
        query (torch.Tensor): Query tensor of shape (batch_size, num_heads, seq_len_q, head_dim).
        key (torch.Tensor): Key tensor of shape (batch_size, num_heads, seq_len_k, head_dim).
        value (torch.Tensor): Value tensor of shape (batch_size, num_heads, seq_len_v, head_dim).
        mask (torch.Tensor, optional): Mask tensor to apply attention mask. Defaults to None.
        dropout (nn.Module, optional): Dropout layer to apply after attention. Defaults to None.

    Returns:
        torch.Tensor: Output tensor after applying attention.
    '''
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2,-1)) / d_k**0.5
    if mask is not None:
        scores = scores.masked_fill(mask==0,float('-inf'))
    attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    output = torch.matmul(attn,value)
    return output

class MultiHeadAttention(nn.Module):
    r'''
    Multi-Head Attention Layer

    Args:
        embed_dim (int): Dimension of the input embeddings.
        num_heads (int): Number of attention heads.
        dropout (float, optional): Dropout rate to apply after attention. Defaults to 0.1.

    Returns:
        torch.Tensor: Output tensor after applying multi-head attention.
    '''
    def __init__(self,embed_dim,num_heads=8,qkv_bias=False,dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3,bias=qkv_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,mask=None):
        batch_size, seq_len, _ = x.size()
        qkv = self.qkv_proj(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attn_output = attention(query,key,value,mask,self.dropout)
        attn_output = attn_output.transpose(1,2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(attn_output)
        return output
    

class MLP(nn.Module):
    r'''
    Multi-Layer Perceptron (MLP) Layer

    Args:
        embed_dim (int): Dimension of the input embeddings.
        hidden_dim (int): Dimension of the hidden layer.
        dropout (float, optional): Dropout rate to apply after each linear layer. Defaults to 0.1.

    Returns:
        torch.Tensor: Output tensor after applying MLP.
    '''
    def __init__(self,embed_dim,hidden_dim,act_layer=nn.GELU,dropout=0.1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.act_layer = act_layer()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_layer(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class LayerNorm(nn.Module):
    r'''
    Layer Normalization Layer

    Args:
        embed_dim (int): Dimension of the input embeddings.
        eps (float, optional): Small value to avoid division by zero. Defaults to 1e-6.

    Returns:
        torch.Tensor: Normalized tensor.
    '''
    def __init__(self,embed_dim,eps=1e-6):
        super(LayerNorm, self).__init__()
        self.embed_dim = embed_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embed_dim))
        self.bias = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias

class Sublayerconnection(nn.Module):
    r'''
    Sublayer Connection Layer for Vision Transformer (ViT)

    Args:
        embed_dim (int): Dimension of the input embeddings.
        dropout (float, optional): Dropout rate to apply after the sublayer. Defaults to 0.1.

    Returns:
        torch.Tensor: Output tensor after applying the sublayer connection.
    '''
    def __init__(self,embed_dim,dropout=0.1):
        super(Sublayerconnection, self).__init__()
        self.norm = LayerNorm(embed_dim)
        self.drop_path = DropPath(dropout) if dropout > 0. else nn.Identity()

    def forward(self, x, sublayer):
        x = x + self.drop_path(sublayer(self.norm(x)))
        return x
    
class EncoderLayer(nn.Module):
    def __init__(self,embed_dim,num_heads,mlp_ratio=4,qkv_bias=False,dropout=0.,act_layer=nn.GELU):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(embed_dim,num_heads,qkv_bias,dropout)
        self.mlp = MLP(embed_dim,int(embed_dim*mlp_ratio),act_layer=act_layer,dropout=dropout)
        self.sublayer = clones(Sublayerconnection(embed_dim,dropout),2)

    def forward(self, x, mask=None):
        x = self.sublayer[0](x, lambda x: self.attn(x, mask))
        x = self.sublayer[1](x, self.mlp)
        return x
    

class ViT(nn.Module):
    def __init__(self,img_size=224,
                 patchsize=16,
                 in_channels=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 qkv_bias=True,
                 representation_size=None,
                 distilled=False,
                 dropout=0.1,
                 embed_layer=PatchEmbedding,
                 norm_layer=LayerNorm,
                 act_layer=nn.GELU):
        super(ViT, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size, in_channels, patchsize, embed_dim, normlayer=norm_layer)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        dpr = [x.item() for x in torch.linspace(0, dropout, depth)]
        self.blocks = nn.Sequential(*[
            EncoderLayer(
                embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias, dropout=dropout,act_layer=act_layer
            ) for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        if representation_size and not distilled:
            self.pre_logits = nn.Sequential(
                nn.Linear(embed_dim, representation_size),
                act_layer(),
                norm_layer(representation_size)
            )
            self.num_features = representation_size
            self.has_logits = True
        else:
            self.pre_logits = nn.Identity()
            self.has_logits = False

        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None 
        if distilled:
            self.head_dist = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        nn.init.trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=.02)
        
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(_init_vit_weights)

    def forward_features(self,x):
        x = self.patch_embed(x) # -> (batch_size, num_patches, embed_dim)
        # [1,1,embed_dim] -> [batch_size,1,enbed_dim]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token,self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0]) # only return the [CLS]
        else:
            return x[:,0],x[:,1]  # return [CLS, DIST] tokens
        
    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x,x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
            return x

                

   
def _init_vit_weights(m):
    r"""
        Initialize weights for Vision Transformer (ViT) layers."""

    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m,nn.Conv2d):
        nn.init.kaiming_normal_(m.weight,mode='fan_out')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
          

