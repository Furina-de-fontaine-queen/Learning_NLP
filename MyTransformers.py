import numpy as np 
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import math, copy , time 
from torch.autograd import Variable
import matplotlib.pyplot as plt 

class Encoder(nn.Module):
    r'''
    整个encoder层.包含N个layers和最后的归一化输出
    ### Args:
        **layer (nn.Module):** 每一个encoder的layer
        **norm (nn.Module):** 归一化层
    ### Methods:
        **forward(x,mask):** 将x进入每一层layers参与计算最后归一化输出
    '''
    def __init__(self,layer,N):
        super(Encoder,self).__init__()
        self.layers = clones(layer,N)
        self.norm = LayerNorm(layer.size)

    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)



class EncoderDecoder(nn.Module):
    def __init__(self,encoder,decoder,src_embed,tgt_embed,generator):
        super(EncoderDecoder,self).__init__()
        self.encoder = encoder
        self.decoder = decoder 
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self,src,tgt,src_mask,tgt_mask):
        return self.decode(memory=self.encode(src,src_mask),
                           src_mask=src_mask,
                           tgt=tgt,
                           tgt_mask=tgt_mask)
    
    def decode(self,memory,src_mask,tgt,tgt_mask):
        return self.decoder(self.tgt_embed(tgt),memory,src_mask,tgt_mask)

    def encode(self,src,src_mask):
        return self.encoder(self.src_embed(src),src_mask)
    

class Embeddings(nn.Module):
    r'''
    将batched词向量映射到为总词汇向量的一个向量.

    

    ### **Methods:**

    **forward(self,x):**

        **Args:**
        x: torch.Tensor 形状为 torch.Size([batch_size, max_seq_len])

        **Renturn:**
        embedding_x: 经过缩放处理的向量 形状为 torch.Size([max_seq_len, d_model])

    ### **Examples**
    ```python
    >>> from MyTransformers import Embeddings
    
    >>> # 创建实例化
    >>> embeddings = Embeddings(vocab_size=7,
                        d_model=8)

    >>> # 获取结果
    >>> x = torch.Tensor([[6, 6, 0, 0],
        [2, 5, 5, 5]]).to(torch.int32)
    >>> print(embeddings.forward(x))

    tensor([[[-2.5468,  1.0744, -0.4898, -1.3988, -0.7323,  2.1291,  2.3446,
           1.0439],
         [-2.5468,  1.0744, -0.4898, -1.3988, -0.7323,  2.1291,  2.3446,
           1.0439],
         [ 0.5055, -3.7903, -5.9608, -2.8386, -2.5474, -1.7102,  2.0484,
          -5.1415],
         [ 0.5055, -3.7903, -5.9608, -2.8386, -2.5474, -1.7102,  2.0484,
          -5.1415]],

        [[ 0.9090, -0.1158,  2.5857,  0.5590,  0.0312,  3.1589, -1.1859,
          -2.2100],
         [ 0.8541, -1.1290, -1.6136,  3.2253,  1.6085,  2.2570, -2.1074,
           0.8792],
         [ 0.8541, -1.1290, -1.6136,  3.2253,  1.6085,  2.2570, -2.1074,
           0.8792],
         [ 0.8541, -1.1290, -1.6136,  3.2253,  1.6085,  2.2570, -2.1074,
           0.8792]]], grad_fn=<MulBackward0>)
    ```
    '''
    def __init__(self,vocab_size,d_model):
        super(Embeddings,self).__init__()
        self.embedding_table = nn.Embedding(vocab_size,d_model) 
        self.d_model = d_model 

    def forward(self,x):
        return self.embedding_table(x) * np.sqrt(self.d_model)
    


class PositionalEncoding(nn.Module):
    r'''
    将词嵌入 (FloatTensor)进行位置编码,使模型能运用序列的位置信息

    ### **method:**
        **forward(x):**
        *Args:*
            x: FloatTensor, 大小为[batch_size,max_len,d_model]
        *Returns*:
            y: FloatTensor, 大小为[batch_size,max_len,d_model]
    
    ### **examples**
    
    ```python
    >>> from MyTransforers import PositionalEncoding

    >>> # 实例化位置编码
    >>> pos_encoder = PositionalEncoding(d_model=8,
                                     dropout=0.1,
                                     max_len=5)

    >>> # 创建 [batch_size,max_len,d_model] 的词嵌入
    >>> word_embed = torch.randn(2,5,8)
     
    >>> # 获得结果(词嵌入+位置编码)
    >>> output = pos_encoder(word_embed)

    >>> # 查看结果
    >>> print(output[:, 0, :])
    tensor([[ 0.0000,  2.6870,  0.7327,  0.0000, -0.0149,  0.5257, -0.3650, -1.0921],
        [-2.0867, -0.0000,  1.0031,  2.8096,  0.4723,  1.1739, -1.5207,  1.6692]])
    >>> print(output.shape)
    torch.Size([2, 5, 8])
    ```
 
    '''
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]
        
    def forward(self, x):
        # x: 词嵌入张量 [batch_size, max_len, d_model]
        x = x + Variable(self.pe[:,:x.size(1)],requires_grad=False) # type: ignore
        return self.dropout(x)


def clones(module,N:int)->nn.ModuleList:
    return nn.ModuleList([module for _ in range(N)])

class LayerNorm(nn.Module):
    r'''
    层归一化模块,是每一层都满足~N(0,1),防止多层叠加导致的梯度爆炸,同时稳定反向传播

    ### Args:
        **features (int):** 传入张量对应向量的空间维度
        **eps (float):** 主要防止分母为零的尴尬

    ### Method:
        **forward(x):** 归一化,公式为
        output = a_2 * (x - mean(x))/(std(x) + eps)  + b_2
    '''
    def __init__(self,features,eps=1e-6):
        super(LayerNorm,self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps 

    def forward(self,x):
        mean = x.mean(-1,keepdim=True)
        std = x.std(-1,keepdim=True)
        return self.a_2 * (x - mean)/ (std + self.eps) + self.b_2
    

class SublayerConnection(nn.Module):
    r'''
    残差连接 + LayerNorm

    ### Arg:
        **size (int):** 输入张量对应向量所在空间的维度
        **dropout (float):** 丢失概率

    ### Method:
        **forword(x,sublayer):**
            **sublayer (nn.Module):** 选择的layer(attn 或者 FFN)
            公式为 output = x + Dropout(Layer(norm(x)))
    '''
    def __init__(self,size,dropout):
        super(SublayerConnection,self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

def subsequent_mask(size):
    attn_shape = (1,size,size)
    subsequent_mask = np.triu(np.ones(attn_shape),k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query,key,value,mask=None,dropout=None):
    r"""
    实现缩放点积注意力机制
    
    ### Args:

        注: 用于单头注意力时,无需加入num_heads

        query  : 查询向量 [batch_size, num_heads, seq_len_q, d_k]

        key    : 键向量   [batch_size, num_heads, seq_len_k, d_k]

        value  : 值向量   [batch_size, num_heads, seq_len_v, d_v] 
                注意: seq_len_k = seq_len_v
        mask   : (可选) 掩码张量 [batch_size,num_heads,seq_len_q,seq_len_k]
                用于屏蔽无效位置(如填充位置)
        dropout: (可选) dropout层实例
        
    ### Return:

        output : 注意力加权的值向量 [batch_size, num_heads, seq_len_q, d_v]

        attn   : 注意力权重矩阵 [batch_size, num_heads, seq_len_q, seq_len_k]

    ### Examples:
    #### Example 1: 
    ```python
    >>> from MyTransformers import attention

    >>> # 创建输入数据 (batch_size=2, seq_len=3, d_model=4)
    >>> query = key = value = torch.randn(2, 3, 4)
    
    >>> # 创建padding mask:第一个序列全有效,第二个序列最后一个位置是padding
    >>> mask = torch.tensor([
        [[1, 1, 1],  # 序列1:所有位置有效
         [1, 1, 1],
         [1, 1, 1]],
        
        [[1, 1, 0],  # 序列2:最后一个位置是padding
         [1, 1, 0],
         [1, 1, 0]]
                   ], dtype=torch.long)
    
    >>> # 应用注意力机制
    >>> output, attn_weights = attention(query, key, value, mask=mask)  
    tensor([[[ 1.9051,  1.4421,  0.8838, -2.0934],
            [ 0.9810, -0.4575,  0.1663, -1.5649],
            [ 0.7209, -0.5860, -0.1027, -0.8587]],

            [[-1.0970,  0.0831, -2.2629, -0.2238],
            [-0.4745, -0.2955,  0.1618, -0.5384],
            [-0.9742,  0.0084, -1.7846, -0.2859]]])  
    ```
    #### Example 2:
    ```python
    >>> # 创建输入数据 (seq_len=3, d_model=4)
    >>> query = key = value = torch.randn(1, 3, 4)  # 单样本
    
    >>> # 创建sequence mask(上三角矩阵)
    >>> mask = torch.tril(torch.ones(3, 3))  # 下三角为1,上三角为0
    >>> mask = mask.unsqueeze(0)  # 增加batch维度
    tensor([[1, 0, 0],
        [1, 1, 0],
        [1, 1, 1]], dtype=torch.int32)

    >>> # 应用注意力机制
    >>> output, attn_weights = attention(query, key, value, mask=mask)  
    tensor([[[ 1.3525,  0.6863, -0.3278,  0.7950],
         [ 0.7698,  0.3434,  0.1350,  0.2327],
         [ 0.5742,  0.4796, -0.0098,  0.5302]]])
    ``` 
    """
    d_k = query.size(-1)
    scores = torch.matmul(query,key.transpose(-2,-1) / math.sqrt(d_k))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores,dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn,value), p_attn

class MultiHeadedAttention(nn.Module):
    r'''
    多头注意力机制。将d_model 又分为h个维度为d_model / h 的子空间，分别进行注意力计算

    ### Args:
        **h (int):** n_head

        **d_model (int):** (k/q子空间维度)

        **dropout (float):** 抑制活动的概率
    ### Notice: 
        d_model/h必须为整数

    '''
    def __init__(self,h,d_model,dropout=0.1):
        super(MultiHeadedAttention,self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h 
        self.h = h 
        self.linears = clones(nn.Linear(d_model,d_model),4)
        self.attn = None 
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,query,key,value,mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [l(x).view(nbatches,-1,self.h,self.d_k).transpose(1,2) 
                             for l,x in zip(self.linears,(query,key,value))]
        
        x,self.attn = attention(query,key,value,mask=mask,dropout=self.dropout)

        x = x.transpose(1,2).contiguous().view(nbatches,-1,self.h*self.d_k)
        return self.linears[-1](x)
    
class PositionwiseFeedForward(nn.Module):

    def __init__(self,d_model,d_ff,dropout=0.1):
        super(PositionwiseFeedForward,self).__init__()
        self.w_1 = nn.Linear(d_model,d_ff)
        self.w_2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(p=dropout) 

    def forward(self,x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
    
class EncoderLayer(nn.Module):
    r'''
    结合子注意力函数与前馈神经网络,有两个子层,第一层使用子注意力函数,再将结果传给前馈神经网络

    ### Args:
        **size (int):** 传入张量对应的向量的维度,即d_model
        **self_attn (function):** 子注意力函数
        **feed_forward (function):** FNN 函数
        **dropout (float):** 丢失概率
    
    ### Method:
        **forward(x,mask)**: 
            第一层: output1 = x + Dropout(self_attn(norm(x)))
            第二层: output2 = x + Dropout(feed_forward(norm(x)))
    '''
    def __init__(self,size,self_attn,feed_forword,dropout):
        super(EncoderLayer,self).__init__()
        self.self_attn = self_attn 
        self.feed_forword = feed_forword 
        self.sublayer = clones(SublayerConnection(size,dropout),2)
        self.size = size  

    def forward(self,x,mask):
        x = self.sublayer[0](x , lambda x: self.self_attn(x,x,x,mask))
        return self.sublayer[1](x, self.feed_forword)


class Decoder(nn.Module):
    r'''
    整个decoder层.包含N个layers和最后的归一化输出
    ### Args:
        **layer (nn.Module):** 每一个decoder的layer
        **norm (nn.Module):** 归一化层
    ### Methods:
        **forward(x,mask):** 将x进入每一层layers参与计算最后归一化输出
    '''
    def __init__(self,layer,N):
        super(Decoder,self).__init__()
        self.layers = clones(layer,N)
        self.norm = LayerNorm(layer.size)

    def forward(self,x,memory,src_mask,tgt_mask):
        for layer in self.layers:
            x = layer(x,memory,src_mask,tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    r'''
    建立三个子层,完成一次decoder工作

    ### Args:
        **size(int):** 输入张量对应的向量维度
        **self_attn (nn.Module):** 子注意力函数
        **src_attn (nn.Module):** 交叉注意力函数
        **feed_forward (nn.Module):** FFN 函数
        **dropout (float):**  丢失概率

    ### Method:
        **forward(x,memory,src_mask,tgt_mask):** 
            x : 输入张量
            memory: encoder 的输出张量
            src_mask: 交叉注意力掩码
            tgt_mask: 子注意力掩码
    '''
    def __init__(self,size,self_attn,src_attn,feed_forward,dropout):
        super(DecoderLayer,self).__init__()
        self.size = size 
        self.self_attn = self_attn 
        self.src_attn = src_attn 
        self.feed_forward = feed_forward 
        self.sublayer = clones(SublayerConnection(size,dropout),3)

    def forward(self,x,memory,src_mask,tgt_mask):
        m = memory
        x = self.sublayer[0](x,lambda x: self.self_attn(x,x,x,tgt_mask))
        x = self.sublayer[1](x,lambda x: self.src_attn(x,m,m,src_mask))
        return self.sublayer[2](x,self.feed_forward)


class Generator(nn.Module):
    r'''建立由d_model维空间的向量到vocab_size维空间的映射参数矩阵,然后归一化为logits'''
    def __init__(self,d_model,vocab_size):
        super(Generator,self).__init__() 
        self.proj = nn.Linear(d_model,vocab_size)

    def forward(self,x):
        return F.log_softmax(self.proj(x),dim=-1)
    
def make_model(src_vocab, tgt_vocab, N = 6, d_model = 512, d_ff = 2048, h=8, dropout=0.1):
    c = copy.deepcopy 
    attn = MultiHeadedAttention(h,d_model)
    ff = PositionwiseFeedForward(d_model,d_ff,dropout)
    position = PositionalEncoding(d_model,dropout)
    model = EncoderDecoder(Encoder(EncoderLayer(d_model,c(attn),c(ff),dropout),N),
                           Decoder(DecoderLayer(d_model,c(attn),c(attn),c(ff),dropout),N),
                           nn.Sequential(Embeddings(src_vocab,d_model),c(position)),
                           nn.Sequential(Embeddings(tgt_vocab,d_model),c(position)),
                           Generator(d_model,tgt_vocab))
    # 初始化参数    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class Batch:
    def __init__(self,src,trg=None,pad=0):
        self.src = src 
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:,:-1]
            self.trg_y = trg[:,1:]
            self.trg_mask = self.make_std_mask(self.trg,pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt,pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


def run_epoch(data_iter,model,loss_compute):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i , batch in enumerate(data_iter):
        out = model.forward(batch.src,batch.trg,
                            batch.src_mask,batch.trg_mask)
        loss = loss_compute(out,batch.trg_y,batch.ntokens)
        total_loss +=loss 
        total_tokens +=batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print('Epoch Step: %d Loss: %f Tokens per Sec: %f' % (i,loss / batch.ntokens,
                                          tokens /  elapsed))
            tokens = 0
    return total_loss / total_tokens


global max_src_in_batch, max_tgt_in_batch # 都是记录当前最长序列长度,包含<s>与</s>
def batch_size_fn(new,count,sofar):
    r'''
    动态扩展批次并计算总token数
    '''
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch, max_tgt_in_batch = 0,0
    max_src_in_batch = max(max_src_in_batch,len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements,tgt_elements)

class NoamOpt:
    def __init__(self,model_size,factor,warmup,optimizer):
        self.optimizer = optimizer
        self._step = 0 
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size 
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate() 
        for p in self.optimizer.param_groups:
            p['lr'] = rate 
        self._rate = rate 
        self.optimizer.step()

    def rate(self,step=None):
        if step is None:
            step = self._step 
        return self.factor * (self.model_size ** (-0.5)) * min(step ** (-0.5), step * self.warmup ** (-1.5))
    
def get_std_opt(model):
    return NoamOpt(model_size=model.src_embed[0].d_model,
                   factor=2,
                   warmup=4000,
                   optimizer=torch.optim.Adam(params=model.parameters(),
                                              lr=0,
                                              betas=(0.9,0.98),
                                              eps=1e-9))

class LabelSmoothing(nn.Module):
    r'''
    平滑函数,负责根据target将原本得到[0,..,1,...0]变为[0,x,..,confidence,..x],降低模型训练效果但是提高准确率.
    处理后带入KLDivLoss中计算.注意predict值一定要对数处理
    '''
    def __init__(self,size,padding_idx,smoothing=0.0):
        super(LabelSmoothing,self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing 
        self.smoothing = smoothing
        self.size = size 
        self.true_dist = None 

    def forward(self,x,target):
        assert x.size(1) == self.size 
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing/(self.size -2))
        true_dist.scatter_(1,target.data.unsqueeze(1).long(),self.confidence)
        true_dist[:,self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0,mask.squeeze(),0.0)
        self.true_dist = true_dist
        return self.criterion(x,Variable(true_dist,requires_grad=False))
    

class SimpleLossCompute:
    r'''
    简单的损失计算函数,将预测值与target进行比较,计算损失并返回

    ### Args:
        **generator (nn.Module):** 生成器
        **criterion (nn.Module):** 损失函数
        **opt (NoamOpt):** 优化器

    ### Method:
        **forward(pred,target,ntokens):** 计算损失并返回
    '''
    def __init__(self,generator,criterion,opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self,x,y,norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1,x.size(-1)),
                              y.contiguous().view(-1) / norm)
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.item() * norm