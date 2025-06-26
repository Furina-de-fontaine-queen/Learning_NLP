import math
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel

class BertEmbeddings(nn.Module):

    def __init__(self,config):
        r'''
        初始化embedding,包含word_embeddings,position_embeddings,token_type_embeddings

        ### Args:
            **config:** 
                vocab_size:  词典大小

                hidden_size: 隐藏层维度

                pad_token_id: [PAD]的对应id

                max_position_embeddings: 最大语句长度,超过截断,否则padding

                type_vocab_size:最大语句种类

                layer_norm_eps: 分母填充eps值

                hidden_dropout_prob: 丢失概率

                position_embedding_type: absolute/relative 位置编码方式

        ### Attributes:
            **word_embeddings:** 词编码(trainable)

            **position_embeddings:** 位置编码(trainable)

            **token_type_embeddings:** 语句编码(trainable)

        ### Call:
            **input_ids:** Optional[torch.LongTensor]
            **token_type_ids:** Optional[torch.LongTensor]
            **position_ids:** Optional[torch.LongTensor]
            **inputs_embeds:** Optional[torch.FloatTensor]
            **past_key_values_length:** int

            **Returns:**  torch.Tensor
        '''
        super(BertEmbeddings,self).__init__()

        self.word_embeddings = nn.Embedding(num_embeddings=config.vocab_size,
                                            embedding_dim=config.hidden_size,
                                            padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(num_embeddings=config.max_position_embeddings,
                                                embedding_dim=config.hidden_size)
        self.token_type_embeddings = nn.Embedding(num_embeddings=config.type_vocab_size,
                                                  embedding_dim=config.hidden_size)
        
        self.LayerNorm = nn.LayerNorm(config.hidden_size,config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.position_embedding_type = getattr(config, 'position_embedding_type','absolute')
        # 当未接受到posiion_ids或token_type_ids时，使用默认序列
        self.register_buffer(
            'position_ids',torch.arange(config.max_position_embeddings).expand((1,-1)),persistent=False
        )
        self.register_buffer(
            'token_type_ids',torch.zeros(self.position_ids.size(),dtype=torch.long), persistent=False
        )

    def forward(self,input_ids: Optional[torch.LongTensor]=None,  # 词在字典的编码
                token_type_ids: Optional[torch.LongTensor]=None,  # 区分句子
                position_ids: Optional[torch.LongTensor]=None,
                inputs_embeds:Optional[torch.FloatTensor]=None,   # 词嵌入向量
                past_key_values_length: int = 0,)-> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:,past_key_values_length:seq_length + past_key_values_length]
        if token_type_ids is None:
            if hasattr(self,'token_type_ids'):
                buffered_token_type_ids = self.token_type_ids[:,:seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0],seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape,dtype=torch.long,device=self.position_ids.device)
        
        # 词嵌入编码
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # segment编码
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings

        # 位置编码
        if self.position_embedding_type == 'absolute':
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings



class BertSelfAttention(nn.Module):
    def __init__(self,config,position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config,'embedding_size'):
            raise ValueError(
                f'维度不能均分到各个注意力头上.'
            )    
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size,self.all_head_size)
        self.key = nn.Linear(config.hidden_size,self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(config,'position_embedding_type','absolute')

        # 相对位置编码
        if self.position_embedding_type == 'relative_key' or self.position_embedding_type == 'relative_key_query':
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(num_embeddings=2 * config.max_position_embeddings - 1,
                                                   embedding_dim=self.attention_head_size)
        
        self.is_decoder = config.is_decoder

    def transpose_for_scores(self,x:torch.Tensor) -> torch.Tensor:
        r'''[batch_size,seq_len,hidden_size] -> [batch_size,num_heads,seq_len,head_size]'''
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,self.attention_head_size)
        x.view(new_x_shape)
        return x.permute(0,2,1,3)
    
    def forward(self,
                hidden_states: torch.Tensor,    # 输入[batch_size,seq_len,hidden_size]
                attention_mask: Optional[torch.FloatTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None,  # 屏蔽注意力[num_layers,num_heads]
                encoder_hidden_states: Optional[torch.FloatTensor] = None,
                encoder_attention_mask: Optional[torch.FloatTensor] = None,
                past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None, # (past_keys,past_values)
                output_attentions: Optional[bool] = False)->Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)
        is_cross_attention = encoder_hidden_states is not None 

        if is_cross_attention and past_key_value is not None:
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask 
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            key_layer = torch.cat([past_key_value[0],key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1],value_layer],dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None 
        

        attention_scores = torch.matmul(query_layer,key_layer.transpose(-1,-2))
        r'''
        ----------------------------------------------------------------
        相对位置编码的实现
        加入了 pos_attn_{ij} = q_i.T @ r_{ij}
        '''

        if self.position_embedding_type == 'relative_key' or self.position_embedding_type == 'relative_key_query':
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length -1,
                                              dtype=torch.long,
                                              device=hidden_states.device).view(-1,1)
            else:
                position_ids_l = torch.arange(query_length,
                                              dtype=torch.long,
                                              device=hidden_states.device).view(-1,1)
            position_ids_r = torch.arange(key_length,
                                          dtype=torch.long,
                                          device=hidden_states.device).view(1,-1)
            distance = position_ids_l - position_ids_r # [q_len,k_len]

            # [q_len,k_len,head_size]
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1) # [-max+1, max-1] → [0, 2*max-2]
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)# fp16 compatibility

            if self.position_embedding_type == 'relative_key':
                relative_position_scores = torch.einsum('bhld,lrd->bhlr',query_layer,positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == 'relative_key_query':
                # [batch_size,num_heads,seq_len,heads_size] @ [seq_len,seq_len,head_size].T -> [batch_size,num_heads,seq_len,seq_len]
                relative_position_scores_query = torch.einsum('bhld,lrd->bhlr',query_layer,positional_embedding)
                relative_position_scores_key = torch.einsum('bhrd,lrd->bhlr',key_layer,positional_embedding)
                attention_scores = attention_scores + relative_position_scores_key + relative_position_scores_query

        r'''
        ---------------------------------------------------------------
        '''

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        r'''
        ----------------------------------------------------------------
        attention_mask的处理:
        mask 的部分为 -inf,形状为[batch_size, 1, seq_len, seq_len]
        e.l.: 
        tensor([[[[ 0., -inf, -inf],
                  [ 0.,   0., -inf],
                  [ 0.,   0.,   0.]]]])
        '''
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        r'''
        -----------------------------------------------------------------
        '''
        attention_probs = nn.functional.softmax(attention_scores,dim=-1)
        attention_scores = self.dropout(attention_probs)

        r'''
        -----------------------------------------------------------------
        head_mask的处理:
        通过 0 来屏蔽特定头的输出,形状为[batch_size,num_head,1,1]
        '''
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        r'''
        ------------------------------------------------------------------
        '''
        context_layer = torch.matmul(attention_probs,value_layer)

        content_layer = context_layer.permute(0,2,1,3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        content_layer = content_layer.view(new_context_layer_shape)
        outputs = (content_layer,attention_probs) if output_attentions else (content_layer,)

        
        return outputs


class BertSdpaSelfAttention(nn.Module):
    pass

BERT_SELF_ATTENTION_CLASSES = {
    'eager': BertSelfAttention,
    'sdpa': BertSdpaSelfAttention,
}

class BertSelfOutput(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size,eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self,hidden_states:torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
class BertAttention(nn.Module):
    def __init__(self,config,position_embedding_type=None):
        super().__init__() 
        self.self = BERT_SELF_ATTENTION_CLASSES[config._attn_implementation](
            config,position_embedding_type=position_embedding_type
        )
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()


    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.FloatTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None,
                encoder_hidden_states: Optional[torch.FloatTensor] = None,
                encoder_attention_mask: Optional[torch.FloatTensor] = None,
                past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
                output_attentions: Optional[bool] = False)->Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        ) # 这里输出为 (content_layer,) / (content_layer, score_prob)

        attention_output = self.output(self_outputs[0],hidden_states) # 残差链接
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

class BertEncoder(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config 
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)]) 
        self.gradient_checkpointing = False

    def forward(self,
                hidden_states: torch.Tensor,    
                attention_mask: Optional[torch.FloatTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None,  # 用于将某些层的某些注意力计算无效化
                encoder_hidden_states: Optional[torch.FloatTensor] = None, #这一部分在BertModel配置为decoder时起作用，将执行cross-attention而不是self-attention
                encoder_attention_mask: Optional[torch.FloatTensor] = None, # 在cross-attention中用于标记encoder端输入的padding
                past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
                output_attentions: Optional[bool] = False,
                output_hidden_states: Optional[bool] = False,
                return_dict: Optional[bool] = True):
        for i , layer_module in enumerate(self.layer):

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func()
            else:
                layer_outputs = layer_module()



class BertLayer(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder 
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError('交叉注意力必须要作为decoder时使用')
            self.crossattention = BertAttention(config,position_embedding_type='absolute')
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)    
class BertPooler(nn.Module):
    pass 

class BertModel(BertPreTrainedModel):

    def __init__(self,config,add_pooling_layer=True):
        self.config = config 

        self.embeddings = BertEmbeddings(config)

        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.attn_implementation = config._attn_implementation
        self.position_mbedding_type = config.position_embedding_type

        self.post_init()

    
    def forward(self,
                input_ids:Optional[torch.Tensor]=None,
                attention_mask:Optional[torch.Tensor]=None,
                token_type_ids:Optional[torch.Tensor]=None,
                position_ids: Optional[torch.Tensor]=None,
                head_mask: Optional[torch.Tensor]=None,
                inputs_embeds: Optional[torch.Tensor]=None,
                encoder_hidden_states: Optional[torch.Tensor]=None,
                encoder_attention_mask: Optional[torch.Tensor]=None,
                past_key_values: Optional[list[torch.FloatTensor]] = None,
                output_attentions: Optional[bool] = None,
                ):
        # 准备工作：
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('input_ids 与 inputs_embeds 不能同时出现')
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1] # input_shape [batch_size,seq_len]
        else:
            raise ValueError('input_ids 与 inputs_embeds 不能都没有')
        
        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values处理
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0


        # 默认token_type_ids的设置
        if token_type_ids is None:
            if hasattr(self.embeddings, 'token_type_ids'):
                buffer_token_type_ids = self.embeddings.token_type_ids[:,:seq_length]
                buffer_token_type_ids_expand = buffer_token_type_ids.expand(batch_size,seq_length)
                token_type_ids = buffer_token_type_ids_expand
            else:
                token_type_ids = torch.zeros(input_shape,dtype=torch.long,device=device)





        # step 1: 词嵌入层
        embedding_output = self.embeddings(
            input_ids = input_ids,
            position_ids = position_ids,
            token_type_ids = token_type_ids,
            inputs_embeds = inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        

        # 是否使用SDPA算法以及是否为decoder的设置
        if attention_mask is None:
            attention_mask = torch.ones((batch_size,seq_length + past_key_values_length),device=device)
        
        # 利用PyTorch 2.x的优化注意力计算，提升30%以上速度
        use_sdpa_attention_masks = (
            self.attn_implementation == 'sdpa' 
            and self.position_embedding_type == 'absolute'
            and head_mask is None      # 无特殊头部掩码
            and not output_attentions  # 不需要输出注意力权重
        )
        if use_sdpa_attention_masks and attention_mask.dim() == 2:
            pass 
        else:
            extended_attention_mask = self.get_extended_attention_mask(attention_mask=attention_mask,
                                                                       input_shape=input_shape)
            
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length,_ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size,encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape,device=device)

            if use_sdpa_attention_masks and encoder_attention_mask.dim() == 2:
                pass
            else:
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask=encoder_attention_mask)

        else:
            encoder_extended_attention_mask = None    



        

        # step 2: encoder layers层
        encoder_output = self.encoder()