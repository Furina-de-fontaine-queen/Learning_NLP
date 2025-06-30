import math 
import os 
import inspect
from typing import Optional, Tuple, Callable,Union
import torch 
import torch.utils.checkpoint
from torch import nn 
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss 
from transformers.modeling_utils import PreTrainedModel 
from transformers.configuration_utils import PretrainedConfig
from transformers.activations import ACT2FN
from transformers.pytorch_utils import apply_chunking_to_forward
class MyBertConfig(PretrainedConfig):
    r"""
    This is the configuration object of my traditional bert. 

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`BertModel`] or [`TFBertModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`BertModel`] or [`TFBertModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
            positional embeddings use `"absolute"`.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.
        num_labels (`int`, defaults to 2)

    Examples:

    ```python
    >>> from transformers import BertConfig, BertModel

    >>> # Initializing a BERT google-bert/bert-base-uncased style configuration
    >>> configuration = BertConfig()

    >>> # Initializing a model (with random weights) from the google-bert/bert-base-uncased style configuration
    >>> model = BertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "bert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        classifier_dropout=None,
        num_labels=2,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.classifier_dropout = classifier_dropout
        self._attn_implementation = 'eager'

config = MyBertConfig()
class MyBertTraditionalPositionalEncoding(nn.Module):
    r'''
    Positional embedding using superposition Trigonometric Functions with different 
    periods


    Arg:
        position_ids (`torch.LongTensor`, *Optional*, defaults to None):
            Mapping of each tokens' positions
    '''
    def __init__(self,config:MyBertConfig):
        super().__init__() 
        self.pe = torch.zeros(config.max_position_embeddings,
                              config.hidden_size)
        position = torch.arange(0,config.max_position_embeddings).unsqueeze(1).float()
        assert position.shape == (config.max_position_embeddings,1)
        div_term = torch.exp(torch.arange(0,config.hidden_size,2).float() *
                             -(torch.log(torch.tensor(10000.0)) / config.hidden_size))
        assert div_term.shape == (config.hidden_size,)
        self.pe[:,0::2] = torch.sin(position * div_term)
        self.pe[:,1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)
        assert self.pe.shape == (1,config.max_position_embeddings,config.hidden_size)

    def forward(self,position_ids:Optional[torch.LongTensor] = None) -> torch.Tensor:
        return self.pe[:,:position_ids.size(1),:]
class MyBertEmbeddings(nn.Module):
    r'''
    Using three trainable embeddings to transform ids (which shapes are 
    [batch_size,seq_len]) into vector in dim=hidden_size space.

    position_ids defaults to natural sequence with size of 
    [1,`config.max_position_embeddings`]

    token_type_ids defaults to `torch.zeros` with size of position_ids

    you can only set input_ids or inputs_embeds in one call

    Args:
        inputs_ids (`torch.LongTensor`, *Optional*, defaults to None):
            Mapping sequence of tokens in the dictionary
        token_type_ids (`torch.LongTensor`, *Optional*, defaults to None):
            Mapping of each tokens to their corresponding sentences
        position_ids (`torch.LongTensor`, *Optional*, defaults to None):
            Mapping of each tokens' positions
        inputs_embeds (`torch.FloatTensor`, *Optional*, default to None):
            Mapping Vectors sequence of tokens in hidden space
    '''
    def __init__(self,config:MyBertConfig):
        super().__init__()
        self.word_embedding = nn.Embedding(num_embeddings=config.vocab_size,
                                           embedding_dim=config.hidden_size,
                                           padding_idx=config.pad_token_id)
        self.position_embedding = nn.Embedding(
            num_embeddings=config.max_position_embeddings,
            embedding_dim=config.hidden_size)
        self.token_type_embedding = nn.Embedding(
            num_embeddings=config.type_vocab_size,
            embedding_dim=config.hidden_size)
        
        self.traditional_position_embedding = MyBertTraditionalPositionalEncoding(config)
        self.layerNorm = nn.LayerNorm(normalized_shape=config.hidden_size,
                                      eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

        self.position_embedding_type = getattr(config, 
                                               'position_embedding_type',
                                               'absolute')
        self.register_buffer(
            name='position_ids',
            tensor=torch.arange(config.max_position_embeddings).expand((1,-1)),
            persistent=False
        )
        self.register_buffer(
            name='token_type_ids',
            tensor=torch.zeros(self.position_ids.size(),dtype=torch.long),
            persistent=False
        )
        
    def forward(self,
                inputs_ids: Optional[torch.LongTensor]=None,
                token_type_ids: Optional[torch.LongTensor] = None,
                position_ids: Optional[torch.LongTensor]=None,
                inputs_embeds: Optional[torch.FloatTensor] = None)->torch.Tensor:
        if inputs_ids is not None:
            input_shape = inputs_ids.size()
        else:
            assert inputs_embeds is not None, 'inputs_ids and inputs_embeds are all None'
            input_shape = inputs_embeds.size()[:-1]
        
        seq_len = input_shape[1]
        position_ids = self.position_ids[:,:seq_len]
        if token_type_ids is None:
            if hasattr(self,'token_type_ids'):
                buffered_token_type_ids = self.token_type_ids[:seq_len]
                buffered_token_type_ids =buffered_token_type_ids.expand(input_shape[0],
                                                                        seq_len)
            else:
                token_type_ids = torch.zeros(input_shape,dtype=torch.long,
                                             device=self.position_ids.device)
        # Word Embeddings:        
        if inputs_embeds is None:
            inputs_embeds = self.word_embedding(inputs_ids)
        # Position Embeddings:
        if self.position_embedding_type == 'absolute':
            position_embeddings = self.position_embedding(position_ids)
        elif self.position_embedding_type == 'traditional':
            position_embeddings = self.traditional_position_embedding(position_ids)
        
        # Segment Embeddings:
        token_type_embeddings = self.token_type_embedding(token_type_ids)
        
        embeddings = inputs_embeds + token_type_embeddings + position_embeddings
        embeddings = self.layerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MyBertSelfAttention(nn.Module):
    r"""Calculate attention to get internal information in sequence.

    You can ensure position_embed_type in instantiation step, defaults to 'absolute'.

    Args:
        hidden_states (torch.Tensor): Tensor to be calculated attention.
        attention_mask (torch.FloatTensor, optional): Mask positions with value `-inf`. Defaults to None.
    """
    def __init__(self,config:MyBertConfig,position_embed_type=None):
        super().__init__()

        assert config.hidden_size % config.num_attention_heads == 0, 'Can not divide the subspace eqaully.'
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = int(self.num_attention_heads * self.attention_head_size) 
        
        self.Query = nn.Linear(config.hidden_size,self.all_head_size)
        self.Key = nn.Linear(config.hidden_size,self.all_head_size)
        self.Value = nn.Linear(config.hidden_size,self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embed_type = position_embed_type or getattr(config,
                                                                  'position_embedding_type',
                                                                  'absolute')
        if self.position_embed_type == 'relative_key' or self.position_embed_type == 'relative_key_query':
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(num_embeddings=2*self.max_position_embeddings -1,
                                                   embedding_dim=self.attention_head_size)
    
    def transpose_for_scores(self,x:torch.Tensor) -> torch.Tensor:
        r'''
        [batch_size, seq_len, hidden_size] -> [batch_size, num_heads, seq_len, attn_head_size]
        '''
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0,2,1,3)
    

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.FloatTensor] = None,
                ) -> Tuple[torch.Tensor]:
        r'''    
        Args:
            hidden_states (torch.Tensor): Tensor to be calculated attention.
            attention_mask (torch.FloatTensor, optional): Mask positions with value `-inf`. Defaults to None.
        '''
        query = self.transpose_for_scores(self.Query(hidden_states))
        key = self.transpose_for_scores(self.Key(hidden_states))
        value = self.transpose_for_scores(self.Value(hidden_states))

        attention_scores = torch.matmul(query,key.transpose(-1,-2))
        assert attention_scores.shape[-2:] == (query.shape[2], key.shape[2]),f'attention score should be shape as {(query.shape[2], key.shape[2])} while shape as {attention_scores.shape[-2:]}'

        if (self.position_embed_type == 'relative_key') or \
            (self.position_embed_type == 'relative_key_query'):
            q_len, k_len = query.shape[2], key.shape[2]
            position_ids_q = torch.arange(q_len,
                                          dtype=torch.long,
                                          device=hidden_states.device).view(-1,1) 
            position_ids_k = torch.arange(k_len,
                                          dtype=torch.long,
                                          device=hidden_states.device).view(1,-1)
            distance = position_ids_q - position_ids_k
            assert distance.shape == (q_len,k_len)


            # all ids it be embeded should positive
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings -1)
            assert positional_embedding.shape == (q_len,k_len,self.attention_head_size)
            
            positional_embedding = positional_embedding.to(dtype=query.dtype)

            if self.position_embed_type == 'relative_key':
                relative_pos_scores = torch.einsum('bhqd,qkd->bhqk',query,positional_embedding)
                attention_scores = attention_scores + relative_pos_scores
            elif self.position_embed_type == 'relative_key_query':
                relative_pos_scores_query = torch.einsum('bhqd,qkd->bhqk',query,positional_embedding)
                relative_pos_scores_key = torch.einsum('bhkd,qkd->bhqk',key,positional_embedding)
                attention_scores = attention_scores + relative_pos_scores_key + relative_pos_scores_query
        
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = nn.functional.softmax(attention_scores,dim=-1)
        attention_scores = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs,value)
        context_layer = context_layer.permute(0,2,1,3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        
        outputs = (context_layer,)

        return outputs
    

BERT_SELF_ATTENTION_CLASSES = {'eager': MyBertSelfAttention}

class MyBertSelfOutput(nn.Module):
    r'''Combining informations from different heads.

    Args:
        hidden_states (`torch.Tensor`):
            Tensor after self attention
        input_tensor (`torch.Tensor`):
            Tensor for residual connection, defaults to be same as hidden_states
    '''
    def __init__(self,config:MyBertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size,config.hidden_size)
        self.layerNorm = nn.LayerNorm(config.hidden_size,eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self,
                hidden_states:torch.Tensor,
                input_tensor:torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layerNorm(hidden_states + input_tensor)
        return hidden_states
    
class MyBertAttention(nn.Module):
    r'''Calculate the attention output of the tensor and do residual connection

    You can customize the position_embedding_type in instantiation step, defaults to 'absolute'

    Args:
        hidden_states (`torch.Tensor`):
            tensor to calculate attention
        attention_mask (`torch.FloatTensor`, *Optional*, defaults to None):
            record positions of the mask by `-'inf'`
    '''
    def __init__(self,config:MyBertConfig,position_embdding_type=None):
        super().__init__() 

        self.attention = BERT_SELF_ATTENTION_CLASSES[config._attn_implementation](
            config,position_embed_type=position_embdding_type
        )

        self.output = MyBertSelfOutput(config)
        
    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.FloatTensor] = None,
                )->Tuple[torch.Tensor]:
        
        self_outputs = self.attention(hidden_states,
                                      attention_mask)
        attention_output = self.output(self_outputs[0],hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs
    
class MyBertIntermediate(nn.Module):
    r'''Combining informations from different feature subspaces

    Notices that Intermediate layer will returns tensor with dim=intermediate_size
    Args:
        hidden_states (`torch.Tensor`):
            Tensor after Attention and residual connection
    '''
    def __init__(self,config:MyBertConfig):
        super().__init__() 
        self.dense = nn.Linear(config.hidden_size,config.intermediate_size)
        if isinstance(config.hidden_act,str):
            self.act_layer = ACT2FN[config.hidden_act]
        else:
            self.act_layer = nn.ReLU()

    def forward(self,hidden_states:torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act_layer(hidden_states)
        return hidden_states
    
class MyBertOutput(nn.Module):
    r'''Transform tensor from intermediate layer

    Args:
        hidden_states (`torch.Tensor`):
            Tensor after intermediate layer
        input_tesnor (`torch.Tensor`):
            Tensor to do residual connection
    '''
    def __init__(self,config:MyBertConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size,config.hidden_size)
        self.layerNorm = nn.LayerNorm(config.hidden_size,eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self,
                hidden_states:torch.Tensor,
                input_tensor:torch.Tensor,
                )-> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layerNorm(hidden_states + input_tensor)
        return hidden_states
    

class MyBertLayer(nn.Module):
    r'''A Encoder layer with attention layer, intermediate layer, output layer

    Args:
        hidden_states (`torch.Tensor`):
            input of the encoder layer
        attention_mask (`torch.FloatTensor`, *Optional*, defaults to None):
            record the position of mask by `-'inf'`
    '''
    def __init__(self,config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward 
        self.seq_len_dim = 1
        self.attention = MyBertAttention(config)
        self.intermediate = MyBertIntermediate(config)
        self.output = MyBertOutput(config)

    def forward(self,
                hidden_states:torch.Tensor,
                attention_mask: Optional[torch.FloatTensor] = None,
                ) -> Tuple[torch.Tensor]:
        self_attention_outputs = self.attention(hidden_states,attention_mask)
        attention_output = self_attention_outputs[0]

        # divide the attention_output into small chunk in seq_len dim to save vm size
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.seq_len_dim,
            attention_output
        )

        outputs = (layer_output,) 
        return outputs


    def feed_forward_chunk(self,attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output,attention_output)
        return layer_output

class MyBertEncoder(nn.Module):
    r'''Stacked multi-encoder layers
    Args:
        hidden_states (`torch.Tensor`):
            input of the Encoderlayers
        attention_mask (`torch.FloatTensor`,*Optional*,defaults to None):
            record the position of the mask by `-'inf'`
    '''
    def __init__(self,config:MyBertConfig):
        super().__init__()
        self.config = config 
        self.layer = nn.ModuleList(
            [MyBertLayer(config) for _ in range(config.num_hidden_layers)])
        
    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.FloatTensor] = None,
                ) -> Tuple[torch.Tensor]:
        for i , bertlayer in enumerate(self.layer):
            layer_outputs = bertlayer(hidden_states,
                                      attention_mask)
            hidden_states = layer_outputs[0]

        return (hidden_states,)
    
class MyBertPooler(nn.Module):
    r'''Control cls token output prepared to other works

    Args:
        hidden_states (`torch.Tensor`):
            outputs after encoder layers
    '''
    def __init__(self,config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size,config.hidden_size)
        self.activation = nn.Tanh() 

    def forward(self,hidden_states:torch.Tensor) -> torch.Tensor:
        cls_outputs = hidden_states[:,0]
        pooled_output = self.dense(cls_outputs)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    
class MyBertPreTrainedModel(PreTrainedModel):
    config_class = MyBertConfig
    base_model_prefix = 'bert'
    supports_gradient_checkpointing = False
    _supports_sdpa = False 

    def _init_weights(self,module):
        if isinstance(module,nn.Linear):
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module,nn.Embedding):
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_() 
        elif isinstance(module,nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
class MyBertModel(MyBertPreTrainedModel):
    r'''Bert Model with embedding layer, encoder layers and pooler layers(if you want to add)

    If you do not want to add pooler layer, set `add_pooling_layer = False` 
    when instantiation

    Args:
        inputs_ids (`torch.Tensor`, *Optional*,defaults to None):
            Mapping of each tokens from dictionary
        attention_mask (`torch.Tensor`, *Optional*, defaults to None):
            record the position of mask by `-'inf'`
        token_type_ids (`torch.Tensor`, *Optional*, defaults to None):
            Mapping of sentences of each tokens
        position_ids (`torch.Tensor`, *Optional*, defaults to None):
            position of each tokens
        inputs_embeds (`torch.Tensor`, *Optional*, defaults to None):
            Vector sequence after word embedding
    '''
    def __init__(self,config:MyBertConfig,
                 add_pooling_layer=True):
        super().__init__(config)
        self.config = config 
        self.embeddings = MyBertEmbeddings(config)
        self.encoder = MyBertEncoder(config)
        self.pooler = MyBertPooler(config) if add_pooling_layer else None

        self.attn_implementation = config._attn_implementation
        self.position_embedding_type = config.position_embedding_type

        self.post_init() 

    def forward(self,
                inputs_ids:Optional[torch.Tensor] = None,
                attention_mask:Optional[torch.Tensor]= None,
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, None]]:
        
        if inputs_ids is not None and inputs_embeds is not None:
            raise ValueError('Can not pass inputs_ids and inputs_embeds in same call!')
        elif inputs_ids is not None:
            inputs_shape = inputs_ids.size() 
        elif inputs_embeds is not None:
            inputs_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError('Can not let inputs_ids and inputs_embeds None in same call!')
    
        batch_size , seq_len = inputs_shape
        device = inputs_ids.device if inputs_ids is not None else inputs_embeds.device

        if token_type_ids is None:
            if hasattr(self.embeddings, 'token_type_ids'):
                buffered_token_type_ids = self.embeddings.token_type_ids[:,:seq_len]
                buffered_token_type_ids = buffered_token_type_ids.expand(batch_size,seq_len)
                token_type_ids = buffered_token_type_ids
            else:
                token_type_ids = torch.zeros(inputs_shape,dtype=torch.long,device=device)


        embedding_output = self.embeddings(
            inputs_ids,
            token_type_ids,
            position_ids,
            inputs_embeds
        )

        if attention_mask is None:
            attention_mask = torch.ones((batch_size,seq_len),device=device)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask
        )

        sequence_output = encoder_outputs[0]

        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None 


        return (sequence_output,pooled_output)


class MyBertPredictionHeadTransform(nn.Module):
    r'''doing operations likes pooler layer

    Args:
        hidden_states (`torch.Tensor`):
            sequence output after BertModel
    '''
    def __init__(self,config:MyBertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act,str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = nn.ReLU()
        self.layerNorm = nn.LayerNorm(config.hidden_size,eps=config.layer_norm_eps)

    def forward(self,
                hidden_states: torch.Tensor)-> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layerNorm(hidden_states)

        return hidden_states

class MyBertLMPredictionHead(nn.Module):
    r'''Transform sequence outputs into dim = vocab_size tensor

    Args:
        hidden_states (`torch.Tensor`):
            sequence outputs after [`MyBertModel`]
    '''
    def __init__(self,config:MyBertConfig):
        super().__init__()
        self.transform = MyBertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size,config.vocab_size,bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        self.decoder.bias = self.bias 

    def forward(self,hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states
    
class MyBertForNSPredictionHead(nn.Module):
    r'''Transform pooled output into dim=num_label score
    '''
    def __init__(self,config:MyBertConfig):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size,config.num_labels)

    def forward(self,pooled_output: torch.Tensor)-> torch.Tensor:
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score

class MyBertPreTrainingHeads(nn.Module):
    r'''Calculate `prediction_scores` and `seq_relationship_score` by sequence_output from
    encoders in [`MyBertModel`] and pooled_output from pooler in [`MyBertModel`].

    it will return `prediction_scores`with shape `[batch_size,seq_len,vocab_size]` 
    and `seq_relationship_score` with shape `[batch_size,1,2]`

    Args:
        sequence_output (`torch.Tensor`):
            sequence_output from encoders
        pooled_output (`torch.Tensor`):
            pooled output from pooler
    '''
    def __init__(self,config:MyBertConfig):
        super().__init__()
        self.predictions = MyBertLMPredictionHead(config)
        self.seq_relationship = MyBertForNSPredictionHead(config)
    
    def forward(self,
                sequence_output: torch.Tensor,
                pooled_output: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return (prediction_scores,seq_relationship_score)
    

class MyBertForPreTraining(MyBertPreTrainedModel):
    r'''PreTraining head with MLM Prediction head and NSP Prediction head.

    it will return total loss including MLM loss and NSP loss.They are calculated 
    by [`CrossEntropyloss`]

    Args:
        inputs_ids (`torch.Tensor`, *Optional*,defaults to None):
            Mapping of each tokens from dictionary
        attention_mask (`torch.Tensor`, *Optional*, defaults to None):
            record the position of mask by `-'inf'`
        token_type_ids (`torch.Tensor`, *Optional*, defaults to None):
            Mapping of sentences of each tokens
        position_ids (`torch.Tensor`, *Optional*, defaults to None):
            position of each tokens
        inputs_embeds (`torch.Tensor`, *Optional*, defaults to None):
            Vector sequence after word embedding
        labels (`torch.Tensor`, *Optional*, defaults to None):
            Answer for MLM prediction
        next_sentence_label (`torch.Tensor`, *Optional*, defaults to None):
            Answer for  NS  prediction

    '''
    def __init__(self,config:MyBertConfig):
        super().__init__(config)

        self.bert = MyBertModel(config)
        self.cls = MyBertPreTrainingHeads(config)

        self.post_init()

    def forward(self,
                inputs_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                next_sentence_label: Optional[torch.Tensor] = None,
                ) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        outputs = self.bert(
            inputs_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            inputs_embeds
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_scores = self.cls(sequence_output,
                                                              pooled_output)
        
        total_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_func = CrossEntropyLoss() 
            mask_lm_loss = loss_func(prediction_scores.view(-1,self.config.vocab_size),
                                     labels.view(-1))
            next_sentence_loss = loss_func(seq_relationship_scores.view(-1,self.config.num_labels),
                                           next_sentence_label.view(-1))
            
            total_loss = mask_lm_loss + next_sentence_loss

        
        return (total_loss,prediction_scores,seq_relationship_scores)


