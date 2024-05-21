import torch.nn as nn
import torch
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,input_dim, embed_dim, num_heads,dropout_value=0,QKV_bias=False):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.K_embed = nn.Linear(input_dim, embed_dim,QKV_bias)
        self.Q_embed = nn.Linear(input_dim, embed_dim, QKV_bias)
        self.V_embed = nn.Linear(input_dim, embed_dim, QKV_bias)
        self.out_embed = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout=nn.Dropout(dropout_value)

    def forward(self,x):
        batch_size, max_length, given_input_dim = x.shape

        assert given_input_dim == self.input_dim
        assert self.embed_dim % self.num_heads == 0

        # compute K, Q, V
        K = self.K_embed(x) # (batch_size, max_length, embed_dim)
        Q = self.Q_embed(x)
        V = self.K_embed(x)

        indiv_dim = self.embed_dim // self.num_heads
        K = K.view(batch_size,-1,self.num_heads,indiv_dim)
        Q = Q.view(batch_size,-1,self.num_heads,indiv_dim)
        V = V.view(batch_size,-1,self.num_heads,indiv_dim)

        K = K.permute(0, 2, 1, 3) # (batch_size, num_heads, max_length, embed_dim / num_heads)
        Q = Q.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        K = K.reshape(batch_size * self.num_heads, max_length, indiv_dim)
        Q = Q.reshape(batch_size * self.num_heads, max_length, indiv_dim)
        V = V.reshape(batch_size * self.num_heads, max_length, indiv_dim)

        #transpose and batch matrix multiply
        K_T = K.permute(0, 2, 1) 
        QK = K@K_T

        #calculate weights by dividing everything by the square root of d (self.embed_dim)
        weights = QK/math.sqrt(self.embed_dim)

        weights = nn.functional.softmax(weights,dim=-1)
        weights=self.dropout(weights)

        w_V = weights@V

        # rejoin heads
        w_V = w_V.reshape(batch_size, self.num_heads, max_length, indiv_dim)
        w_V = w_V.permute(0, 2, 1, 3) # (batch_size, max_length, num_heads, embed_dim / num_heads)
        w_V = w_V.reshape(batch_size, max_length, self.embed_dim)

        out = self.out_embed(w_V)

        return out
    
class Embedding(nn.Module):
    def __init__(self,config,sequence=False):
        super().__init__()

        self.sequence=sequence

        self.token_embedding=nn.Embedding(config.vocabulary_size,config.embed_dim,padding_idx=config.pad_token_id)
        self.position_embedding=nn.Embedding(config.max_position_embedding,config.embed_dim)
        
        if sequence is True:
            self.segment_embedding=nn.Embedding(config.type_vocabulary_size,config.embed_dim)

        self.LayerNorm=nn.LayerNorm(config.embed_dim,eps=config.layer_norm_eps)
        self.dropout=nn.Dropout(config.dropout_prob)
        
        #I consider the Position_ids always that goes from 0 to max_position_embedding
        self.register_buffer("position_ids",torch.arange(config.max_position_embedding).expand((1,-1)),persistent=False)

    def forward(self, input_ids,token_type_ids=None) -> torch.Tensor:
        if self.sequence:
            assert token_type_ids is not None, ValueError("If sequence embedding is used, token_type_ids needs to be passed as a non-None value.") 

            embeddings= self.token_embedding(input_ids)+self.segment_embedding(token_type_ids)+self.position_embedding(self.position_ids)
        else:
            embeddings= self.token_embedding(input_ids)+self.position_embedding(self.position_ids)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    
class MaskedMultiHeadSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_attention_heads
        self.indiv_dim = self.embed_dim // self.num_heads

        self.Q_embed = nn.Linear(self.embed_dim, self.embed_dim, config.QKV_bias)
        self.K_embed = nn.Linear(self.embed_dim, self.embed_dim, config.QKV_bias)
        self.V_embed = nn.Linear(self.embed_dim, self.embed_dim, config.QKV_bias)
        self.out_embed = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.dropout=nn.Dropout(p=config.dropout_prob)
        self.dropout_2=nn.Dropout(p=config.dropout_prob)

    def forward(self,x,attention_mask=None):
        batch_size, max_length, given_input_dim = x.shape

        #print(input_shape)
        #print(batch_size,max_length,given_input_dim)
        assert given_input_dim == self.embed_dim
        assert self.embed_dim % self.num_heads == 0

        # compute K, Q, V
        K = self.K_embed(x) # (batch_size, max_length, embed_dim)
        Q = self.Q_embed(x)
        V = self.K_embed(x)

        #print(K.shape)

        K = K.view(batch_size,-1,self.num_heads,self.indiv_dim)
        Q = Q.view(batch_size,-1,self.num_heads,self.indiv_dim)
        V = V.view(batch_size,-1,self.num_heads,self.indiv_dim)

        #print(K.shape)
        K = K.permute(0, 2, 1, 3) # (batch_size, num_heads, max_length, embed_dim / num_heads)
        Q = Q.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        K = K.reshape(batch_size * self.num_heads, max_length, self.indiv_dim)
        Q = Q.reshape(batch_size * self.num_heads, max_length, self.indiv_dim)
        V = V.reshape(batch_size * self.num_heads, max_length, self.indiv_dim)

        #transpose and batch matrix multiply
        K_T = K.permute(0, 2, 1) 
        QK = K@K_T

        #calculate weights by dividing everything by the square root of d (self.embed_dim)
        weights = QK/math.sqrt(self.embed_dim)

        #Now, i have computed the weights BUT i need to consider that some elements ([PAD] tokens) of the input
        #should not be considered (i can do it by using the attention_mask)
        if attention_mask is not None:
            #print(weights.shape)
            #print(attention_mask.shape)
            extended_attention_mask=attention_mask.unsqueeze(1).unsqueeze(2)
            #print(extended_attention_mask.shape)
            extended_attention_mask=extended_attention_mask.expand(-1,self.num_heads,-1,-1)
            #print(extended_attention_mask.shape)
            extended_attention_mask=extended_attention_mask.reshape(batch_size*self.num_heads,max_length,1)
            #print(extended_attention_mask.shape)
            
            #NOTE: i used -inf before -1e9, but even though -inf is correct conceptually, by using it, it generates
            #nan as loss.
            weights=weights.masked_fill(extended_attention_mask==0,float('-1e9'))

        weights = nn.functional.softmax(weights,dim=-1)
        weights=self.dropout(weights)

        w_V = weights@V

        # rejoin heads
        w_V = w_V.reshape(batch_size, self.num_heads, max_length, self.indiv_dim)
        w_V = w_V.permute(0, 2, 1, 3) # (batch_size, max_length, num_heads, embed_dim / num_heads)
        w_V = w_V.reshape(batch_size, max_length, self.embed_dim)

        out = self.out_embed(w_V)
        out = self.dropout_2(out)

        return out
    
class AttentionAddAndNorm(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.LayerNorm=nn.LayerNorm(config.embed_dim,eps=config.layer_norm_eps)

    def forward(self,emb_multi_head,input):
        #Here i neeed to perform 2 things:
        #1) Residual connection
        #2) Layer norm

        out=emb_multi_head+input
        out=self.LayerNorm(out)

        return out
    
class AttentionBlock(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.MultiHeadSelfAttention=MaskedMultiHeadSelfAttention(config)
        self.add_and_norm=AttentionAddAndNorm(config)

    def forward(self,input,attention_mask):
        embedding=self.MultiHeadSelfAttention(input,attention_mask)
        out=self.add_and_norm(embedding,input)

        return out
    
class PositionWiseFFN(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config

        self.conv1=nn.Sequential(
            nn.Conv1d(config.embed_dim,config.intermediate_size,kernel_size=3,stride=1,padding=1),
            nn.Dropout(p=config.dropout_prob),
            nn.ReLU()
        )
        
        #The number of conv1 hidden is defined in the config class
        self.conv_hidden=self._create_conv1d_layers(config.conv1_hidden)
        
        self.conv2=nn.Sequential(
            nn.Conv1d(config.intermediate_size,config.embed_dim,kernel_size=3,stride=1,padding=1),
            nn.Dropout(p=config.dropout_prob),
            nn.ReLU()
        )

    def _create_conv1d_layers(self,num_hidden):
        layers=list()

        if num_hidden <=1:
            return None
        
        #I perform it for num_hidden_layers - 1 because 1 hidden layer will always be created.
        for _ in range(num_hidden-1):
            layers.append(nn.Conv1d(self.config.intermediate_size,self.config.intermediate_size,kernel_size=3,stride=1,padding=1))
            layers.append(nn.Dropout(p=self.config.dropout_prob))
            layers.append(nn.ReLU())

        return nn.Sequential(*layers)
    
    def forward(self,output_attention):
        #print(output_attention.shape)
        output_attention=output_attention.permute(0,2,1)
        out=self.conv1(output_attention)
        #print(out.shape)

        #If it is None it means that the number of hidden conv1 has been set to <=1
        if self.conv_hidden is not None:
            out=self.conv_hidden(out)
        #print(out.shape)
        
        out=self.conv2(out)
        #print(out.shape)
        
        out=out.permute(0,2,1)

        return out
    
class PositionWiseAddAndNorm(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.LayerNorm=nn.LayerNorm(config.embed_dim,eps=config.layer_norm_eps)

    def forward(self,position_wise_output,attention_output):
        #Here i need to perform the same concept as performed in the AttentionAddAndNorm
        out=position_wise_output+attention_output
        out=self.LayerNorm(out)

        return out
    
class EncoderBlock(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.attention_block=AttentionBlock(config)
        self.position_wise_ffn=PositionWiseFFN(config)
        self.position_wise_add_and_norm=PositionWiseAddAndNorm(config)

    def forward(self,embedding,attention_mask):
        attention_output=self.attention_block(embedding,attention_mask)
        out=self.position_wise_ffn(attention_output)
        out=self.position_wise_add_and_norm(out,attention_output)
        
        return out
    
class Encoder(nn.Module):
    def __init__(self,config):
        super().__init__()
        #I concatenate multiple Encoder blocks one after the other. The number of the 
        #EncoderBlock is an hyperparameter.
        self.encoder=nn.ModuleList([EncoderBlock(config) for foo in range(config.num_hidden_layers)])

    def forward(self,embedding,attention_mask):
        for encoder_block in self.encoder:
            embedding=encoder_block(embedding,attention_mask)
        return embedding
    
class Pooler(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.linear=nn.Linear(config.embed_dim,config.embed_dim)
        self.tanh=nn.Tanh()

    def forward(self,enc_out):
        cls_token_vector=enc_out[:,0]
        out=self.linear(cls_token_vector)
        out=self.tanh(out)

        return out

