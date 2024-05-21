import torch.nn as nn
from model.utils.config import *
from model.utils.attention import *

class TransformerDDI_FCL(nn.Module):
    """
        This is the implementation of the Transformer-DDI that exploits Fully-connected layers
        as feature fusion and classification.
    """
    def __init__(self,left_config,right_config,num_classes=86,dropout_linear=0.1,num_hidden_linear_layers=2,embedding_dim=1024):
        super().__init__()
        assert num_hidden_linear_layers >= 1, ValueError("The number of hidden layers needs to be at least 1.") 

        self.num_hidden_layers=num_hidden_linear_layers

        self.left_embedding=Embedding(left_config)
        self.left_encoder=Encoder(left_config)
        self.left_pooler=Pooler(left_config)

        self.right_embedding=Embedding(right_config)
        self.right_encoder=Encoder(right_config)
        self.right_pooler=Pooler(right_config)

        self.feature_vector_size=left_config.embed_dim+right_config.embed_dim

        self.linear_1=nn.Sequential(
            nn.Linear(in_features=self.feature_vector_size,out_features=embedding_dim,bias=True),
            nn.ReLU(),
            nn.Dropout(p=dropout_linear)
        )

        self.hidden_layers=self._create_hidden_layers(num_hidden_linear_layers,dropout_linear,embedding_dim)
        #print(self.hidden_layers)

        self.classificator=nn.Linear(in_features=embedding_dim,out_features=num_classes,bias=True)

    def _create_hidden_layers(self,num_hidden_layers,dropout,embedding_dim):
        layers=list()
        #I perform it for num_hidden_layers - 1 because 1 hidden layer will always be created.
        for _ in range(num_hidden_layers-1):
            layers.append(nn.Linear(embedding_dim,embedding_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))

        return nn.Sequential(*layers)

    def forward(self,left_input_smiles,left_attention_mask,right_input_smiles,right_attention_mask):
        left_embedding=self.left_embedding(left_input_smiles)
        left_out=self.left_encoder(left_embedding,left_attention_mask)
        left_out=self.left_pooler(left_out)
        
        right_embedding=self.right_embedding(right_input_smiles)
        right_out=self.right_encoder(right_embedding,right_attention_mask)
        right_out=self.right_pooler(right_out)
        
        #print(out1.shape)
        #print(out2.shape)
        out=torch.cat([left_out,right_out],dim=1)
        out=self.linear_1(out)
        if self.num_hidden_layers > 1:
            out=self.hidden_layers(out)
        #print(out.shape)
        out=self.classificator(out)
        return out
    

class TransformerDDI_MHA(nn.Module):
    """
        This is the implementation of the Transformer-DDI that exploits Multi-head self-attention
        as feature fusion.
    """
    def __init__(self,left_config,right_config,num_classes=86,num_multi_head=4,multi_head_dropout=0.1):
        super().__init__()
        self.left_embedding=Embedding(left_config)
        self.left_encoder=Encoder(left_config)
        self.left_pooler=Pooler(left_config)

        self.right_embedding=Embedding(right_config)
        self.right_encoder=Encoder(right_config)
        self.right_pooler=Pooler(right_config)

        self.feature_vector_size=left_config.embed_dim+right_config.embed_dim
        self.multihead=MultiHeadSelfAttention(self.feature_vector_size,self.feature_vector_size\
                                              ,num_multi_head,dropout_value=multi_head_dropout)
        
        self.classificator=nn.Linear(in_features=self.feature_vector_size,out_features=num_classes,bias=True)

    def forward(self,left_input_smiles,left_attention_mask,right_input_smiles,right_attention_mask):
        left_embedding=self.left_embedding(left_input_smiles)
        left_out=self.left_encoder(left_embedding,left_attention_mask)
        left_out=self.left_pooler(left_out)
        
        right_embedding=self.right_embedding(right_input_smiles)
        right_out=self.right_encoder(right_embedding,right_attention_mask)
        right_out=self.right_pooler(right_out)
        
        #print(out1.shape)
        #print(out2.shape)
        out=torch.cat([left_out,right_out],dim=1)
        #print(out.shape)
        out=out.unsqueeze(1)
        #print(out.shape)
        out=self.multihead(out)
        #print(out.shape)
        out=torch.squeeze(out,1)
        out=self.classificator(out)
        return out

class TransformerDDI_Sequence(nn.Module):
    """
        This is the implementation of the Transformer-DDI that accepts multiple SMILES strings as input.
    """
    def __init__(self,config,num_classes=86):
        super().__init__()

        self.embedding=Embedding(config,sequence=True)
        self.encoder=Encoder(config)
        self.pooler=Pooler(config)

        self.classificator=nn.Linear(in_features=config.embed_dim,out_features=num_classes,bias=True)

    def forward(self,input_smiles,attention_mask,segment_mask):
        embedding=self.embedding(input_smiles,segment_mask)
        out=self.encoder(embedding,attention_mask)
        out=self.pooler(out)
        
        out=self.classificator(out)
        return out
