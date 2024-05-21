class DrugConfig():
    """
    This class maintains the Encoder configurations. 
    """ 
    def __init__(self,vocabulary_size=72,embed_dim=128,num_attention_heads=8,\
                 num_hidden_layers=8,pad_token_id=0,\
                 max_position_embedding=128, type_vocabulary_size=2,\
                 layer_norm_eps=1e-12,dropout_prob=0.1,intermediate_size=512,\
                 conv1_hidden=1,QKV_bias=True, **kwargs):
        self.vocabulary_size=vocabulary_size
        self.embed_dim=embed_dim
        self.pad_token_id=pad_token_id
        self.max_position_embedding=max_position_embedding
        self.type_vocabulary_size=type_vocabulary_size
        self.layer_norm_eps=layer_norm_eps
        self.dropout_prob=dropout_prob
        self.intermediate_size=intermediate_size
        self.num_attention_heads=num_attention_heads
        self.QKV_bias=QKV_bias
        self.num_hidden_layers=num_hidden_layers
        self.conv1_hidden=conv1_hidden
    
