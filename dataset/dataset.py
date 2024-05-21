import torch
from torch.utils.data import Dataset
import pandas as pd
from tokenization.tokenizer import *

class DDIdataset(Dataset):
    def __init__(self,csv_file,vocab,config) -> None:
        self.dataset=pd.read_csv(csv_file,sep=";")
        self.tokenizer=SmilesSingleTokenizer(vocab,max_length=config.max_position_embedding)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,idx):
        """
            This function is used for giving as a result the sample with the name
            name_idx and its label.
        """
        assert isinstance(idx,int)

        label=self.dataset.iloc[idx]['Y']
        
        drug1_ID=self.dataset.iloc[idx]['Drug1_ID']
        drug1=self.dataset.iloc[idx]['Drug1']
        
        drug2_ID=self.dataset.iloc[idx]['Drug2_ID']
        drug2=self.dataset.iloc[idx]['Drug2']

        drug1_token_id=self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(drug1))
        drug2_token_id=self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(drug2))

        drug1_masked=self.tokenizer.get_attention_mask(drug1_token_id)
        drug2_masked=self.tokenizer.get_attention_mask(drug2_token_id)
        
        return drug1_ID,torch.tensor(drug1_token_id),torch.tensor(drug1_masked),drug2_ID,torch.tensor(drug2_token_id),torch.tensor(drug2_masked),torch.tensor(label)
    
class DDIdatasetSequence(Dataset):
    def __init__(self,csv_file,vocab,config) -> None:
        self.dataset=pd.read_csv(csv_file,sep=";")
        self.tokenizer=SmilesSequenceTokenizer(vocab,max_length=config.max_position_embedding)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,idx):
        """
            This function is used for giving as a result the sample with the name
            name_idx and its label.
        """
        assert isinstance(idx,int)

        label=self.dataset.iloc[idx]['Y']
        
        drug1_ID=self.dataset.iloc[idx]['Drug1_ID']
        drug1=self.dataset.iloc[idx]['Drug1']
        
        drug2_ID=self.dataset.iloc[idx]['Drug2_ID']
        drug2=self.dataset.iloc[idx]['Drug2']

        drug_sentence=drug1+'|'+drug2
        drug_token_id=self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenization(drug_sentence))
        attention_mask=self.tokenizer.get_attention_mask(drug_token_id)
        sentence_mask=self.tokenizer.get_sentence_mask(drug_token_id)

        
        return drug1_ID,drug2_ID,torch.tensor(drug_token_id),torch.tensor(attention_mask),torch.tensor(sentence_mask),torch.tensor(label)