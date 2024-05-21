import random
from collections import OrderedDict

class SmilesTokenizer():
    def __init__(self, vocab_file,max_length):
        self.max_length=max_length
        self.vocab=self.load_vocabulary(vocab_file)

    def load_vocabulary(self,vocab_file):
        vocabulary=OrderedDict()
        with open(vocab_file,"r") as vf:
            token_vocab=vf.readlines()

        for i,token in enumerate(token_vocab):
            token=token.split('\n')[0]
            vocabulary[token]=i

        return vocabulary
    
    def convert_tokens_to_ids(self,tokens):
        ris=[self.vocab[token] for token in tokens]
        return ris

    def _truncate_seq_pair(self,tokens, max_num_tokens, rng):
        while True:
            total_length = len(tokens)
            if total_length <= max_num_tokens:
                trunc_tokens=tokens
                break
            
            trunc_tokens = tokens
            assert len(trunc_tokens) >= 1
    
            #We want to sometimes truncate from the front and sometimes from the
            #back to add more randomness and avoid biases.
            if rng.random() < 0.5:  
                del trunc_tokens[0]
            else:
                trunc_tokens.pop()
        
        return trunc_tokens

    def get_attention_mask(self,tokens):
        #Since it masks tokens, they should have been processed 
        #by the .tokenize method, which means that their length should be 100
        assert len(tokens) == self.max_length

        attention_mask=[]
        for i in range(self.max_length):
            if tokens[i] != 0:
                attention_mask.append(1)
            else:
                attention_mask.append(0)
        
        return attention_mask
    
class SmilesSingleTokenizer(SmilesTokenizer):
    """
        SmilesTokenizer employed for the TransformerDDI_FCL and TransformerDDI_MHA
        It performs the tokenization process for one drug once at a time.
    """
    def tokenize(self, text):
        tokens = []
        tokens.append("[CLS]")

        #[BEGIN] and [END] tokens are inserted.
        if len(text) < self.max_length-2:
            tokens.append("[BEGIN]")
        for c in text:
            tokens.append(c)
        if len(text) < self.max_length-2:
            tokens.append("[END]")
        
        if len(text) >= self.max_length:
            tokens[1:self.max_length]=self._truncate_seq_pair(tokens[1:len(tokens)],self.max_length-1,random.Random())
            for i in range(len(tokens)):
                if len(tokens) > self.max_length:
                    tokens.pop()
                else:
                    break

        #Here i'm padding other elements so that
        #it remains consistent with the original pre-trained one
        for i in range(self.max_length):
            if len(tokens) < self.max_length:
                tokens.append("[PAD]")
            else:
                break

        #print(tokens)
        return tokens


class SmilesSequenceTokenizer(SmilesTokenizer):
    """
        SmilesTokenizer employed for the TransformerDDI_Sequence.
        It performs the tokenization process for multiple SMILES strings separated by |.
    """
    def tokenize(self, text):
        tokens = []

        #[BEGIN] and [END] tokens are inserted.
        if len(text) < (int(self.max_length/2))-4:
            tokens.append("[BEGIN]")
        for c in text:
            tokens.append(c)
        if len(text) < (int(self.max_length/2))-4:
            tokens.append("[END]")
        
        if len(text) >= (int(self.max_length/2))-1:
            tokens[0:(int(self.max_length/2))-3]=self._truncate_seq_pair(tokens[0:len(tokens)],(int(self.max_length/2))-2,random.Random())
            for i in range(len(tokens)):
                if len(tokens) > (int(self.max_length/2))-2:
                    tokens.pop()
                else:
                    break

        #print(tokens)
        return tokens

    def tokenization(self,text):
        tokens = []
        tokens.append("[CLS]")

        tmp=[]
        for i in text:
            if i == '|':
                tokens.extend(self.tokenize(tmp))
                tokens.append('[SEP]')
                tmp=[]
            else:
                tmp.append(i)

        if len(tmp) != 0:
            tokens.extend(self.tokenize(tmp))
        #Here i'm padding other elements so that
        #it remains consistent with the original pre-trained one
        for i in range(self.max_length):
            if len(tokens) < self.max_length:
                tokens.append("[PAD]")
            else:
                break
        
        return tokens
    
    def get_sentence_mask(self,tokens):
        assert len(tokens) == self.max_length
        j=0
        sentence_mask=[]
        for i in range(self.max_length):
            sentence_mask.append(j)
            if tokens[i] == 4:
                j+=1
        return sentence_mask