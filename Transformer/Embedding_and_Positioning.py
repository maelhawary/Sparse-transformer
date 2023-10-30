
import torch
import torch.nn as nn

class Embedding_and_Postional_endocing():
    
    def __init__(self,vocab_size, d_model,seq_len) -> None:
        super().__init__
        self.vocab_size=vocab_size
        self.d_model=d_model
        self.seq_len=seq_len
    def embedding(self):
        embed_tbl=nn.Embedding(self.vocab_size, self.d_model)
        return embed_tbl
    def positioning (self):        
        pos_tbl=nn.Embedding(self.seq_len, self.d_model)
        return pos_tbl
