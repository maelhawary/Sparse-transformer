import torch
import torch.nn as nn
from Transformer import Position_wise_FFN as FFN
from Transformer import Local_Multi_attention_heads as Matt


class TransformerBlock(nn.Module):

    def __init__(self,number_of_heads,d_model,d_att_weigh,seq_len,d_FFN,dropout,window_size) -> None:
        super().__init__()

        self.att= Matt.Multiheads( number_of_heads, d_model, d_att_weigh, seq_len, dropout,window_size)
        self.LN_1=nn.LayerNorm(d_model)
        self.LN_2=nn.LayerNorm(d_att_weigh)
        self.FFN=FFN.PositionWiseFFN(d_att_weigh, d_FFN,dropout)

    def forward(self,idx):
        #print('shapp',idx.shape)
        out=idx+self.att(self.LN_1(idx))
        out=out+self.FFN(self.LN_2(out))# (B,T,d_FFN)
        return out