import torch
import torch.nn as nn
from Transformer import Embedding_and_Positioning as encoding
from Transformer import Transformer_block as Trans
import config as config



class DecoderBlock(nn.Module):
    def __init__(self, device, vocab_size) -> None:
        super().__init__()
        self.confi=config.get_config()
        self.window_size=self.confi['window_size']
        self.d_model=self.confi['d_model']
        self.seq_len= self.confi['seq_len']
        self.number_of_heads=self.confi['number_of_heads']
        self.number_of_layers=self.confi['number_of_layers']
        self.d_att_weigh=self.confi['d_att_weigh']
        self.d_FFN=self.confi['d_FFN']
        self.dropout=self.confi['dropout']
        self.encod=encoding.Embedding_and_Postional_endocing(vocab_size, self.d_model,self.seq_len)
        self.embd=self.encod.embedding()
        self.pos=self.encod.positioning()
        self.blocks= nn.Sequential(*[Trans.TransformerBlock(self.number_of_heads,self.d_model,self.d_att_weigh,
                                                            self.seq_len,self.d_FFN,self.dropout,self.window_size) for i in range(self.number_of_layers)])
        self.LN=nn.LayerNorm(self.d_model)
        self.output = nn.Linear(self.d_model, vocab_size)
        self.device=device

    def forward(self,idx,ground_truth=None):
        B,T=idx.shape
        idx=idx.to(self.device)
        x=self.embd(idx)+self.pos(torch.arange(T, device=self.device))
        x=self.blocks(x)
        x=self.LN(x)
        pred=self.output(x)

        if ground_truth is None:
            loss = None
        else:
            B, T, C = pred.shape
            #print('pred_shape',pred.shape)
            pred = pred.view(B*T, C)
            #print('pred_shape_2',pred.shape)
            ground_truth = ground_truth.view(B*T)
            #print('ground_truth_shape',ground_truth.shape)
            loss = nn.functional.cross_entropy(pred, ground_truth)

        return pred, loss



