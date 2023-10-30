import torch
from torch.utils.data import Dataset
import config as config

class get_ds(Dataset):
    def __init__(self, dt):
        self.confi=config.get_config()
        self.ds = dt
        self.sequence_length = self.confi['seq_len']

    def __len__(self):
        return len(self.ds)- self.sequence_length 

    def __getitem__(self, index):
        input_seq = self.ds[index:index + self.sequence_length]
        target_seq = self.ds[index+1:index +1+ self.sequence_length]
        return {
            "input_seq": input_seq,  
            "target_seq": target_seq,  
        }
class Tokiniazation():

    def __init__(self, chars: int) -> None:
        super().__init__
# create a mapping from characters to integers (i.e. Tokanization)
        self.chars=chars
        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.itos = { i:ch for i,ch in enumerate(self.chars) }      
    def tok_encode(self):
        encode_text = lambda s: [self.stoi[c] for c in s] # encoder: take a string, output a list of integers
        return encode_text
    def tok_decode(self):
        decode_tok = lambda l: ''.join([self.itos[i] for i in l]) # decoder: take a list of integers, output a string
        return decode_tok
    
def get_batch_test(dt,T_len,B_len,device,max_len):
    # generate a small batch of data of inputs x and targets y
    ix = torch.randint(len(dt) - T_len, (B_len,))
    x = torch.stack([dt[i:i+T_len] for i in ix])
    y = torch.stack([dt[i+T_len:i+T_len+max_len] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y    