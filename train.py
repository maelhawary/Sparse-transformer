import dataset as dt
import torch
import torch.nn as nn
import config as config
from Transformer import Embedding_and_Positioning as encoding
from Transformer import Multi_attention_heads as Matt
from Transformer import Transformer_block as TRA
from Transformer import Transformer as Transformer
import os
from datasets import load_metric
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Apply the initialization to your model
def train(device,confi,dir):    #warnings.filterwarnings("ignore")
    #config = get_config()
   # train_model(config)    
    #introduce_device
    #wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    tokk=dt.Tokiniazation(chars)
    enc=tokk.tok_encode()
    # Train and test splits
    data = torch.tensor(enc(text), dtype=torch.long)
    n = int(confi['split']*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]
    # Create a custom dataset
    tr_dataset = dt.get_ds(train_data)
    val_dataset= dt.get_ds(val_data)
    # Create a DataLoader
    tr_dataloader = DataLoader(tr_dataset, batch_size=confi['batch_size'], shuffle=True)
    model=Transformer.DecoderBlock(device,vocab_size)
    model=model.to(device)
    # intializing the model parameters
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    optimizer = torch.optim.AdamW(model.parameters(), lr=confi['lr'])
    bleu_list=[]
    initial_epoch=0
    for epoch in range(initial_epoch, confi['num_epochs']):
        torch.cuda.empty_cache()
        model.train() 
        iter=0
            
        for batch in tr_dataloader:
            train_input=batch['input_seq'].to(device)
            train_tgt=batch['target_seq'].to(device)
            pred, loss = model(train_input, train_tgt)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()  
            # make a folder for saving the model 
            if not os.path.exists(dir):
                os.makedirs(dir) 
            #save the model each 1000 iterations     
            if iter % 1000 == 0:
                torch.save(model , dir+'model_iter_'+str(iter)+'.pth')
                torch.save(model.state_dict(), dir+'mode_state_iter_'+str(iter)+'.pt') 
                torch.save(optimizer.state_dict(), dir+'optimizer_state_dict'+str(iter)+'.pt')           
            # every once in a while evaluate the loss on train and val sets
                #loss_val=model(val_input, val_tgt)[1]
            #print(f"step {iter}: train loss {estimate_loss.loss(model,confi['eval_iter'],train_input, train_tgt):.4f},val loss {estimate_loss.loss(model,confi['eval_iter'],val_input, val_tgt):.4f}")
            print(f"epoch {epoch}: step {iter}: of_total_iter {len(tr_dataloader)}: train loss {loss:.4f}:", end="\r")
            ## we evaluate at each epoch for the half of the number of iteration in the trainging
