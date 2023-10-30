import torch
import Dataset as dt
import torch
import torch.nn as nn
from config import get_config
from datasets import load_metric
from Transformer import Transformer as Transformer
from nltk.translate.bleu_score import sentence_bleu


state_dict_path='mode_state_iter_0.pt'
bleu_metric = load_metric("bleu")
confi=get_config()
#train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = dt.get_ds(config)

with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    # here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
tokk=dt.Tokiniazation(chars)
enc=tokk.tok_encode()
dec=tokk.tok_decode()
# Train and test splits
data = torch.tensor(enc(text), dtype=torch.long)
#train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)#
#val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model=Transformer.DecoderBlock(device,vocab_size)
m=model.to(device)
 
## load stat dic
state_dict = torch.load(state_dict_path,map_location=torch.device('cpu'))
m.load_state_dict(state_dict)
max_len=16

for m in range(5):   

        test_input, test_tgt =dt.get_batch_test(data,confi['seq_len'],1,device,max_len)#(B,T)
        print('shape',test_tgt.shape)
        print('test_input',test_input[0].shape)
        print('test_tgt',test_tgt[0].shape)
        print('test_input.size(1)',test_input.size(1))
        output=[]
        i=0
        #print('predicted============',predicted)
        #print('type============',type(predicted))
        while True:
                if i == max_len:
                    break
                
                # build mask for target

                # calculate output
                pred, loss = model(test_input)
               # print('expec============',pred.shape)
               # print('type============',type(pred))
                prob = pred.squeeze(0)
               # print('expec============',prob.shape)
               
                _, next_word = torch.max(prob[-1:], dim=1)  
        #print(next_word.item())
                #output.append[next_word]
                #output.append(next_word.item())
        #print(inp)
                #print('test_input',test_input)
               # print('next_word',next_word)
                test_input = torch.cat((test_input[0][1:].unsqueeze(0), next_word.unsqueeze(0)), dim=1)
                #print('test_input',test_input)
                i=i+1
                #print(i)
               #print('prob',prob)
               # print('worddd',pred_word_token)
               # print('GT',GT[0][:i])
               # loss = loss_fn(prob, GT[0][:i].view(-1))
               # print('loss',loss)
        # Load the BLEU metric
 
        #print('all_inp============',test_input[0][-max_len:])
        #print('all_inp_item============',test_input.shape)       
        #print('targt',test_tgt)
        can=test_input[0][-max_len:].tolist()
        ref=test_tgt.tolist()[0]

        reference_text = ' '.join(map(str, ref))
        candidate_text = ' '.join(map(str, can))

# Calculate BLEU score
        bleu_score = sentence_bleu([reference_text.split()], candidate_text.split())
        print(bleu_score)
        results = bleu_metric.compute(predictions=[candidate_text.split()], references=[[reference_text.split()]])
        print(results)

       # print('type============',type(expected))
        #print('predicted============',predicted)
        #print('type============',type(predicted))

