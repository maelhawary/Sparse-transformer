import torch
import train as tr
from config import get_config
import warnings
import torch
import time
import pandas as pd


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    dir='save_models'+'/'+'new_tokenize_'+'/'
    config=get_config()
    start_time = time.time()
    tr.train(device,config,dir)
    total_time = time.time() - start_time 
    print('total_time=',total_time)
    time_list=[total_time]
