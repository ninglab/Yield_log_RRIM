import pdb
import math
import numpy as np
np.seterr(all="ignore")
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import sys
import socket
import time
import random
import glob
import argparse, json
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from common import load_weights
from load_data import Graph_DataLoader
from wholemodel_pretrain import Net
from index2 import get_reactant_batch, get_reagent_batch, get_product_batch
from transformer import make_model



use_gpu = True
if torch.cuda.is_available() and use_gpu:
    print('cuda available with GPU:',torch.cuda.get_device_name(0))
    device = torch.device("cuda")
else:
    print('cuda not available')
    device = torch.device("cpu")


    #load default config
with open('/fs/ess/PCON0041/xiaohu/MAT/Configs/w_best.json') as f:
    config = json.load(f)
# network parameters
net_params = config['net_params']
net_params['device'] = device
net_params['eb_n_layers'] = int(net_params['eb_n_layers'])
 
net_params['self_n_layers'] = int(net_params['self_n_layers'])
net_params['self_num_heads'] = int(net_params['self_num_heads'])
net_params['self_input_dim'] = int(net_params['self_input_dim'])
net_params['self_hidden_dim'] = int(net_params['self_hidden_dim'])
net_params['self_out_dim'] = int(net_params['self_out_dim'])  
net_params['edge_dim'] = int(net_params['edge_dim']) 
net_params['cross_num_heads'] = int(net_params['cross_num_heads'])
net_params['cross_input_dim'] = int(net_params['cross_input_dim'])
net_params['cross_hidden_dim'] = int(net_params['cross_hidden_dim'])
net_params['cross_out_dim'] = int(net_params['cross_out_dim'])  
net_params['dropout'] = float(net_params['dropout'])  
net_params['molecule_pooling_method'] = str(net_params['molecule_pooling_method'])
net_params['residual'] = True if net_params['residual']==True else False
net_params['readout'] = net_params['readout']
net_params['layer_norm'] = True if net_params['layer_norm']==True else False
net_params['batch_norm'] = True if net_params['batch_norm']==True else False
print(net_params)

params = config['params']
params['use_pretrain'] = bool(True) if params['use_pretrain']=='True' else bool(False)
print(params)


embedding_transformer_params = {
            'd_atom': 28, #28 fixed
            'd_model': 1024,  #1024 fixed
            'N': 8, #8 tunnale
            'h': 16, #16 fixed
            'N_dense': 1,
            'lambda_attention': 0.33, #0.33 fixed
            'lambda_distance': 0.33, #0.33 fixed
            'leaky_relu_slope': 0.1, #fixed
            'dense_output_nonlinearity': 'relu', 
            'distance_matrix_kernel': 'exp', 
            'dropout': 0,
            'aggregation_type': 'mean'
        }
eb_model = make_model(**embedding_transformer_params).to(device)

print('============Loading pretrained weights to generate initialization============')
load_weights(eb_model, 8)
for name, param_i in eb_model.named_parameters():
    param_i.requires_grad = False



print('============Creating new layers============')
transformer_params = {
            'd_atom': net_params['self_input_dim'], #1024 fixed
            'd_model': 256, #tunnable
            'N': net_params['self_n_layers'], #8 tunnale
            'h': 16, #16 fixed
            'N_dense': 1,
            'lambda_attention': 0.33, #0.33 fixed
            'lambda_distance': 0.33, #0.33 fixed
            'leaky_relu_slope': 0.1, #fixed
            'dense_output_nonlinearity': 'relu', 
            'distance_matrix_kernel': 'exp', 
            'dropout': 0, #0.0 
            'aggregation_type': 'mean'
        }
model_r = make_model(**transformer_params).to(device)
model_a = make_model(**transformer_params).to(device)
model_p = make_model(**transformer_params).to(device)

print('============Creating Model============')
model = Net(net_params, model_r, model_a, model_p)
model = model.to(device)

print('============Loading weights============')
best = torch.load('/fs/ess/PCON0041/xiaohu/MAT/results/temp/models/best.pkl', map_location=torch.device('cpu'))
model_state_dict = model.state_dict()
for name, param in best.items():
    print(name, param)
    model_state_dict[name] = param






#loading data
data_dir = '/fs/ess/PCON0041/xiaohu/MAT/Data/preprocessed_datasets/BH/BH_processed.csv'
data = pd.read_csv(data_dir) #reactions data
with open('/fs/ess/PCON0041/xiaohu/MAT/Data/preprocessed_datasets/BH/BH_reactant_feats.pkl', 'rb') as f:
    r_feats = pickle.load(f)
with open('/fs/ess/PCON0041/xiaohu/MAT/Data/preprocessed_datasets/BH/BH_reagents_feats.pkl', 'rb') as f:
    a_feats = pickle.load(f)
with open('/fs/ess/PCON0041/xiaohu/MAT/Data/preprocessed_datasets/BH/BH_products_feats.pkl', 'rb') as f:
    p_feats = pickle.load(f)

result_table = torch.zeros(data.shape[0],2).to(device)

split_set_num = 1
with open('/fs/ess/PCON0041/xiaohu/MAT/Data/preprocessed_datasets/BH/train_test_idxs.pickle', 'rb') as handle:
    idx_dict = pickle.load(handle)
            
train_index = idx_dict['train_idx'][split_set_num]
test_index = idx_dict['test_idx'][split_set_num]
valid_index = train_index[int(0.9*len(train_index)):]
train_index = train_index[0:int(0.9*len(train_index))]

training_set = Graph_DataLoader(data.loc[train_index].values, data_dir)
validation_set = Graph_DataLoader(data.loc[valid_index].values, data_dir) 
testing_set = Graph_DataLoader(data.loc[test_index].values, data_dir)

train_loader = DataLoader(training_set, batch_size=32, shuffle=True)
valid_loader = DataLoader(validation_set, batch_size=32, shuffle=False)
test_loader = DataLoader(testing_set, batch_size=32, shuffle=False)
print("Training Graphs Batches: ", len(train_loader))
print("Valid Graphs Batches: ", len(valid_loader))
print("Test Graphs Batches: ", len(test_loader))

result_table = evaluate_network_for_analysis(eb_model, model, device, train_loader, params, r_feats, a_feats, p_feats, train_index, result_table) 
print(result_table)
result_table = evaluate_network_for_analysis(eb_model, model, device, valid_loader, params, r_feats, a_feats, p_feats, valid_index, result_table) 
result_table = evaluate_network_for_analysis(eb_model, model, device, test_loader, params, r_feats, a_feats, p_feats, test_index, result_table) 