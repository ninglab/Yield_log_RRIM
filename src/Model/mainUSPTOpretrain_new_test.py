import dgl
import math
import numpy as np
import pdb
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

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit

#from tensorboardX import SummaryWriter
from tqdm import tqdm
from common import load_weights
from load_data import Graph_DataLoader
from wholemodel_pretrain_test import Net
from train_pretrain2_test import train_epoch, evaluate_network, evaluate_network_for_analysis
from transformer import make_model
import pdb

"""
    GPU Setup
"""
def gpu_setup(use_gpu):
    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device


def get_scheduler(params, optimizer):
    if params['lrtype'] == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=params['factor'],
            threshold=0.01,
            patience=params['patience'],
            min_lr=params['min_lr'],
            verbose=True
        )
        
def seed_worker(worker_id):                                                          
    worker_seed = np.random.get_state()[1][0]
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)        
    torch.manual_seed(params['seed'])
    os.environ['PYTHONHASHSEED'] = str(params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def load_seed(params, device):
    # setting seeds
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    os.environ['PYTHONHASHSEED'] = str(params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)



"""
    TRAINING CODE
"""
def train_val_pipeline(DATASET_NAME, MODEL_NAME, testset, params, net_params, dirs, r_test_feats, a_test_feats, p_test_feats, test_index, result_table):
    #t0 = time.time()
    #per_epoch_time = []
    #root_log_dir = dirs
    device = net_params['device']

    ckpt_dir = dirs + '_testmodels' #used for save models
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    #result_file_name = dirs + 'results.txt' #used to write the parameters and final performace
    
    print("==========Training Start==========")
    print("Test Graphs: ", len(testset))
    embedding_transformer_params = {
            'd_atom': 28, #28 fixed
            'd_model': 1024,  #1024 fixed
            'N': net_params['eb_n_layers'], #8 tunnale
            'h': 16, #16 fixed
            'N_dense': 1,
            'lambda_attention': 0.33, #0.33 fixed
            'lambda_distance': 0.33, #0.33 fixed
            'leaky_relu_slope': 0.1, #fixed
            'dense_output_nonlinearity': 'relu', 
            'distance_matrix_kernel': 'exp', 
            'dropout': net_params['dropout'], #0.0 
            'aggregation_type': 'mean'
        }
    eb_model = make_model(**embedding_transformer_params).to(device)
    if params['use_pretrain']==True:
        print('============Loading pretrained weights to generate initialization============')
        load_weights(eb_model, net_params['eb_n_layers'])
        for name, param_i in eb_model.named_parameters():
            param_i.requires_grad = False
    else:
        print('============Not pretrained weights used============') 
        for name, param_i in eb_model.named_parameters():
            param_i.requires_grad = False


    print('============Creating new layers============')
    if params['use_pretrain']==True:
        net_params['self_input_dim'] = 1024
    else:
        net_params['self_input_dim'] = 29
    transformer_params = {
            'd_atom': net_params['self_input_dim'], #1024 fixed
            'd_model': net_params['self_hidden_dim'], #tunnable
            'N': net_params['self_n_layers'], #8 tunnale
            'h': net_params['self_num_heads'], #16 fixed
            'N_dense': 1,
            'lambda_attention': 0.33, #0.33 fixed
            'lambda_distance': 0.33, #0.33 fixed
            'leaky_relu_slope': 0.1, #fixed
            'dense_output_nonlinearity': 'relu', 
            'distance_matrix_kernel': 'exp', 
            'dropout': net_params['dropout'], #0.0 
            'aggregation_type': 'mean'
        }
    model_r = make_model(**transformer_params).to(device)
    model_a = make_model(**transformer_params).to(device)
    model_p = make_model(**transformer_params).to(device)

    print('============Creating Model============')
    model = Net(net_params, model_r, model_a, model_p)
    #model.load_state_dict(torch.load(checkpoint_file))
    checkpoint = torch.load("/fs/ess/PCON0041/xiaohu/MAT/results/final/uspto_new_wsigmoid/womodels/best_model.pkl", map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    for name, param_i in model.named_parameters():
        #print(name, param_i)
        param_i.requires_grad = False

    best_model = model
    test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False)
    print("Test Graphs Batches: ", len(test_loader))
    sys.stdout.flush() 

    print("==========Test============")
    #torch.save(best_model.state_dict(), '{}.pkl'.format(ckpt_dir + "/best_model"))
    epoch_test_loss, epoch_test_mae, epoch_test_rmse, epoch_test_r2 = evaluate_network(eb_model, best_model, device, test_loader, params, r_test_feats, a_test_feats, p_test_feats, test_index)   
    result_table = evaluate_network_for_analysis(eb_model, best_model, device, test_loader, params, r_test_feats, a_test_feats, p_test_feats, test_index, result_table)
    torch.save(result_table, '{}.pt'.format(ckpt_dir + "/test_result"))
    return epoch_test_loss, epoch_test_mae, epoch_test_rmse, epoch_test_r2, result_table


def r_select_reactions(test_index, feats):
    test_feats = {'reactants':[], 'reactant_afm':[], 'reactant_adj':[], 'reactant_dist':[], 'reaction_index':[], 'molecule_index':[], 'center_index':[], 'yield':[]}
    for idx in test_index:
        occurrence = [i for i, val in enumerate(feats['reaction_index']) if val == idx]
        # Add corresponding elements to the selected list
        for occur in occurrence:
            test_feats['reactants'].append(feats['reactants'][occur])
            test_feats['reactant_afm'].append(feats['reactant_afm'][occur])
            test_feats['reactant_adj'].append(feats['reactant_adj'][occur])
            test_feats['reactant_dist'].append(feats['reactant_dist'][occur])
            test_feats['reaction_index'].append(feats['reaction_index'][occur])
            test_feats['molecule_index'].append(feats['molecule_index'][occur])
            test_feats['center_index'].append(feats['center_index'][occur])
            test_feats['yield'].append(feats['yield'][occur])
    return test_feats

def a_select_reactions(test_index, feats):
    test_feats = {'reagents':[], 'reagent_afm':[], 'reagent_adj':[], 'reagent_dist':[], 'reaction_index':[], 'molecule_index':[], 'yield':[]}
    for idx in test_index:
        occurrence = [i for i, val in enumerate(feats['reaction_index']) if val == idx]
        # Add corresponding elements to the selected list
        for occur in occurrence:
            test_feats['reagents'].append(feats['reagents'][occur])
            test_feats['reagent_afm'].append(feats['reagent_afm'][occur])
            test_feats['reagent_adj'].append(feats['reagent_adj'][occur])
            test_feats['reagent_dist'].append(feats['reagent_dist'][occur])
            test_feats['reaction_index'].append(feats['reaction_index'][occur])
            test_feats['molecule_index'].append(feats['molecule_index'][occur])
            test_feats['yield'].append(feats['yield'][occur])
    return test_feats

def p_select_reactions(test_index, feats):
    test_feats = {'products':[], 'product_afm':[], 'product_adj':[], 'product_dist':[], 'reaction_index':[], 'molecule_index':[], 'yield':[]}
    for idx in test_index:
        occurrence = [i for i, val in enumerate(feats['reaction_index']) if val == idx]
        # Add corresponding elements to the selected list
        for occur in occurrence:
            test_feats['products'].append(feats['products'][occur])
            test_feats['product_afm'].append(feats['product_afm'][occur])
            test_feats['product_adj'].append(feats['product_adj'][occur])
            test_feats['product_dist'].append(feats['product_dist'][occur])
            test_feats['reaction_index'].append(feats['reaction_index'][occur])
            test_feats['molecule_index'].append(feats['molecule_index'][occur])
            test_feats['yield'].append(feats['yield'][occur])
    return test_feats

def main():    
    sys.stdout.flush()
    """
        USER CONTROLS
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/fs/ess/PCON0041/xiaohu/MAT/Configs/test.json', help="config.json file with training/model/data/param details")
    parser.add_argument('--gpu_id', default=1, help="gpu id")
    parser.add_argument('--model', default='GraphTransformer', type=str, help="model to learn node representation")
    parser.add_argument('--dataset', default='USPTO', type=str, help="dataset name")
    parser.add_argument('--out_dir', default='/fs/ess/PCON0041/xiaohu/MAT/script/single/', help="out_dir")
    parser.add_argument('--seed', default=0, type=int, help="random seed")
    parser.add_argument('--epochs', default=5, type=int, help="number of epochs")
    parser.add_argument('--batch_size', default=32, type=int, help="batch_size")
    parser.add_argument('--test_size', default=0.3, help='test_size')
    parser.add_argument('--valid_size', default=0.9, type=float, help='valid_size')
    parser.add_argument('--split', default=1, type=int, help='n_splits')
    parser.add_argument('--gnorm', default=1000, type=float, help='using gradient norm clipping') 
    parser.add_argument('--weight_decay', default=0, type=float, help='vweight')

    parser.add_argument('--lrtype', default='plateau', type=str, help='lr_decay_type') 
    parser.add_argument('--min_lr', default=1e-6, type=float, help='min_lr')
    parser.add_argument('--init_lr', default=3e-4, type=float, help="init_lr")
    parser.add_argument('--patience', default=5, type=float, help="")
    parser.add_argument('--factor', default=0.9, type=float, help="")

    parser.add_argument('--dropout', default=0.0, type=float, help="")
    parser.add_argument('--molecule_pooling_method', default='con', type=str, help="")

    parser.add_argument('--print_epoch_interval', default=20, type=int, help="print_epoch_interval")    
    
    parser.add_argument('--eb_n_layers', default=8, type=int, help="number of graph transformer layers")
    parser.add_argument('--self_n_layers', default=5, type=int, help="number of graph transformer layers")
    parser.add_argument('--self_input_dim', default=29, type=int, help="node embedding dimension")
    parser.add_argument('--self_hidden_dim', default=256, type=int, help="feature dimension in QKV in transformer")
    parser.add_argument('--self_out_dim', default=256, type=int, help="node features dimension transformer output")
    parser.add_argument('--self_num_heads', default=16, type=int, help="number of multiheads")
    parser.add_argument('--edge_dim', default=7, type=int, help="edge features dimension")

    parser.add_argument('--cross_input_dim', default=256, type=int, help="node features dimension")
    parser.add_argument('--cross_hidden_dim', default=256, type=int, help="feature dimension in QKV in transformer")
    parser.add_argument('--cross_out_dim', default=256, type=int, help="node features dimension transformer output")
    parser.add_argument('--cross_num_heads', default=16, type=int, help="number of multiheads")

    parser.add_argument('--residual', default=True, help="whether residual after attention")
    parser.add_argument('--readout', default=False, help="whether include readout out function")
    parser.add_argument('--layer_norm', default=True, help="layer_norm")
    parser.add_argument('--batch_norm', default=False, help="batch_norm")

    parser.add_argument('--use_pretrain', default=0, type=int, help="0 for False, 1 for True")
    args = parser.parse_args()

    #load default config
    with open(args.config) as f:
        config = json.load(f)
    # device
    if args.gpu_id is not None:
        device = gpu_setup(True)
    # model, dataset, out_dir
    if args.model is not None:
        MODEL_NAME = args.model
    if args.dataset is not None:
        DATASET_NAME = args.dataset
        args.data_dir = '/fs/ess/PCON0041/xiaohu/MAT/Data/preprocessed_datasets/' + DATASET_NAME 
    if args.out_dir is not None:
        out_dir = args.out_dir
        

    # parameters
    params = config['params']
    if args.seed is not None:
        params['seed'] = int(args.seed)
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)
    if args.batch_size is not None:
        params['batch_size'] = int(args.batch_size)
    if args.test_size is not None:
        params['test_size'] = float(args.test_size)
    if args.valid_size is not None:
        params['valid_size'] = float(args.valid_size)
    if args.split is not None:
        params['split'] = int(args.split)
    if args.weight_decay is not None:
        params['weight_decay'] = float(args.weight_decay)

    if args.lrtype is not None:
        params['lrtype'] = str(args.lrtype)
    if args.min_lr is not None:
        params['min_lr'] = float(args.min_lr)
    if args.factor is not None:
        params['factor'] = float(args.factor)
    if args.patience is not None:
        params['patience'] = float(args.patience)
    if args.init_lr is not None:
        params['init_lr'] = float(args.init_lr)
    if args.gnorm is not None:
        params['gnorm'] = float(args.gnorm)
    if args.print_epoch_interval is not None:
        params['print_epoch_interval'] = int(args.print_epoch_interval)
    if args.use_pretrain==1:
        params['use_pretrain'] = bool(True) 
    else:
        params['use_pretrain'] = bool(False)
 

    
    print("==========Load Seed==========")
    load_seed(params, device)

    # network parameters
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']

    if args.eb_n_layers is not None:
        net_params['eb_n_layers'] = int(args.eb_n_layers)
    if args.self_n_layers is not None:
        net_params['self_n_layers'] = int(args.self_n_layers)
    if args.self_num_heads is not None:
        net_params['self_num_heads'] = int(args.self_num_heads)
    if args.self_input_dim is not None:
        net_params['self_input_dim'] = int(args.self_input_dim)
    if args.self_hidden_dim is not None:
        net_params['self_hidden_dim'] = int(args.self_hidden_dim)
    if args.self_out_dim is not None:
        net_params['self_out_dim'] = int(args.self_hidden_dim)  
    if args.edge_dim is not None:
        net_params['edge_dim'] = int(args.edge_dim) 
    if args.cross_num_heads is not None:
        net_params['cross_num_heads'] = int(args.cross_num_heads)
    if args.cross_input_dim is not None:
        net_params['cross_input_dim'] = int(args.self_hidden_dim)
    if args.cross_hidden_dim is not None:
        net_params['cross_hidden_dim'] = int(args.self_hidden_dim)
    if args.cross_out_dim is not None:
        net_params['cross_out_dim'] = int(args.self_hidden_dim)  
    if args.dropout is not None:
        net_params['dropout'] = float(args.dropout)  
    if args.molecule_pooling_method is not None:
        net_params['molecule_pooling_method'] = str(args.molecule_pooling_method)
    if args.residual is not None:
        net_params['residual'] = True if args.residual==True else False
    if args.readout is not None:
        net_params['readout'] = args.readout
    if args.layer_norm is not None:
        net_params['layer_norm'] = True if args.layer_norm==True else False
    if args.batch_norm is not None:
        net_params['batch_norm'] = True if args.batch_norm==True else False



    # setting seeds
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device.type == 'cuda':
        print('set_random_seed')
        torch.cuda.manual_seed(params['seed'])
    print(np.random.get_state()[1][0])

    #setting dirs
    dirs = out_dir 
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print("==========Load Data==========")
    #test_data_dir = '/fs/ess/PCON0041/xiaohu/MAT/Data/preprocessed_datasets/USPTO/USPTO500MT_test_processed_100.csv'
    #test_data = pd.read_csv(test_data_dir)
    #test_index = torch.Tensor(np.arange(test_data.shape[0]+1))
    #result_table = torch.zeros(test_data.shape[0],2).to(device)


########## Testing on BH #######################
#    bh_data_dir = '/fs/ess/PCON0041/xiaohu/MAT/Data/preprocessed_datasets/BH/BH_processed.csv'
#    bh_data = pd.read_csv(bh_data_dir) #reactions data
#    with open('/fs/ess/PCON0041/xiaohu/MAT/Data/preprocessed_datasets/BH/BH_reactant_feats.pkl', 'rb') as f:
#        r_feats = pickle.load(f)
#    with open('/fs/ess/PCON0041/xiaohu/MAT/Data/preprocessed_datasets/BH/BH_reagents_feats.pkl', 'rb') as f:
#        a_feats = pickle.load(f)
#    with open('/fs/ess/PCON0041/xiaohu/MAT/Data/preprocessed_datasets/BH/BH_products_feats.pkl', 'rb') as f:
#        p_feats = pickle.load(f)
#    #result_table = torch.zeros(bh_data.shape[0],2).to(device)
#    split_set_num = 2
#    with open('/fs/ess/PCON0041/xiaohu/MAT/Data/preprocessed_datasets/BH/train_test_idxs.pickle', 'rb') as handle:
#        idx_dict = pickle.load(handle)
#    #test_index = np.array([i for i in range(3955)])
#    test_index = idx_dict['test_idx'][split_set_num]
#    result_table = torch.zeros(len(test_index),2).to(device)
#    testing_set = Graph_DataLoader(bh_data.loc[test_index].values, bh_data_dir)
#    r_test_feats = r_select_reactions(test_index, r_feats)
#    a_test_feats = a_select_reactions(test_index, a_feats)
#    p_test_feats = p_select_reactions(test_index, p_feats)

####### Testing on Sampled reactions ###################
    sample_data_dir = '/fs/ess/PCON0041/xiaohu/MAT/Data/sample/sample_test_CJ_processed_final_5w_wo0.csv'
    sample_data = pd.read_csv(sample_data_dir) #reactions data
    test_index = np.array([i for i in range(sample_data.shape[0])])
    with open('/fs/ess/PCON0041/xiaohu/MAT/Data/sample/sample_test_CJ_reactant_feats_final_5w_wo0.pkl', 'rb') as f:
        r_test_feats = pickle.load(f)
    with open('/fs/ess/PCON0041/xiaohu/MAT/Data/sample/sample_test_CJ_reagents_feats_final_5w_wo0.pkl', 'rb') as f:
        a_test_feats = pickle.load(f)
    with open('/fs/ess/PCON0041/xiaohu/MAT/Data/sample/sample_test_CJ_products_feats_final_5w_wo0.pkl', 'rb') as f:
        p_test_feats = pickle.load(f)
    result_table = torch.zeros(sample_data.shape[0],2).to(device)
    testing_set = Graph_DataLoader(sample_data.values, sample_data_dir)


########## Testing on USPTO #######################

#    valid_data_dir = '/fs/ess/PCON0041/xiaohu/MAT/Data/preprocessed_datasets/USPTO/USPTO500MT_valid_processed_100.csv'
#    valid_data = pd.read_csv(valid_data_dir)
#    valid_index = torch.Tensor(np.arange(valid_data.shape[0]+1))

#    train_data_dir = '/fs/ess/PCON0041/xiaohu/MAT/Data/preprocessed_datasets/USPTO/USPTO500MT_train_processed_100.csv'
#    train_data = pd.read_csv(train_data_dir)
#    train_index = torch.Tensor(np.arange(train_data.shape[0]+1))

#    print("==========Load Traininig Featuers==========")
#    with open('/fs/ess/PCON0041/xiaohu/MAT/Data/preprocessed_datasets/USPTO/USPTO500MT_train_processed_100_reactant_feats.pkl', 'rb') as f1:
#        r_train_feats = pickle.load(f1)
#    with open('/fs/ess/PCON0041/xiaohu/MAT/Data/preprocessed_datasets/USPTO/USPTO500MT_train_processed_100_reagents_feats.pkl', 'rb') as f2:
#        a_train_feats = pickle.load(f2)
#    with open('/fs/ess/PCON0041/xiaohu/MAT/Data/preprocessed_datasets/USPTO/USPTO500MT_train_processed_100_products_feats.pkl', 'rb') as f3:
#        p_train_feats = pickle.load(f3)
    

#    print("==========Load Validation Featuers==========")
#    with open('/fs/ess/PCON0041/xiaohu/MAT/Data/preprocessed_datasets/USPTO/USPTO500MT_valid_processed_100_reactant_feats.pkl', 'rb') as f1:
#        r_valid_feats = pickle.load(f1)
#    with open('/fs/ess/PCON0041/xiaohu/MAT/Data/preprocessed_datasets/USPTO/USPTO500MT_valid_processed_100_reagents_feats.pkl', 'rb') as f2:
#        a_valid_feats = pickle.load(f2)
#    with open('/fs/ess/PCON0041/xiaohu/MAT/Data/preprocessed_datasets/USPTO/USPTO500MT_valid_processed_100_products_feats.pkl', 'rb') as f3:
#        p_valid_feats = pickle.load(f3)

#    print("==========Load Testing Featuers==========")
#    with open('/fs/ess/PCON0041/xiaohu/MAT/Data/preprocessed_datasets/USPTO/USPTO500MT_test_processed_100_reactant_feats.pkl', 'rb') as f1:
#        r_test_feats = pickle.load(f1)
#    with open('/fs/ess/PCON0041/xiaohu/MAT/Data/preprocessed_datasets/USPTO/USPTO500MT_test_processed_100_reagents_feats.pkl', 'rb') as f2:
#        a_test_feats = pickle.load(f2)
#    with open('/fs/ess/PCON0041/xiaohu/MAT/Data/preprocessed_datasets/USPTO/USPTO500MT_test_processed_100_products_feats.pkl', 'rb') as f3:
#        p_test_feats = pickle.load(f3)
###

#    sys.stdout.flush()

#    training_set = Graph_DataLoader(train_data.values, train_data_dir)
#    validation_set = Graph_DataLoader(valid_data.values, valid_data_dir)
    #testing_set = Graph_DataLoader(test_data.values, test_data_dir)

    test_mse, test_mae, test_rmse, test_r2, result_table = train_val_pipeline(DATASET_NAME, MODEL_NAME, testing_set, params, net_params, dirs, r_test_feats, a_test_feats, p_test_feats, test_index, result_table)
    print("=======================================")
    print('mse:', test_mse)
    print('rmse:', test_rmse)
    print('mae:', test_mae)
    print('r2:', test_r2)
main()    