"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import sys
import time
import torch.nn as nn
import math
import scipy
from metrics import MAE, RMSE
from index2 import get_reactant_batch, get_reagent_batch, get_product_batch
from sklearn.metrics import r2_score
from common import pad_array, mol_collate_func

import pdb

def train_epoch(eb_model, model, optimizer, device, data_loader, params, r_feats, a_feats, p_feats, train_index):
    #pdb.set_trace()
    model.train()
    epoch_loss = 0
    epoch_train_mae = 0
    epoch_train_rmse = 0
    epoch_train_r2 = 0
    training = bool(True) #for dropout
    
    for iter, (batch_index, batch_targets) in enumerate(data_loader):
        #pdb.set_trace()
        batch_targets = batch_targets.float().to(device)
        index_indeed = torch.Tensor(train_index[batch_index])
        batch_targets_sorted = batch_targets[torch.sort(index_indeed)[1]]
        #get the molecule index from reaction index, and get the padded molecule batch for model.
        r_batch_adjacency_matrix, r_batch_node_features, r_batch_distance_matrix, r_batch_mask, r_batch_center_index, r_batch_index  = get_reactant_batch(eb_model, r_feats, train_index, batch_index, params, device)
        a_batch_adjacency_matrix, a_batch_node_features, a_batch_distance_matrix, a_batch_mask, a_batch_index = get_reagent_batch(eb_model, a_feats, train_index, batch_index, params, device)
        p_batch_adjacency_matrix, p_batch_node_features, p_batch_distance_matrix, p_batch_mask, p_batch_index = get_product_batch(eb_model, p_feats, train_index, batch_index, params, device)
        optimizer.zero_grad()
        #torch.autograd.set_detect_anomaly(True)
       
        batch_scores, batch_representation, r_h, a_h, p_h = model.forward(r_batch_adjacency_matrix, r_batch_node_features, r_batch_distance_matrix, r_batch_mask, r_batch_center_index, r_batch_index, 
        a_batch_adjacency_matrix, a_batch_node_features, a_batch_distance_matrix, a_batch_mask, a_batch_index, 
        p_batch_adjacency_matrix, p_batch_node_features, p_batch_distance_matrix, p_batch_mask, p_batch_index, training, device)

        batch_scores = batch_scores.flatten().float()
        loss = model.loss(batch_scores, batch_targets_sorted).float()
        loss.backward()

        #if(iter%1000==0):
        #    print('===================Batch %d==================='%iter)
        #    print('batch_scores', batch_scores)
        #    print('batch_targets', batch_targets_sorted)
        #    print('loss', loss)
        #    print('reactant representations', r_h)
        #    print('reagent representations', a_h)
        #    print('product representations', p_h)
        #    print('reaction representations', batch_representation)

        #    print('Check Params Grad')
            #for name, param_i in model.named_parameters():
            #    if param_i.grad is not None:
            #        print(name, param_i.requires_grad, param_i.grad.norm()) 

            #sys.stdout.flush()

        if params['gnorm']!=1000:
            nn.utils.clip_grad_norm_(model.parameters(), params['gnorm'])

        optimizer.step()
        epoch_loss += loss.item()
        epoch_train_mae += MAE(batch_scores, batch_targets_sorted)
        epoch_train_rmse += RMSE(batch_scores, batch_targets_sorted) 
        r2_value = r2_score(batch_targets_sorted.cpu().detach(), batch_scores.cpu().detach()) 
        #r2_value = 0
        epoch_train_r2 += r2_value

    epoch_loss /= (iter + 1)
    epoch_train_mae /= (iter + 1)
    epoch_train_rmse /= (iter + 1)
    epoch_train_r2 /= (iter + 1)
    return epoch_loss, epoch_train_mae, epoch_train_rmse, epoch_train_r2, optimizer



def evaluate_network(eb_model, model, device, data_loader, params, r_feats, a_feats, p_feats, eval_index):
    model.eval()
    epoch_test_loss = 0
    epoch_test_mae = 0
    epoch_test_rmse = 0
    epoch_test_r2 = 0
    training = bool(False)
    with torch.no_grad():
        for iter, (batch_index, batch_targets) in enumerate(data_loader):
            batch_targets = batch_targets.float().to(device)
            index_indeed = torch.Tensor(eval_index[batch_index])
            batch_targets_sorted = batch_targets[torch.sort(index_indeed)[1]].float().to(device)
            
            r_batch_adjacency_matrix, r_batch_node_features, r_batch_distance_matrix, r_batch_mask, r_batch_center_index, r_batch_index  = get_reactant_batch(eb_model, r_feats, eval_index, batch_index, params, device)
            a_batch_adjacency_matrix, a_batch_node_features, a_batch_distance_matrix, a_batch_mask, a_batch_index = get_reagent_batch(eb_model, a_feats, eval_index, batch_index, params, device)
            p_batch_adjacency_matrix, p_batch_node_features, p_batch_distance_matrix, p_batch_mask, p_batch_index = get_product_batch(eb_model, p_feats, eval_index, batch_index, params, device)

            batch_scores, batch_representation, r_h, a_h, p_h = model.forward(r_batch_adjacency_matrix, r_batch_node_features, r_batch_distance_matrix, r_batch_mask, r_batch_center_index, r_batch_index, 
            a_batch_adjacency_matrix, a_batch_node_features, a_batch_distance_matrix, a_batch_mask, a_batch_index, 
            p_batch_adjacency_matrix, p_batch_node_features, p_batch_distance_matrix, p_batch_mask, p_batch_index, training, device) 

            batch_scores = batch_scores.flatten().float()
            #print(batch_scores, batch_targets_sorted)
            loss = model.loss(batch_scores, batch_targets_sorted).float()

            epoch_test_loss += loss.detach().item()
            epoch_test_mae += MAE(batch_scores, batch_targets_sorted)
            epoch_test_rmse += RMSE(batch_scores, batch_targets_sorted)
            r2_value = r2_score(batch_targets_sorted.cpu().detach(), batch_scores.cpu().detach())
            epoch_test_r2 += r2_value

        epoch_test_loss /= (iter + 1)
        epoch_test_mae /= (iter + 1)
        epoch_test_rmse /= (iter + 1)
        epoch_test_r2 /= (iter + 1)
    return epoch_test_loss, epoch_test_mae, epoch_test_rmse, epoch_test_r2


def evaluate_network_for_analysis(eb_model, model, device, data_loader, params, r_feats, a_feats, p_feats, eval_index, result_table):
    eb_model.eval()
    model.eval()
    training = bool(False)
    with torch.no_grad():
        for iter, (batch_index, batch_targets) in enumerate(data_loader):
            batch_targets = batch_targets.float().to(device)
            index_indeed = torch.Tensor(eval_index[batch_index])
            batch_targets_sorted = batch_targets[torch.sort(index_indeed)[1]].float().to(device)
            
            r_batch_adjacency_matrix, r_batch_node_features, r_batch_distance_matrix, r_batch_mask, r_batch_center_index, r_batch_index  = get_reactant_batch(eb_model, r_feats, eval_index, batch_index, params, device)
            #print(r_batch_node_features.shape)
            a_batch_adjacency_matrix, a_batch_node_features, a_batch_distance_matrix, a_batch_mask, a_batch_index = get_reagent_batch(eb_model, a_feats, eval_index, batch_index, params, device)
            p_batch_adjacency_matrix, p_batch_node_features, p_batch_distance_matrix, p_batch_mask, p_batch_index = get_product_batch(eb_model, p_feats, eval_index, batch_index, params, device)

            batch_scores, batch_representation, r_h, a_h, p_h = model.forward(r_batch_adjacency_matrix, r_batch_node_features, r_batch_distance_matrix, r_batch_mask, r_batch_center_index, r_batch_index, 
            a_batch_adjacency_matrix, a_batch_node_features, a_batch_distance_matrix, a_batch_mask, a_batch_index, 
            p_batch_adjacency_matrix, p_batch_node_features, p_batch_distance_matrix, p_batch_mask, p_batch_index, training, device) 

            batch_scores = batch_scores.float()
            #print(torch.sort(index_indeed)[0])
            
            result_table[torch.sort(index_indeed)[0].long(),0] = batch_scores.flatten()
            result_table[torch.sort(index_indeed)[0].long(),1] = batch_targets_sorted.flatten()

    return result_table


