import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_transformer_copy import GraphTransformerCrossAttLayer
from common import AtomPoolingLayer
from common import MoleculePoolingLayer
from index2 import retrieve_center_index
import pdb
import time
class Net(nn.Module):
    def __init__(self, net_params, model_r, model_a, model_p):
        super().__init__()
        self_input_dim = net_params['self_input_dim']
        self_hidden_dim = net_params['self_hidden_dim']
        self_out_dim = net_params['self_out_dim']
        self_num_heads = net_params['self_num_heads']
        self_n_layers = net_params['self_n_layers']
        self_e_dim = net_params['edge_dim']

        cross_input_dim = net_params['cross_input_dim']
        cross_hidden_dim = net_params['cross_hidden_dim']
        cross_out_dim = net_params['cross_out_dim']
        cross_num_heads = net_params['cross_num_heads']    

        dropout = net_params['dropout']   

        molecule_pooling_method = net_params['molecule_pooling_method']
        
        self.readout = net_params['readout']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.device = net_params['device']

        #reactant transformer
        self.transformer_r = model_r
        #reagent transformer
        self.transformer_a = model_a
        #product transformer
        self.transformer_p = model_p
        
        self.atom_pooling_r = AtomPoolingLayer(self_out_dim, 1)
        self.atom_pooling_a = AtomPoolingLayer(self_out_dim, 1)
        self.atom_pooling_p = AtomPoolingLayer(self_out_dim, 1)

        self.mol_pooling = MoleculePoolingLayer(cross_out_dim, 1, molecule_pooling_method)

        self.cross_layer = GraphTransformerCrossAttLayer(cross_input_dim, cross_hidden_dim, cross_out_dim, cross_num_heads, self.layer_norm, self.batch_norm, self.residual)

        self.dim_align = nn.Linear(self_out_dim, cross_out_dim)

        predict_layer = [nn.Linear(cross_out_dim,256),nn.ReLU(),nn.Linear(256,1)]
        self.predict = nn.Sequential(*predict_layer)

        self.sigmoid = nn.Sigmoid()

    def forward(self, r_adj, r_node, r_dist, r_mask, r_center, r_index, a_adj, a_node, a_dist, a_mask, a_index, p_adj, p_node, p_dist, p_mask, p_index, training, device):
        #pdb.set_trace()
        r_node = r_node.to(device)
        r_mask = r_mask.to(device)
        r_adj = r_adj.to(device)
        r_dist = r_dist.to(device)
        #pdb.set_trace()
        r_new = self.transformer_r(r_node, r_mask, r_adj, r_dist, None)
        #print(r_new.shape)

        a_node = a_node.to(device)
        a_mask = a_mask.to(device)
        a_adj = a_adj.to(device)
        a_dist = a_dist.to(device)
        a_new = self.transformer_a(a_node, a_mask, a_adj, a_dist, None)
        #print(a_new.shape)

        p_node = p_node.to(device)
        p_mask = p_mask.to(device)
        p_adj = p_adj.to(device)
        p_dist = p_dist.to(device)
        p_new = self.transformer_p(p_node, p_mask, p_adj, p_dist, None)
        #print(p_new.shape)

        #reagents molecule representation
        #print('reagent atom pooling', a_new.shape)
        a_mol = self.atom_pooling_a(a_new, device, 'weighted_summation') #molecules
        #print('product atom pooling', p_new.shape)
        p_mol = self.atom_pooling_p(p_new, device, 'weighted_summation')

        #get reactants reaction center and create src and dst
        #pdb.set_trace()
        center_src, center_dst, center_index, r_node_new = retrieve_center_index(r_new, r_index, r_center, a_mol, a_index, device)
       
        #center cross attention with reagents
        r_center_hat = self.cross_layer(r_node_new, a_mol, center_src, center_dst, center_index, device)
        r_node_new[center_index] = r_center_hat
        r_update = r_node_new.reshape(r_new.shape)
        #reactants molecule representsation 
        #r_update = r_new
        #print('reactant atom pooling', r_update.shape)
        r_mol = self.atom_pooling_r(r_update, device, 'weighted_summation')
        
        #interact reactants/reagents/products
        reaction_h = self.mol_pooling(r_mol, r_index, a_mol, a_index, p_mol, p_index, device)
       
        #regression
        scores = self.predict(reaction_h)

        scores01 = self.sigmoid(scores)
        return scores01, reaction_h, r_mol, a_mol, p_mol

    def loss(self, scores, targets):
        #loss = nn.MSELoss()(scores,targets)
        loss = nn.L1Loss()(scores, targets)
        return loss
