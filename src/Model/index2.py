import numpy as np
np.seterr(all="ignore")
import torch
from data_preprocessing import get_node_feat
from common import mol_collate_func
import pdb
def retrieve_center_index(r, r_index, r_center, a_mol, a_index, device):
    #pdb.set_trace()
    r = r.to(device)
    #r_center = torch.Tensor(r_center).to(device)
    r_center_list = [torch.Tensor(k).to(device) for k in r_center]
    r_node = r.reshape(-1, r.shape[-1]) # #nodes * features
    r_index = torch.Tensor(r_index).to(device)
    a_mol = a_mol.to(device)
    a_index = torch.Tensor(a_index).to(device)
    

    src = torch.tensor([],dtype=int).to(device)
    dst = torch.tensor([],dtype=int).to(device)
    center_index = torch.tensor([],dtype=int).to(device)
    num_nodes = 0
    for i in range(r.shape[0]): # reaction i
        ri =  r[i]# reaction i's reactants
        ri_center_index = r_center_list[i] + num_nodes
        len_center = ri_center_index.shape[0]

        ri_reagents_index = torch.where(a_index==r_index[i])[0] # reaction i's reagents index
        len_reagents = len(ri_reagents_index)
        
        ri_dst = torch.tensor([[i]*len_reagents for i in ri_center_index]).flatten().to(device)
        ri_src = ri_reagents_index.repeat(1,len_center).view(-1).to(device)

        dst = torch.cat((dst, ri_dst), dim=0)
        src = torch.cat((src, ri_src), dim=0)
        center_index = torch.cat((center_index, ri_center_index.to(device)), dim=0)

        num_nodes = num_nodes + ri.shape[0] 
    return src.long(), dst.long(), center_index.long(), r_node



"""
    Function to generate molecule batch for the model.
    Input: feats, reaction_index(train/valid/test reaction index), batch_index(index of each reaction in train/valid/test)
    Output: batch_adj(#molecules * max_atoms * #max_atoms), batch_dist(), batch_feats(#molecules * max_atoms * #atom_features), batch_mask: #atoms
"""
def get_reactant_batch(eb_model, r_feats, set_index, batch_index, params, device):
    #pdb.set_trace()
    batch_index_indeed = set_index[batch_index] #the indices of batch_reactions in original dataset(3955)
    temp = torch.Tensor([])
    for i in range(len(batch_index)):
        temp = torch.cat((temp, torch.where(torch.Tensor(r_feats['reaction_index']) == batch_index_indeed[i])[0]), dim=0) # index is reactant index
    index = [int(i.item()) for i in temp] #indices of molecules in batch_reactions

    r_batch_adjacency_matrix = np.array(r_feats['reactant_adj'])[index] 
    r_batch_node_features = np.array(r_feats['reactant_afm'])[index]
    r_batch_distance_matrix = np.array(r_feats['reactant_dist'])[index]
    r_batch_center_index = np.array(r_feats['center_index'])[index]
    r_batch_index = np.array(r_feats['reaction_index'])[index]

    r_batch_adjacency_matrix, r_batch_node_features, r_batch_distance_matrix = mol_collate_func(r_batch_adjacency_matrix, r_batch_node_features, r_batch_distance_matrix, device) #padded function
    r_batch_mask = torch.sum(torch.abs(r_batch_node_features), dim=-1) != 0    
    if params['use_pretrain']:
        eb_model.eval()
        r_batch_node_embedding = eb_model(r_batch_node_features, r_batch_mask, r_batch_adjacency_matrix, r_batch_distance_matrix, None)
    else:
        r_batch_node_embedding = r_batch_node_features 
    return r_batch_adjacency_matrix, r_batch_node_embedding, r_batch_distance_matrix, r_batch_mask, r_batch_center_index, r_batch_index


def get_reagent_batch(eb_model, a_feats, set_index, batch_index, params, device):
    batch_index_indeed = set_index[batch_index]
    temp = torch.Tensor([])
    for i in range(len(batch_index)):
        temp = torch.cat((temp, torch.where(torch.Tensor(a_feats['reaction_index']) == batch_index_indeed[i])[0]), dim=0) # index is reactant index
    index = [int(i.item()) for i in temp]
    
    a_batch_adjacency_matrix = np.array(a_feats['reagent_adj'])[index] 
    a_batch_node_features = np.array(a_feats['reagent_afm'])[index]
    a_batch_distance_matrix = np.array(a_feats['reagent_dist'])[index]
    a_batch_index = np.array(a_feats['reaction_index'])[index] 
    a_batch_adjacency_matrix, a_batch_node_features, a_batch_distance_matrix = mol_collate_func(a_batch_adjacency_matrix, a_batch_node_features, a_batch_distance_matrix, device) #padded function
    a_batch_mask = torch.sum(torch.abs(a_batch_node_features), dim=-1) != 0    
    if params['use_pretrain']:
        eb_model.eval()
        a_batch_node_embedding = eb_model(a_batch_node_features, a_batch_mask, a_batch_adjacency_matrix, a_batch_distance_matrix, None)
    else: 
        a_batch_node_embedding = a_batch_node_features
    return a_batch_adjacency_matrix, a_batch_node_embedding, a_batch_distance_matrix, a_batch_mask, a_batch_index


def get_product_batch(eb_model, p_feats, set_index, batch_index, params, device):
    batch_index_indeed = set_index[batch_index]
    temp = torch.Tensor([])
    for i in range(len(batch_index)):
        temp = torch.cat((temp, torch.where(torch.Tensor(p_feats['reaction_index']) == batch_index_indeed[i])[0]), dim=0) # index is reactant index
    index = [int(i.item()) for i in temp]
    
    p_batch_adjacency_matrix = np.array(p_feats['product_adj'])[index] 
    p_batch_node_features = np.array(p_feats['product_afm'])[index]
    p_batch_distance_matrix = np.array(p_feats['product_dist'])[index]
    p_batch_index = np.array(p_feats['reaction_index'])[index] 
    p_batch_adjacency_matrix, p_batch_node_features, p_batch_distance_matrix = mol_collate_func(p_batch_adjacency_matrix, p_batch_node_features, p_batch_distance_matrix, device) #padded function
    p_batch_mask = torch.sum(torch.abs(p_batch_node_features), dim=-1) != 0    
    if params['use_pretrain']:
        eb_model.eval()
        p_batch_node_embedding = eb_model(p_batch_node_features, p_batch_mask, p_batch_adjacency_matrix, p_batch_distance_matrix, None)
    else:
        p_batch_node_embedding = p_batch_node_features
    return p_batch_adjacency_matrix, p_batch_node_embedding, p_batch_distance_matrix, p_batch_mask, p_batch_index
  