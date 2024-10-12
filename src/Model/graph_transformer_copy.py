import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import radius_graph, knn_graph
from torch_scatter import scatter_softmax, scatter_sum

from common import  MLP
from index2 import retrieve_center_index
import pdb

class CrossAttLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, norm=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads

        # attention key func
        kv_input_dim = input_dim
        self.hk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm)

        # attention value func
        self.hv_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm)

        # attention query func
        self.hq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm)

    def forward(self, h, a_mol, src, dst, center_index, device):
        #pdb.set_trace()
        h = h.to(device)
        a_mol = a_mol.to(device)
        src = src.to(device)
        dst = dst.to(device)
        center_index = center_index.to(device)

        h_center = h[center_index]
        N = h_center.shape[0]
        hi = h[dst].to(device)
        hj = a_mol[src].to(device)

        dst_new = torch.zeros(dst.shape,dtype=dst.dtype).to(device)

        for i in range(N): #may take a lot of time, changing the index
            dst_new[torch.where(dst==center_index[i])[0]] = i

        # multi-head attention
        # decide inputs of k_func and v_func
        kv_input = hj
        # compute k
        k = self.hk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads)
        # compute v

        v = self.hv_func(kv_input)
        v = v.view(-1, self.n_heads, self.output_dim // self.n_heads)

        # compute q
        q = self.hq_func(hi).view(-1, self.n_heads, self.output_dim // self.n_heads) # hq_func(h) or 

        # compute attention weights
        alpha = scatter_softmax((q * k / np.sqrt(k.shape[-1])).sum(-1), dst_new, dim=0, dim_size=N)  # [num_edges, n_heads]

        # perform attention-weighted message-passing
        m = alpha.unsqueeze(-1) * v  # (E, heads, H_per_head)
        output = scatter_sum(m, dst_new, dim=0, dim_size=N)  # (N, heads, H_per_head)
        output = output.view(-1, self.output_dim)
        
        return output

class GraphTransformerCrossAttLayer(nn.Module):
    """
        Param: 
    """
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, layer_norm=False, batch_norm=True, residual=True):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.residual = residual
        self.layer_norm = layer_norm        
        self.batch_norm = batch_norm    

        self.attention = CrossAttLayer(input_dim, hidden_dim, output_dim, n_heads)
        
        self.O = nn.Linear(output_dim, output_dim)

        if self.layer_norm: # check if it can be used for batch
            self.layer_norm1 = nn.LayerNorm(output_dim)
            
        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(output_dim)
        
        # FFN
        self.FFN_layer1 = nn.Linear(output_dim, output_dim*2)
        self.FFN_layer2 = nn.Linear(output_dim*2, output_dim)

        if self.layer_norm:
            self.layer_norm2 = nn.LayerNorm(output_dim)
            
        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(output_dim)
        
    def forward(self, h, a_mol, center_src, center_dst, center_index, device):
        h_in1 = h[center_index] # for first residual connection
        
        # multi-head attention out
        h = self.attention(h, a_mol, center_src, center_dst, center_index, device)
        h = self.O(h)
        
        if self.residual:
            h = h_in1 + h # residual connection
        
        if self.layer_norm:
            h = self.layer_norm1(h)
            
        if self.batch_norm:
            h = self.batch_norm1(h)
        
        h_in2 = h # for second residual connection
        
        # FFN
        h = self.FFN_layer1(h)
        h = F.relu(h)
        #h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_layer2(h)

        if self.residual:
            h = h_in2 + h # residual connection
        
        if self.layer_norm:
            h = self.layer_norm2(h)
            
        if self.batch_norm:
            h = self.batch_norm2(h)       

        return h



class SelfAttLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, e_dim, norm=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.e_dim = e_dim
        #self.out_fc = out_fc

        # attention key func
        kv_input_dim = input_dim * 2 + e_dim
        self.hk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm)

        # attention value func
        self.hv_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm)

        # attention query func
        self.hq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm)

        #if self.out_fc:
        #    self.node_output = MLP(2 * hidden_dim, hidden_dim, hidden_dim, norm=norm)

    def forward(self, h, e, edge_index, device):
        #pdb.set_trace()
        h = h.to(torch.float32).to(device)
        e = e.to(torch.float32).to(device)
        N = h.size(0)
        
        src, dst = edge_index
        src = src.to(device)
        dst = dst.to(device)
        hi, hj = h[dst].to(device), h[src].to(device)

        # multi-head attention
        # decide inputs of k_func and v_func
        kv_input = torch.cat([e, hi, hj], -1).to(device)

        # compute k
        k = self.hk_func(kv_input)
        k = k.view(-1, self.n_heads, self.output_dim // self.n_heads)
        # compute v
        v = self.hv_func(kv_input)
        v = v.view(-1, self.n_heads, self.output_dim // self.n_heads)

        # compute q
        q = self.hq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads)
    
        # compute attention weights
        alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0,
                                dim_size=N)  # [num_edges, n_heads]

        # perform attention-weighted message-passing
        m = alpha.unsqueeze(-1) * v  # (E, heads, H_per_head)
        output = scatter_sum(m, dst, dim=0, dim_size=N)  # (N, heads, H_per_head)
        output = output.view(-1, self.output_dim)

        return output

class GraphTransformerSelfAttLayer(nn.Module):
    """
        Param: 
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, e_dim, dropout, layer_norm=False, batch_norm=True, residual=True):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_heads = num_heads
        self.e_dim = e_dim
        self.residual = residual
        self.layer_norm = layer_norm        
        self.batch_norm = batch_norm 
        self.dropout =  dropout
        self.dp = nn.Dropout(self.dropout)

        self.attention = SelfAttLayer(input_dim, hidden_dim, output_dim, num_heads, e_dim)
        
        self.O = nn.Linear(output_dim, output_dim)
        self.alien = nn.Linear(input_dim, output_dim)

        if self.layer_norm: # check if it can be used for batch
            self.layer_norm1 = nn.LayerNorm(output_dim)
            
        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(output_dim)
        
        # FFN
        self.FFN_layer1 = nn.Linear(output_dim, output_dim*2)
        self.FFN_layer2 = nn.Linear(output_dim*2, output_dim)

        if self.layer_norm:
            self.layer_norm2 = nn.LayerNorm(output_dim)
            
        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(output_dim)

        
        
    def forward(self, h, e, edge_index, training, device):
        h_in1 = h # for first residual connection
        h_in1 = h_in1.to(torch.float32) 
        # multi-head attention out
        h = self.attention(h, e, edge_index, device)
        h = self.O(h) 
        h = F.dropout(h, self.dropout, training=training)

        if self.residual:
            if h_in1.shape != h.shape:
                h_in1 = self.alien(h_in1)
            h = h_in1 + h # residual connection
        
        if self.layer_norm:
            h = self.layer_norm1(h)
        if self.batch_norm:
            h = self.batch_norm1(h)
        
        h_in2 = h # for second residual connection
        
        # FFN
        h = self.FFN_layer1(h)
        h = F.relu(h)
        h = self.FFN_layer2(h)
        h = F.dropout(h, self.dropout, training=self.training)

        if self.residual:
            h = h_in2 + h # residual connection

        if self.layer_norm:
            h = self.layer_norm2(h)
        if self.batch_norm:
            h = self.batch_norm2(h)       

        return h
