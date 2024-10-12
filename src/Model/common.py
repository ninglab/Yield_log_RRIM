import numpy as np
np.seterr(all="ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

def load_weights(model_i, num_layers_to_keep):
    #pdb.set_trace()
    pretrained_name = '/fs/ess/PCON0041/xiaohu/MAT/pretrained_weights.pt'  # This file should be downloaded first (See README.md).
    pretrained_state_dict = torch.load(pretrained_name)

    # Define the layers to keep
    layers_to_keep = [
        "encoder.norm.a_2",
        "encoder.norm.b_2",
        "src_embed.lut.weight",  # these are additional the layers to keep
        "src_embed.lut.bias",
        "generator.proj.weight",
        "generator.proj.bias",
        *[f"encoder.layers.{i}." for i in range(num_layers_to_keep)]  # load the first 6 self-attention layers
    ]
    model_state_dict = model_i.state_dict()

    # Filter out the parameters corresponding to the specified layers
    filtered_state_dict = {k: v for k, v in pretrained_state_dict.items() if any(layer in k for layer in layers_to_keep)}
    for name, param in filtered_state_dict.items():
        if 'generator' in name:
            continue
        if isinstance(param, torch.nn.Parameter):
            param = param.data
        model_state_dict[name].copy_(param)



def pad_array(array, shape, dtype=np.float32):
    """Pad a 2-dimensional array with zeros.

    Args:
        array (ndarray): A 2-dimensional array to be padded.
        shape (tuple[int]): The desired shape of the padded array.
        dtype (data-type): The desired data-type for the array.

    Returns:
        A 2-dimensional array of the given shape padded with zeros.
    """
    padded_array = np.zeros(shape, dtype=dtype)
    padded_array[:array.shape[0], :array.shape[1]] = array
    return padded_array


def mol_collate_func(batch_adjacency_matrix, batch_node_matrix, batch_distance_matrix, device):

    """Create a padded batch of molecule features.

    Args:
        batch (list[Molecule]): A batch of raw molecules.

    Returns:
        A list of FloatTensors with padded molecule features:
        adjacency matrices, node features, distance matrices, and labels.
    """
    use_cuda = torch.cuda.is_available()
    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
    IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
    DoubleTensor = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor

    adjacency_list, distance_list, features_list = [], [], []

    max_size = 0
    for i in range(batch_adjacency_matrix.shape[0]):
        molecule_adjacency_matrix = batch_adjacency_matrix[i]
        molecule_node_matrix = batch_node_matrix[i]
        molecule_distance_matrix = batch_distance_matrix[i]
        
        if molecule_adjacency_matrix.shape[0] > max_size:
            max_size = molecule_adjacency_matrix.shape[0]
            
    #print(max_size)
    for i in range(batch_adjacency_matrix.shape[0]):
        adjacency_list.append(pad_array(batch_adjacency_matrix[i], (max_size, max_size)))
        distance_list.append(pad_array(batch_distance_matrix[i], (max_size, max_size)))
        features_list.append(pad_array(batch_node_matrix[i], (max_size, batch_node_matrix[i].shape[1])))

    return [FloatTensor(features).to(device) for features in (adjacency_list, features_list, distance_list)]



class MLP(nn.Module):
    """MLP with the same hidden dim across all layers."""

    def __init__(self, in_dim, out_dim, hidden_dim, num_layer=2, norm=False):
        super().__init__()
        layers = []
        for layer_idx in range(num_layer):
            if layer_idx == 0:
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ReLU())
            elif layer_idx == num_layer - 1:
                layers.append(nn.Linear(hidden_dim, out_dim))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
            if layer_idx < num_layer - 1:
                if norm:
                    layers.append(nn.LayerNorm(hidden_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class AtomPoolingLayer(nn.Module):
    """input: #nodes x f, output: #molecules x f."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        layers = [nn.Linear(in_dim, int(in_dim/4))]
        layers.append(nn.ReLU())
        layers.append(nn.Linear(int(in_dim/4), out_dim))
        layers.append(nn.Sigmoid()) 
        self.pooling = nn.Sequential(*layers)
    def forward(self, h, device, method):
        h = h.to(device)
        #print(h.shape)
        total_mol_h = torch.tensor([]).to(device)
        if(method == 'mean'):
            total_mol_h = torch.mean(h, dim=0)
        if(method == 'weighted_summation'):
            #print('atomic weight')
            w_h = self.pooling(h) # #molecules * #nodes * 1
            for i in range(h.shape[0]): # molecule i 
                mol_h = h[i]
                mol_w_h = w_h[i]
                #print(i,w_h[i]) 
                total_mol_h = torch.cat((total_mol_h,torch.mm(mol_w_h.T, mol_h)), dim=0)
        return total_mol_h


class MoleculePoolingLayer(nn.Module):
    """input: #nodes x f, output: #molecules x f."""

    def __init__(self, in_dim, out_dim, method):
        super().__init__()   
        layers_r = [nn.Linear(in_dim, int(in_dim/4))]
        layers_r.append(nn.ReLU())
        layers_r.append(nn.Linear(int(in_dim/4), out_dim))
        layers_r.append(nn.Sigmoid())
        self.net_r = nn.Sequential(*layers_r) 

        layers_a = [nn.Linear(in_dim, int(in_dim/4))]
        layers_a.append(nn.ReLU())
        layers_a.append(nn.Linear(int(in_dim/4), out_dim))
        layers_a.append(nn.Sigmoid())
        self.net_a = nn.Sequential(*layers_a) 

        layers_p = [nn.Linear(in_dim, int(in_dim/4))]
        layers_p.append(nn.ReLU())
        layers_p.append(nn.Linear(int(in_dim/4), out_dim))
        layers_p.append(nn.Sigmoid())
        self.net_p = nn.Sequential(*layers_p) 

        #for reactant/reagent/product ws
        #layers_t = [nn.Linear(in_dim, out_dim)]
        #layers_t.append(nn.Sigmoid())
        #self.net_t = nn.Sequential(*layers_t) 

        #for concatenate
        self.dim_fit = nn.Linear(3*in_dim, in_dim)

        self.method = method

    def forward(self, r_mol, r_molecule_index, a_mol, a_molecule_index, p_mol, p_molecule_index, device):
        #pdb.set_trace()
        a_molecule_index = torch.Tensor(a_molecule_index).to(device)
        r_molecule_index = torch.Tensor(r_molecule_index).to(device)
        p_molecule_index = torch.Tensor(p_molecule_index).to(device)
        reaction_index = torch.unique(r_molecule_index)

        total_reaction_h = torch.tensor([]).to(device)
        w_r = self.net_r(r_mol)
        w_a = self.net_a(a_mol)
        w_p = self.net_p(p_mol)
        for i in reaction_index:
            index_r = torch.where(r_molecule_index==i)[0]
            index_a = torch.where(a_molecule_index==i)[0]
            index_p = torch.where(p_molecule_index==i)[0]

            r_hi = torch.matmul(w_r[index_r].T, r_mol[index_r]) 
            a_hi = torch.matmul(w_a[index_a].T, a_mol[index_a])
            p_hi = torch.matmul(w_p[index_p].T, p_mol[index_p]) 
            #print('reactant weight', index_r, w_r[index_r])
            #print('reagent weight', index_a, w_a[index_a])
            #print('product weight', index_p, w_p[index_p])
            #print(w_r[index_r], w_a[index_a], w_p[index_p])
            #r_hi = torch.mean(r_mol[index_r], dim=0).view(1,-1)
            #a_hi = torch.mean(a_mol[index_a], dim=0).view(1,-1)
            #p_hi = torch.mean(p_mol[index_p], dim=0).view(1,-1)

            if self.method == 'con':
                reaction_i = torch.cat((r_hi, a_hi, p_hi), dim=1)
                reaction_i_h = self.dim_fit(reaction_i)
          
            elif self.method == 'mean':
                reaction_i = torch.cat((r_hi, a_hi, p_hi), dim=0)
                reaction_i_h = torch.mean(reaction_i, dim=0).view(1, -1)
  
            #elif self.method == 'ws':
            #    reaction_i = torch.cat((r_hi, a_hi, p_hi), dim=0)
            #    w_reaction_i = self.net_t(reaction_i)
            #    reaction_i_h = torch.mm(w_reaction_i.T, reaction_i)

            total_reaction_h = torch.cat((total_reaction_h, reaction_i_h), dim=0)  

        return total_reaction_h