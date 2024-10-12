import os
import csv
import numpy as np
import pandas as pd
import pickle  

import torch
from rdkit import Chem
from rdkit import RDConfig
from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import ChemicalFeatures,AllChem
from rdkit.Chem.rdmolops import FastFindRings
from collections import defaultdict

"""
get the node features
"""

def get_node_feat(mol):
    mol = Chem.MolFromSmiles(mol)
    #mol = Chem.AddHs(mol)
    #AllChem.EmbedMolecule(mol)
    #AllChem.UFFOptimizeMolecule(mol)
    
    atom_feats_dict = defaultdict(list)
    is_donor = defaultdict(int)
    is_acceptor = defaultdict(int)
    fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    mol.UpdatePropertyCache()
    FastFindRings(mol)
    mol_featurizer = ChemicalFeatures.BuildFeatureFactory(fdef_name)
    mol_feats = mol_featurizer.GetFeaturesForMol(mol)
    
    for i in range(len(mol_feats)):
        if mol_feats[i].GetFamily() == 'Donor':
            node_list = mol_feats[i].GetAtomIds()
            for u in node_list:
                is_donor[u] = 1
        elif mol_feats[i].GetFamily() == 'Acceptor':
            node_list = mol_feats[i].GetAtomIds()
            for u in node_list:
                is_acceptor[u] = 1
                
    num_atoms = mol.GetNumAtoms()
    atoms = mol.GetAtoms()
    #AllChem.ComputeGasteigerCharges(mol) #charges
    #conformer = mol.GetConformer() #3d positio

    h_u = []    
    for u in range(num_atoms):
        ato = mol.GetAtomWithIdx(u)
        atom_type = ato.GetAtomicNum() #atomic number
        aromatic = ato.GetIsAromatic()
        #hybridization = ato.GetHybridization()
        #num_h = ato.GetTotalNumHs() #Returns the total number of Hs (explicit and implicit) on the atom.
        #atom_feats_dict['pos'].append(torch.FloatTensor(conformer.GetAtomPosition(u)))
        atom_feats_dict['node_type'].append(atom_type)   
        h_u.append(atom_type)
        h_u.append(is_acceptor[u])
        h_u.append(is_donor[u])
        h_u.append(int(aromatic))
        #h_u += [
        #    int(hybridization == x)
        #    for x in (Chem.rdchem.HybridizationType.SP,
        #              Chem.rdchem.HybridizationType.SP2,
        #              Chem.rdchem.HybridizationType.SP3)
        #]
        #h_u.append(num_h)
        #h_u += conformer.GetAtomPosition(u)
        #h_u.append(ato.GetDoubleProp('_GasteigerCharge')) #charges
    h_u=np.array(h_u).reshape(-1,4)
    return h_u, Chem.GetAdjacencyMatrix(mol, useBO = True), num_atoms

#def get_node_feat_fromfile(molecule, reaction_index, molecule_index):
#    
#    h_u=np.array(h_u).reshape(-1,4)
#    return h_u, Chem.GetAdjacencyMatrix(mol, useBO = True), num_atoms
# get the molecule features(node, edge) of each molecule
def get_feat_dataset(molecule_list, y):
    molecule_feats_list = []
    for index, molecule in enumerate(molecule_list):
        if(index%1000==0):
            print(index)
        atom_feats, bond_feats = get_feat(molecule)
        atom_feats = torch.tensor(atom_feats, dtype=torch.float32)
        bond_feats = torch.tensor(bond_feats, dtype=torch.float32)
        yie = torch.tensor(y[index])
        molecule_feats_list.append({'atom_feature':torch.tensor(atom_feats), 'bond_type':torch.tensor(bond_feats), 'num_atom':atom_feats.shape[0], 'yield': yie})
    return molecule_feats_list




#if __name__ == '__main__':
#    preprocess()