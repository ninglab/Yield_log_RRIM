from rdkit import RDLogger
import numpy as np
import pandas as pd
import torch
import sys
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
import rdkit.Chem.QED as QED
import networkx as nx

def similarity(a, b, sim_type):
    if a is None or b is None: 
        return 0.0
    amol = Chem.MolFromSmiles(a)
    bmol = Chem.MolFromSmiles(b)
    if amol is None or bmol is None:
        return 0.0

    if sim_type == "binary":
        fp1 = AllChem.GetMorganFingerprintAsBitVect(amol, 2, nBits=2048, useChirality=False)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(bmol, 2, nBits=2048, useChirality=False)
    else:
        fp1 = AllChem.GetMorganFingerprint(amol, 2, useChirality=False)
        fp2 = AllChem.GetMorganFingerprint(bmol, 2, useChirality=False)

    sim = DataStructs.TanimotoSimilarity(fp1, fp2)
    
    return sim


RDLogger.DisableLog('rdApp.*')   
df1 = pd.read_csv('/fs/ess/PCON0041/xiaohu/t5chem/models/uspto500mt/predictions.csv')
data = pd.read_csv('/fs/ess/PCON0041/xiaohu/MAT/Data/preprocessed_datasets/USPTO/USPTO500MT_test_processed_100.csv')
t5target = df1['target'].values / 100
t5pred = df1['prediction'].values / 100
ourpred = np.array(torch.load('/fs/ess/PCON0041/xiaohu/MAT/results/final/uspto/womodels/result.pt',map_location=torch.device('cpu'))[:,0])
#index = 0
#print(index)
#temp_data = sep_data[index]
temp_data = data
l = temp_data.shape[0]

reag_unique = temp_data['reagents'].unique()
reac_unique = temp_data['reactants'].unique()
print(len(reag_unique))
print(len(reac_unique))

m_reactants = temp_data['reactants'].values
m_reagents = temp_data['reagents'].values
m_products = temp_data['products'].values


print(l)
sys.stdout.flush()
reaction_sim = np.zeros((l,l))
t_yield_sim = np.zeros((l,l))
t5_yield_sim = np.zeros((l,l))
our_yield_sim = np.zeros((l,l))

for i in range(l):
    ri_r = m_reactants[i] 
    ri_a = m_reagents[i] 
    #ri = m_reactants[i] + '.' + m_reagents[i] + '.' + m_products[i]
    for j in range(i):
        if((i%500==0)&(j%500==0)):
            print(i,j)
            sys.stdout.flush()
        #rj = m_reactants[j] + '.' + m_reagents[j] + '.' + m_products[j] 
        rj_r = m_reactants[j] 
        rj_a = m_reagents[j] 
        #reaction_sim[i,j] = (similarity(m_reagents[i], m_reagents[j], '') + similarity(m_reagents[i], m_reagents[j], '') + similarity(m_reagents[i], m_reagents[j], '')) / 3 
        reaction_sim[i,j] = (similarity(ri_r, rj_r, '') + similarity(ri_a, rj_a, '')) / 2
        #t_yield_sim[i,j] = np.abs(t5target[i] - t5target[j])
        #t5_yield_sim[i,j] = np.abs(t5pred[i] - t5pred[j])
        #our_yield_sim[i,j] = np.abs(ourpred[i] - ourpred[j])
with open('reaction_similarity.npy', 'wb') as f:
    np.save(f, reaction_sim)