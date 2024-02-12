import pandas as pd
import torch.utils.data as data
import torch
import numpy as np
from functools import partial
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from utils import integer_label_protein

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}


def label_sequence(line, smi_ch_ind, MAX_SEQ_LEN=1000):
    X = np.zeros(MAX_SEQ_LEN, np.int64())
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]
    return X


class DTIDataset(data.Dataset):
    def __init__(self, list_IDs, df, max_drug_nodes=290):

        self.list_IDs = list_IDs
        self.df = df
        self.max_drug_nodes = max_drug_nodes
        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):

        index = self.list_IDs[index]
        v_d = self.df.iloc[index]['SMILES']
        v_d = self.fc(smiles=v_d, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
        actual_node_feats = v_d.ndata.pop('h')
        num_actual_nodes = actual_node_feats.shape[0]
        num_virtual_nodes = self.max_drug_nodes - num_actual_nodes
        virtual_node_bit = torch.zeros([num_actual_nodes, 1])
        actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
        v_d.ndata['h'] = actual_node_feats
        virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 74), torch.ones(num_virtual_nodes, 1)), 1)
        v_d.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
        v_d = v_d.add_self_loop()
        v_p = self.df.iloc[index]['Protein']
        pro_len = len(v_p)
        origin_v_p = integer_label_protein(v_p, max_length=1000)
        y = self.df.iloc[index]["Y"]
        protein_max = 1000
        protein_mask = np.zeros(protein_max)
        v_p = torch.from_numpy(label_sequence(v_p, CHARPROTSET, protein_max))
        if pro_len > protein_max:
            protein_mask[:] = 1
        else:
            protein_mask[:pro_len] = 1
        return v_d, origin_v_p, v_p, protein_mask, y


class MultiDataLoader(object):
    def __init__(self, dataloaders, n_batches):
        if n_batches <= 0:
            raise ValueError("n_batches should be > 0")
        self._dataloaders = dataloaders
        self._n_batches = np.maximum(1, n_batches)
        self._init_iterators()

    def _init_iterators(self):
        self._iterators = [iter(dl) for dl in self._dataloaders]

    def _get_nexts(self):

        def _get_next_dl_batch(di, dl):
            try:
                batch = next(dl)
            except StopIteration:
                new_dl = iter(self._dataloaders[di])
                self._iterators[di] = new_dl

                batch = next(new_dl)
            return batch
        return [_get_next_dl_batch(di, dl) for di, dl in enumerate(self._iterators)]

    def __iter__(self):
        for _ in range(self._n_batches):
            yield self._get_nexts()

        self._init_iterators()

    def __len__(self):
        return self._n_batches
