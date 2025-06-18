import numpy as np 
import pandas as pd 
import scipy.sparse as sp

import torch.utils.data as data
import torch


def load_all(cfg, test_num=100):
    """ We load all the three file here to save time in each epoch. """
    train_rating = cfg.path + '{}.train.rating'.format(cfg.dataset)
    test_rating = cfg.path + '{}.test.rating'.format(cfg.dataset)
    test_negative = cfg.path + '{}.test.negative'.format(cfg.dataset)
    train_data = pd.read_csv(
        train_rating, 
        sep='\t', header=None, names=['user', 'item'], 
        usecols=[0, 1], dtype={0: np.int32, 1: np.int32})

    user_num = train_data['user'].max() + 1
    item_num = train_data['item'].max() + 1
    
#     train_mat = train_data.groupby('user').apply(lambda group: sorted(set(range(item_num)) - set(group['item'].tolist())))

    train_data = train_data.values.tolist()

    # load ratings as a dok matrix
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for x in train_data:
        train_mat[x[0], x[1]] = 1.0

    test_data = []
    with open(test_negative, 'r') as fd:
        line = fd.readline()
        while line != None and line != '':
            arr = line.split('\t')
            u = eval(arr[0])[0]
            test_data.append([u, eval(arr[0])[1]])
#             train_mat[u, eval(arr[0])[1]] = 1.0
            for i in arr[1:]:
                test_data.append([u, int(i)])
            line = fd.readline()
    return train_data, test_data, user_num, item_num, train_mat


class NCFData(data.Dataset):
    def __init__(self, features, 
                num_item, train_mat=None, num_ng=0, is_training=None):
        super(NCFData, self).__init__()
        """ Note that the labels are only useful when training, we thus 
            add them in the ng_sample() function.
        """
        self.data = features
        self.nb_items = num_item
        self.train_mat = train_mat
        self.nb_neg = num_ng
        self.is_training = is_training


    def __len__(self):
        if self.is_training:
            return (self.nb_neg + 1) * len(self.data)
        else:
            return len(self.data)

    def __getitem__(self, idx):
        if idx % (self.nb_neg + 1) == 0:
            if self.is_training:
                idx = idx // (self.nb_neg + 1)
            return (self.data[idx][0], self.data[idx][1]), np.ones(1, dtype=np.float32)  # noqa: E501
        else:
            if self.is_training:
                idx = idx // (self.nb_neg + 1)
                u = self.data[idx][0]
#                 j = torch.LongTensor(1).random_(0, len(self.train_mat[u])).item()
#                 return (u, self.train_mat[u][j]), np.zeros(1, dtype=np.float32)
                j = torch.LongTensor(1).random_(0, self.nb_items).item()
                while (u, j) in self.train_mat:
                    j = torch.LongTensor(1).random_(0, self.nb_items).item()
                return (u, j), np.zeros(1, dtype=np.float32)
            else:
                return (self.data[idx][0], self.data[idx][1]), np.zeros(1, dtype=np.float32)