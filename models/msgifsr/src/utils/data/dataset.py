import itertools
import numpy as np
import pandas as pd


def create_index(sessions):
    # lens = np.fromiter(map(len, sessions), dtype=np.long)
    lens = np.fromiter(map(len, sessions), dtype=np.int64)
    # session_idx = np.repeat(np.arange(len(sessions)), lens - 1)
    session_idx = np.arange(len(sessions))
    # label_idx = map(lambda l: range(1, l), lens)
    label_idx = lens - 1
    # label_idx = itertools.chain.from_iterable(label_idx)
    # label_idx = np.fromiter(label_idx, dtype=np.long)
    # label_idx = np.fromiter(label_idx, dtype=np.int64)
    idx = np.column_stack((session_idx, label_idx))
    # exit()
    return idx


def read_sessions(filepath):
    # sessions = pd.read_csv(filepath, sep='\t', header=None, squeeze=True)
    sessions = pd.read_csv(filepath, sep='\t', header=None)
    sessions = sessions.squeeze()
    sessions = sessions.apply(lambda x: list(map(int, x.split(',')))).values
    return sessions


def read_dataset(dataset_dir):
    train_sessions = read_sessions(dataset_dir / 'train.txt')
    test_sessions = read_sessions(dataset_dir / 'test.txt')
    with open(dataset_dir / 'num_items.txt', 'r') as f:
        num_items = int(f.readline())
    return train_sessions, test_sessions, num_items

class AugmentedDataset:
    def __init__(self, sessions, sort_by_length=False):
        self.sessions = sessions[0]
        # self.graphs = graphs
        index = create_index(sessions[0])  # columns: sessionId, labelIndex


        if sort_by_length:
            # sort by labelIndex in descending order
            ind = np.argsort(index[:, 1])[::-1]
            index = index[ind]
        self.index = index
        self.explicit = np.asarray(sessions[1])
        self.latent = np.asarray(sessions[2])

    def __getitem__(self, idx):
        # print(idx)
        sid, lidx = self.index[idx]
        seq = self.sessions[sid][:lidx]
        label = self.sessions[sid][lidx]
        # seq = self.sessions[idx][:-1]
        # label = self.sessions[idx][-1]

        explicit = self.explicit[sid]
        latent = self.latent[sid]

        # print(seq)
        # print(label)
        # print(label)
        # exit()
        
        return seq, label, explicit, latent #,seq

    def __len__(self):
        return len(self.index)
