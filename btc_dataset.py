import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from torchvision.transforms import ToTensor

lookup = dict(K=1_000, M=1_000_000, B=1_000_000_000)
LEARNING_RATE = 0.00001

df = pd.read_csv('./BTC-USD.csv')
pv = df[['Close', 'Volume']]
pv = pv.to_numpy()

LOOKBACK = 2
NUM_BATCHES = 3
PRINT_THRESHOLD = 0

# Dimension of the data point
DPOINT_DIMENSION = len(pv[0])

def handle_strings(pv):
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            if 'K' in str(pv[i][j]) or 'M' in str(pv[i][j]) or 'B' in str(pv[i][j]):
                pv[i][j] = float(pv[i][j][:-1]) * lookup[pv[i][j][-1]]
            if ',' in str(pv[i][j]):
                pv[i][j] = float(str(pv[i][j]).replace(',', ''))
            else:
                pv[i][j] = float(pv[i][j])


def make_X_y(data):
    X = np.zeros((len(data)-LOOKBACK, LOOKBACK, DPOINT_DIMENSION))
    y = np.zeros((len(data)-LOOKBACK,))
    for i in range(LOOKBACK, len(data)):
        X[i-LOOKBACK] = data[i-LOOKBACK:i]
        y[i-LOOKBACK] = data[i][0]
    return X, y

handle_strings(pv)


# flatten
train_X, train_y = make_X_y(pv[0:len(pv)-365])
test_X, test_y = make_X_y(pv[len(pv)-365:])

train_X = [torch.as_tensor(_x.flatten()) for _x in train_X]
test_X = [torch.as_tensor(_x.flatten()) for _x in test_X]
train_y = torch.as_tensor(train_y).reshape(len(train_y), 1)
test_y = torch.as_tensor(test_y).reshape(len(test_y), 1)

class BTCTrainDataset(Dataset):
    def __len__(self):
        return len(train_X)

    def __getitem__(self, idx):
        return train_X[idx], train_y[idx]


class BTCTestDataset(Dataset):
    def __len__(self):
        return len(test_X)

    def __getitem__(self, idx):
        return test_X[idx], test_y[idx]
