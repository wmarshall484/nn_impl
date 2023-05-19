import numpy as np
import pandas as pd

lookup = dict(K=1_000, M=1_000_000, B=1_000_000_000)
LEARNING_RATE = 0.01

df = pd.read_csv('/home/will/Downloads/bitcoin_prices.csv')
pv = df[['Price', 'Vol.']]
pv = pv[::-1].to_numpy()


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
    size_of_individual_X = (100, 2)
    size_of_individual_y = (1, 1)
    X = np.zeros((len(data)-100, 100, 2))
    y = np.zeros((len(data)-100, 1))
    for i in range(100, len(data)):
        X[i-100] = data[i-100:i]
        y[i-100] = data[i][0]
    return X, y

handle_strings(pv)


# flatten
train_X, train_y = make_X_y(pv[0:len(pv)-365])
test_X, test_y = make_X_y(pv[len(pv)-365:])

train_X_flat = np.zeros((len(train_X), 200))
test_X_flat = np.zeros((len(test_X), 200))

for i in range(len(train_X_flat)):
    train_X_flat[i] = train_X[i].flatten()

for i in range(len(test_X_flat)):
    test_X_flat[i] = test_X[i].flatten()


train_X = train_X_flat
test_X = test_X_flat


W_sizes = [(200, 400), (400, 800), (800, 400), (400, 200), (200, 1)]

sigmoid = lambda x: 1/(1 + np.exp(-x))
Ws = [np.random.rand(*x) - 0.5 for x in W_sizes]
bs = [np.random.rand(1, x[1]) - 0.5 for x in W_sizes]

def feed_forward(_input):
    activation = _input
    for i in range(len(W_sizes)):
        # ReLU
        # activation = np.maximum(activation @ Ws[i] + bs[i], 0)

        # Sigmoid
        activation = sigmoid(activation @ Ws[i] + bs[i])
        print(activation)
    return activation


def train():
    yhat = feed_forward(train_X)
    prev_error = sum((train_y - yhat)**2)[0]

    for i in range(100000):
        d_Ws = [(np.random.rand(*x) - 0.5) * LEARNING_RATE for x in W_sizes]
        d_bs = [(np.random.rand(1, x[1]) - 0.5) * LEARNING_RATE for x in W_sizes]

        for i in range(len(W_sizes)):
            Ws[i] += d_Ws[i]
            bs[i] += d_bs[i]

        yhat = feed_forward(test_X)
        test_error = sum((test_y - yhat)**2)[0]

        yhat = feed_forward(train_X)
        error = sum((train_y - yhat)**2)[0]
        print(f'TRAINING_ERROR: {error}     TESTING_ERROR: {test_error}')
    
        if error < prev_error:
            prev_error = error
            continue

        for i in range(len(W_sizes)):
            Ws[i] -= d_Ws[i]
            bs[i] -= d_bs[i]
