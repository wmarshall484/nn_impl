import numpy as np
import pandas as pd

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
    size_of_individual_X = (LOOKBACK, DPOINT_DIMENSION)
    size_of_individual_y = (1, 1)
    X = np.zeros((len(data)-LOOKBACK, LOOKBACK, DPOINT_DIMENSION))
    y = np.zeros((len(data)-LOOKBACK, 1))
    for i in range(LOOKBACK, len(data)):
        X[i-LOOKBACK] = data[i-LOOKBACK:i]
        y[i-LOOKBACK] = data[i][0]
    return X, y

handle_strings(pv)


# flatten
train_X, train_y = make_X_y(pv[0:len(pv)-365])
test_X, test_y = make_X_y(pv[len(pv)-365:])

train_X_flat = np.zeros((len(train_X), DPOINT_DIMENSION * LOOKBACK))
test_X_flat = np.zeros((len(test_X), DPOINT_DIMENSION * LOOKBACK))

for i in range(len(train_X_flat)):
    train_X_flat[i] = train_X[i].flatten()

for i in range(len(test_X_flat)):
    test_X_flat[i] = test_X[i].flatten()


train_X = train_X_flat
test_X = test_X_flat


W_sizes = [
    (DPOINT_DIMENSION * LOOKBACK, DPOINT_DIMENSION * LOOKBACK),
    (DPOINT_DIMENSION * LOOKBACK, DPOINT_DIMENSION * LOOKBACK),
    (DPOINT_DIMENSION * LOOKBACK, 1)
]

sigmoid = lambda x: 1/(1 + np.exp(-x))
Ws = [(np.random.rand(*x)*.1 - .05) + 1 for x in W_sizes]
bs = [(np.random.rand(1, x[1])*.1 - .05) + 1 for x in W_sizes]

j = 0


def feed_forward(_input):
    activation = _input
    for i in range(len(W_sizes)):
        # Sigmoid
        if i == len(W_sizes) - 1:
            activation = activation @ Ws[i] + bs[i]
        else:
            # ReLU
            activation = np.maximum(activation, 0.01 * activation)
            # activation = sigmoid(activation @ Ws[i] + bs[i])
    return activation

def train():
    yhat = feed_forward(train_X)
    prev_error = sum((train_y - yhat)**2)[0]
    global j
    while True:
        batchsize = int(len(train_X)/NUM_BATCHES)
        batch = j % NUM_BATCHES
        # d_Ws = [1 + (np.random.randint(2, size=x)*2 - 1) * LEARNING_RATE for x in W_sizes]
        # d_bs = [1 + (np.random.randint(2, size=(1,x[1]))*2 - 1) * LEARNING_RATE for x in W_sizes]
        d_Ws = [(np.random.randint(2, size=x)*2 - 1) * LEARNING_RATE for x in W_sizes]
        d_bs = [(np.random.randint(2, size=(1,x[1]))*2 - 1) * LEARNING_RATE for x in W_sizes]

        for i in range(len(W_sizes)):
            Ws[i] += d_Ws[i]
            bs[i] += d_bs[i]
        try:
            xbatch=train_X[batch*batchsize:(batch+1)*batchsize]
            mdp = xbatch[len(xbatch)//2]
            s = [f'{_x}' for _x in mdp]
            yhat = feed_forward(xbatch)
            error = sum((train_y[batch*batchsize:(batch+1)*batchsize] - yhat)**2)[0]

            if error < prev_error and j > PRINT_THRESHOLD and j%100==0:
                # test_yhat = feed_forward(test_X)
                # test_error = sum((test_y - test_yhat)**2)[0]
                for x,y,iyhat in zip(train_X[-20:], train_y[-20:], yhat[-20:]):
                    p_x = ', '.join([f'{_x:5f}' for _x in x])
                    p_y = ', '.join([f'{_y:5f}' for _y in y])
                    p_yhat = ', '.join([f'{_yhat}' for _yhat in iyhat])
                    print(p_x, ' --> ', p_y, f'(guess: {p_yhat})')
                print()
                print('Ws')
                for w in Ws:
                    print(w)
                    print()
                print('bs')
                for b in bs:
                    print(b)
                    print()
                test_yhat = feed_forward(test_X)
                test_error = sum((test_y - test_yhat)**2)[0]
                print(f'TRAINING_ERROR({j}): {error/len(train_y)}', f'LEARNING RATE: {LEARNING_RATE}')
                print(f'TESTING_ERROR: {test_error/len(test_y)}')

                print()
                print(f'yhat min: {yhat.min()}')
                print(f'yhat max: {yhat.max()}')
                print(f'yhat average: {np.average(yhat)}')
                print(f'train_y average: {np.average(train_y)}')
                print()

                prev_error = error
                continue
        except:
            print('in except')
            for i in range(len(W_sizes)):
                Ws[i] -= d_Ws[i]
                bs[i] -= d_bs[i]
            raise

        for i in range(len(W_sizes)):
            Ws[i] -= d_Ws[i]
            bs[i] -= d_bs[i]

        j += 1

train()
