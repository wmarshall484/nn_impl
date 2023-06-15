# Get cpu, gpu or mps device for training.
import torch
from torch import nn
from torch.utils.data import DataLoader

from btc_dataset import BTCTrainDataset, BTCTestDataset

train_ds = BTCTrainDataset()
test_ds = BTCTestDataset()

batch_size = 351

train_dataloader = DataLoader(train_ds, batch_size=batch_size)
test_dataloader = DataLoader(test_ds, batch_size=batch_size)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = len(train_ds[0][0].flatten())
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, input_size, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(input_size, input_size, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(input_size, 1, dtype=torch.float64)
        )
        for w in self.parameters():
            nn.init.uniform_(w)

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load('./some_model2.tch'))
print(model)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=5e-23)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        # w = model.parameters().__next__()
        # Ws = list(model.parameters())
        # grads = [_w.grad for _w in Ws]
        # print('Ws before')
        # for w in Ws:
        #     print(w)
        # print('grad')
        # print(grad)

        optimizer.step()
        optimizer.zero_grad()
        # print('===========================================')
        # print('Ws after')
        # Ws = list(model.parameters())
        # for w in Ws:
        #     print(w)
        # print('\n\n\n')

        # if batch % 100 == 0:
        #     loss, current = loss.item(), (batch + 1) * len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return loss
    # print('grads')
    # print(grads)
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred.flatten(), y.flatten()).item()
    test_loss /= size
    correct /= size
    print("Avg test loss :",  f"{test_loss:>8f}".rjust(20))

def run():
    epochs = 0
    while True:
        loss = train(train_dataloader, model, loss_fn, optimizer)
        if epochs % 10 == 0:
            print(f"Epoch {epochs+1}\n-------------------------------")
            print(f'Avg train loss:', f'{loss:>8f}'.rjust(20))
            test(test_dataloader, model, loss_fn)
        #     x,y = test_dataloader.__iter__().__next__()
        #     x,y = x.to(device), y.to(device)
        #     with torch.no_grad():
        #         print('First preds')
        #         for x, y in zip(model(x), y):
        #             print(x.item(),y.item())
        epochs += 1
    print("Done!")

run()