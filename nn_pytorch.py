import glob
import datetime
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from typing import List

from btc_dataset import BTCDataset

class NeuralNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_layers: List[int]):
        super().__init__()
        layers = self._build_layers(input_size, output_size, hidden_layers)
        self.flatten = nn.Flatten()

        self.linear_relu_stack = nn.Sequential(
            *layers,
        )
        for w in self.parameters():
            nn.init.uniform_(w)

    def _build_layers(self, input_size, output_size, hidden_layers):
        layer_sizes = [input_size] + hidden_layers
        layer_matrix_dims = list(zip(layer_sizes, layer_sizes[1:]))
        layer_matrix_dims += (layer_sizes[-1], output_size),
        layers = []
        for m in layer_matrix_dims[:-1]:
            layers.append(nn.Linear(m[0], m[1], dtype=torch.float64))
            layers.append(nn.ReLU())
        last_matrix_dim = layer_matrix_dims[-1]
        layers.append(nn.Linear(last_matrix_dim[0], last_matrix_dim[1], dtype=torch.float64))
        return layers
    

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


class NNTrainHarness:
    def __init__(self, test_split=0.2):
        self.test_split = test_split
        self._setup_data()

    def train(self):
        self.model.train()
        total_loss = 0
        for batch, (X, y) in enumerate(self.train_dataloader):
            X, y = X.to(self.device), y.to(self.device)

            # Compute prediction error
            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            total_loss += loss

            # Backpropagation
            loss.backward()

            #print("====WEIGHTS BEFORE====")
            #print_ws()

            self.optimizer.step()
            self.optimizer.zero_grad()

            #print('====WEIGHTS AFTER====')
            #print_ws()


        return total_loss/self.train_size

    def test(self):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for X, y in self.test_dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += self.loss_fn(pred.flatten(), y.flatten()).item()
        test_loss /= self.test_size
        return test_loss

    def run(self, learning_rate=5e-23, batch_size=None):
        self._setup_runtime(batch_size, learning_rate)

        epoch = 0
        while True:
            try:
                loss = self.train()
                if epoch % 100 == 0:
                    self._print_test(epoch, loss)
            except KeyboardInterrupt:
                to_save = input('save model? [y/n]: ')
                if to_save == 'y':
                    torch.save(self.model.state_dict(), self.model_name)
                    print("Saved.")
                    break
                else:
                    print("Not saved.")
                    break
            epoch += 1

    # Define model
    def define_model(self, hidden_layers, model_name=None):
        if not model_name:
            curr_date = str(datetime.datetime.now()).split(' ')[0]
            existing_versions = [int(name.split('_')[1].split('.')[0]) for name in glob.glob(f'{curr_date}*.tch')]
            if not existing_versions:
                new_version = 0
            else:
                new_version = max(existing_versions) + 1
            model_name = f'{curr_date}_{new_version}.tch'
        self.model_name = model_name
        dpoint_size = len(self.dataset[0][0].flatten())
        self.model = NeuralNetwork(
            input_size=dpoint_size,
            output_size=1,
            hidden_layers=hidden_layers
        ).to(self.device)
        try:
            self.model.load_state_dict(torch.load(self.model_name))
        except Exception:
            print("Couldn't find model. Creating from scratch.")
        print(self.model)

    def _setup_data(self):
        dataset = BTCDataset()
        self.dataset = dataset
        indices = list(range(len(dataset)))
        np.random.shuffle(indices)

        train_cutoff = int(len(dataset) * (1-self.test_split))
        self.train_size = len(indices[:train_cutoff])
        self.test_size = len(indices[train_cutoff:])
        self.train_sampler = SubsetRandomSampler(indices[:train_cutoff])
        self.test_sampler = SubsetRandomSampler(indices[train_cutoff:])

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using {self.device} device")

    def _setup_runtime(self, batch_size, learning_rate):
        if not batch_size:
            batch_size = self.train_size
        self.batch_size = batch_size

        generator = torch.Generator().manual_seed(1234)
        self.train_dataloader = DataLoader(self.dataset, batch_size=batch_size, sampler=self.train_sampler, generator=generator)
        self.test_dataloader = DataLoader(self.dataset, batch_size=self.test_size, sampler=self.test_sampler, generator=generator)

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

    def _print_test(self, epoch, loss):
        print(f"Epoch {epoch+1}\n-------------------------------")
        print(f'Avg train loss:', f'{loss:>8f}'.rjust(20))
        test_loss = self.test()
        print("Avg test loss :",  f"{test_loss:>8f}".rjust(20))
        if not (hasattr(self, 'x') and hasattr(self, 'y') and self.x is not None and self.y is not None):
            x,y = self.test_dataloader.__iter__().__next__()
            x,y = x[:10], y[:10]
            self.x,self.y = x.to(self.device), y.to(self.device)
        with torch.no_grad():
            print('First preds')
            print('y'.ljust(20), 'yhat'.rjust(20))
            for yhat, y in zip(self.model(self.x), self.y):
                print(f'{y.item():>8f}'.ljust(20),f'{yhat.item():>8f}'.rjust(20))

    def _print_batch_loss(self, loss, batch):
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * self.batchsize
            print(f"loss: {loss:>7f}  [{current:>5d}/{self.train_size/self.batchsize:>5d}]")



nn_harness = NNTrainHarness()

dpoint_size = len(nn_harness.dataset[0][0].flatten())
nn_harness.define_model([dpoint_size, dpoint_size], model_name='2023-06-19_0.tch')

nn_harness.run(learning_rate=1e-23, batch_size=2*2*2*281)