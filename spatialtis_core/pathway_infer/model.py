import torch
import torch.nn.functional as F
from torch.nn import Linear, Flatten
from torch_geometric.nn import GCNConv, global_max_pool
from pytorch_lightning.core.lightning import LightningModule


class GCNG(LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.conv1 = GCNConv(2, 32)
        self.conv2 = GCNConv(32, 32)
        self.dense1 = Linear(32, 512)
        self.dense2 = Linear(512, 1)
        self.flatten = Flatten()

        self.lr = lr
        self.correct = 0
        self.test_data_len = 0
        self.acc = 0

    def forward(self, data):
        # x, edge_index = data.x, data.edge_index
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = global_max_pool(x, batch)
        x = self.flatten(x)
        x = self.dense1(x)
        x = F.elu(x)
        x = self.dense2(x)

        return torch.sigmoid(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, data, batch_idx):
        x = self(data)
        loss_in = x.flatten()
        loss_out = data.y
        loss = F.binary_cross_entropy(loss_in, loss_out)
        return loss

    def test_step(self, data, batch_idx):
        x = self(data)
        pred = x.argmax(dim=1)  # Use the class with highest probability.
        self.correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        self.test_data_len += len(data.y)
        self.acc = self.correct / self.test_data_len
        return self.acc


