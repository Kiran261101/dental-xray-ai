import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import lightning as L
from torchmetrics.classification import Accuracy

# Fake dataset
x = torch.randn(1000, 10)
y = torch.randint(0, 2, (1000,))

dataset = TensorDataset(x, y)

# Split dataset
train_size = 800
val_size = 200

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32)
val_loader = DataLoader(val_dataset, batch_size=32)

# Lightning model
class SimpleModel(L.LightningModule):

    def __init__(self):
        super().__init__()

        self.layer = nn.Linear(10, 2)

        self.loss_fn = nn.CrossEntropyLoss()

        self.accuracy = Accuracy(task="multiclass", num_classes=2)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):

        x, y = batch

        preds = self(x)

        loss = self.loss_fn(preds, y)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):

        x, y = batch

        preds = self(x)

        loss = self.loss_fn(preds, y)

        acc = self.accuracy(preds, y)

        self.log("val_loss", loss)

        self.log("val_acc", acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

# Create model
model = SimpleModel()

# Trainer
trainer = L.Trainer(max_epochs=5)

# Train
trainer.fit(model, train_loader, val_loader)