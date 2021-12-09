import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from Model import MyModel

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device", default='cpu', type=str)
args = parser.parse_args()

# hyperparameters
latent_size = 10
disc_inp_sz = 28 * 28
img_size = 28
epochs = 10
batch_size = 32
lr = 0.001

device = torch.device("cuda")
# device = torch.device("cpu")

model = MyModel(disc_inp_sz, latent_size).to(device)

model = model.to(device)
model.train(True)


def calc_acc(preds, labels):
    _, preds_max = torch.max(preds, 1)
    acc = torch.sum(preds_max == labels.data, dtype=torch.float64) / len(preds)
    return acc


# Data Preparing

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0), (1))
])

dataset = torchvision.datasets.FashionMNIST("./dataset", train=True, download=True, transform=transform)
train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# compile
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_function = torch.nn.CrossEntropyLoss()

# train

for epoch in range(1, epochs + 1):
    train_loss = 0.0
    train_acc = 0.0
    for images, labels in train_data_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # 1- forwarding
        preds = model(images)
        # 2- backwarding
        loss = loss_function(preds, labels)
        loss.backward()
        # 3- Update
        optimizer.step()

        train_loss += loss
        train_acc += calc_acc(preds, labels)

    total_loss = train_loss / len(train_data_loader)
    total_acc = train_acc / len(train_data_loader)

    print(f"Epoch: {epoch}, Loss: {total_loss}, Acc: {total_acc}")

# save
torch.save(model.state_dict(), "FAMnist.pth")
