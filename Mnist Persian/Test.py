import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from Model import MyModel

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device", default='cpu', type=str)
parser.add_argument("--dataset", type=str)
args = parser.parse_args()

# hyperparameters
latent_size = 10
disc_inp_sz = 28 * 28
img_size = 28
epochs = 10
batch_size = 32
lr = 0.001

device = torch.device(args.device)

model = MyModel(disc_inp_sz, latent_size).to(device)

model = model.to(device)
model.train(True)


def calc_acc(preds, labels):
    _, preds_max = torch.max(preds, 1)
    acc = torch.sum(preds_max == labels.data, dtype=torch.float64) / len(preds)
    return acc

# Data Preparing


transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((28, 28)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

Dataset = torchvision.datasets.ImageFolder(args.dataset, transform=transform)
n = len(Dataset)  # total number of examples
n_test = int(0.1 * n)  # take ~10% for test
test_set = torch.utils.data.Subset(Dataset, range(n_test))  # take first 10%
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

device = torch.device(args.device)
model = MyModel(disc_inp_sz)

model.load_state_dict(torch.load("PersianMnistFinal.pth"))
model.eval()

test_acc = 0.0
for img, label in test_loader:
    img = img.to(device)
    label = label.to(device)

    pred = model(img)
    test_acc += calc_acc(pred, label)

total_acc = test_acc / len(test_loader)
print(f"test accuracy: {total_acc}")
