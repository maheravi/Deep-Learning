import torch
from torch import nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import os
import cv2
import numpy as np
import argparse
from Model import MyModel

parser = argparse.ArgumentParser()
parser.add_argument("--device", default='cpu', type=str)
parser.add_argument("--Image", type=str)
args = parser.parse_args()

# hyperparameters
latent_size = 10
disc_inp_sz = 224*224
img_size = 224
epochs = 10
batch_size = 32
lr = 0.001
width = height = 224

device = torch.device(args.device)
model = MyModel()

model.load_state_dict(torch.load('FaceAgePrediction.pth'))
model.eval()

transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((28, 28)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

img = cv2.imread(args.Image)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
tensor = transform(img).unsqueeze(0).to(device)
# tensor = tensor.permute((0, 3, 2, 1))
preds = model(tensor)

preds = preds.cpu().detach().numpy()

output = np.argmax(preds)

print(output)
