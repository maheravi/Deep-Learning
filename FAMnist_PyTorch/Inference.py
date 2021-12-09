import cv2
import numpy as np
from Model import MyModel
import torchvision
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device", default='cpu', type=str)
parser.add_argument("--Image", type=str)
args = parser.parse_args()

latent_size = 10
disc_inp_sz = 28*28

device = torch.device(args.device)
model = MyModel(disc_inp_sz, latent_size)

model.load_state_dict(torch.load('FAMnist.pth'))
model.eval()

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0), (1))
    # torchvision.transforms.RandomHorizontalFILIP(),
])

img = cv2.imread(args.Image)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (28, 28))
tensor = transform(img).unsqueeze(0).to(device)

preds = model(tensor)

preds = preds.cpu().detach().numpy()

output = np.argmax(preds)
print(output)
