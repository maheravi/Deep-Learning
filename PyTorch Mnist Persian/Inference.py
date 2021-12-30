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

disc_inp_sz = 28*28

device = torch.device(args.device)
model = MyModel(disc_inp_sz)

model.load_state_dict(torch.load('PersianMnistFinal.pth'))
model.eval()

transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((28, 28)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

img = cv2.imread(args.Image)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img.shape)
tensor = transform(img).unsqueeze(0).to(device)

preds = model(tensor)

preds = preds.cpu().detach().numpy()

output = np.argmax(preds)

print(output)
