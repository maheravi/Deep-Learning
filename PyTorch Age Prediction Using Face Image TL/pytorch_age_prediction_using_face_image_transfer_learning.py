
import wandb

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

import wandb

wandb.init(project="AgePredictionTL", entity="ma_heravi")

# hyperparameters
latent_size = 10
disc_inp_sz = 224*224
img_size = 224
epochs = 10
batch_size = 32
lr = 0.001
width = height = 224
wandb.config = {
  "learning_rate": lr,
  "epochs": epochs,
  "batch_size": batch_size
}

images = []
ages = []

for image_name in os.listdir('crop_part1')[0:9000]:
    part = image_name.split('_')
    ages.append(int(part[0]))

    image = cv2.imread(f'crop_part1/{image_name}')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(image)

images = pd.Series(images, name= 'Images')
ages = pd.Series(ages, name= 'Ages')

df = pd.concat([images, ages], axis= 1)
df.head()

plt.figure(figsize=(24, 8))
plt.hist(df['Ages'], bins= 116)
plt.show()

under4 = []

for i in range(len(df)):
    if df['Ages'].iloc[i] <= 4:
        under4.append(df.iloc[i])

under4 = pd.DataFrame(under4)
under4 = under4.sample(frac= 0.3)

up4 = df[df['Ages'] > 4]

df = pd.concat([under4, up4])

df = df[df['Ages'] < 90]

plt.figure(figsize=(24, 8))
plt.hist(df['Ages'], bins= 89)
plt.show()

X = []
Y = []

for i in range(len(df)):
    df['Images'].iloc[i] = cv2.resize(df['Images'].iloc[i], (width, height))

    X.append(df['Images'].iloc[i])
    Y.append(df['Ages'].iloc[i])

X = np.array(X)
Y = np.array(Y)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2)

# X_train = X_train.astype(np.float32)

X_train = torch.tensor(X_train)
Y_train = torch.tensor(Y_train)
X_train = torch.permute(X_train, (0, 3, 2, 1))

# from torch.utils.data import TensorDataset

class MyDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.data = X
        self.target = y
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        # Normalize your data here
        if self.transform:
            x = self.transform(x)

        return x, y
    
    def __len__(self):
        return len(self.data)

transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((28, 28)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

dataset = MyDataset(X_train, Y_train, transform)
train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torchvision.models.resnet50(pretrained=True)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features,1)

ct = 0
for child in model.children():
    ct += 1
    if ct < 7:
        for param in child.parameters():
            param.requires_grad = False

model = model.to(device)

# compile
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_function = torch.nn.MSELoss()

# train
wandb.watch(model)

for epoch in range(1, epochs+1):
    train_loss = 0.0
    train_acc = 0.0
    for images, labels in train_data_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # 1- forwarding
        preds = model(images)
        # print(labels)
        # print(preds)
        # 2- backwarding
        loss = loss_function(preds, labels.float())
        loss.backward()
        # 3- Update
        optimizer.step()

        train_loss += loss
    
    total_loss = train_loss / len(train_data_loader)

    print(f"Epoch: {epoch}, Loss: {total_loss}")
    wandb.log({'epochs':  epoch,
              'loss': total_loss,
                              })

# save
torch.save(model.state_dict(), "FaceAgePredictionTL.pth")

torch.save(model.state_dict(), "/content/drive/MyDrive/FaceAgePredictionTL.pth")

