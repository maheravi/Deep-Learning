import torch
from torch import nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from google_drive_downloader import GoogleDriveDownloader as gdd
import argparse

gdd.download_file_from_google_drive(file_id='1lIF9rV5LaBdrYNc5GN3mkzdFL2k6hlJL',
                                    dest_path='./FaceAgePredictionTL.pth')

parser = argparse.ArgumentParser()
parser.add_argument("--device", default='cpu', type=str)
parser.add_argument("--dataset", type=str)
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

model = MyModel().to(device)

model = model.to(device)
model.train(True)

# Data Preparing


transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((28, 28)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

dataset = torchvision.datasets.ImageFolder(args.dataset, transform=transform)
n = len(dataset)
n_test = int(0.1 * n)
test_set = torch.utils.data.Subset(dataset, range(n_test))
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

# compile
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_function = torch.nn.MSELoss()

device = torch.device(args.device)
model = MyModel()

model.load_state_dict(torch.load("FaceAgePredictionTL.pth"))
model.eval()

test_loss = 0.0
for img, label in test_loader:
    img = img.to(device)
    label = label.to(device)

    pred = model(img)
    loss = loss_function(preds, labels)
    loss.backward()

    test_loss += loss

total_acc = test_loss / len(test_loader)
print(f"test accuracy: {total_acc}")
