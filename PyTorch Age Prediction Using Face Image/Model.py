from torch import nn


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(3, 32, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 32, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )

        self.fc = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(3136, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.ReLU(),
        )

    def forward(self, input_t):
        x = self.conv2d(input_t)
        # print(x.shape)
        return self.fc(x)