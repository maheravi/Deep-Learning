from torch import nn


class MyModel(nn.Module):
    def __init__(self, input_dims):
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
            nn.Flatten(start_dim=1)

        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.Softmax(),
        )

    def forward(self, input_t):
        x = self.conv2d(input_t)
        # print(x.shape)
        return self.fc(x)
