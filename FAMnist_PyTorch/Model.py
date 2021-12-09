from torch import nn


class MyModel(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(MyModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dims, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.25),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),

            nn.Linear(64, output_dims),
            nn.Sigmoid(),
        )

    def forward(self, input_t):
        input_t = input_t.reshape((input_t.shape[0], 784))
        return self.fc(input_t)