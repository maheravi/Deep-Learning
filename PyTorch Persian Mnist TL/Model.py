from torch import nn
import torchvision


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        model = torchvision.models.resnet50(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 10)

        ct = 0
        for child in model.children():
            ct += 1
            if ct < 7:
                for param in child.parameters():
                    param.requires_grad = False
