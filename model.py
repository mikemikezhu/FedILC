import torch
import torch.nn.functional as F
from torch import nn
from backpack import extend

"""CIFAR CNN"""


class CifarCNN(nn.Module):

    def __init__(self):
        super(CifarCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn1 = nn.GroupNorm(8, 64)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.maxpool = nn.MaxPool2d((2, 2))
        self.lin1 = nn.Linear(128, 128)
        self.classifier = extend(nn.Linear(in_features=128,
                                           out_features=10))

    def forward(self, x):
        
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn3(x)
        x = self.maxpool(x)

        x = x.view(len(x), -1)
        features = self.lin1(x)
        logits = self.classifier(features)
        return features, logits


"""MNIST MLP"""


class MnistMLP(nn.Module):

    def __init__(self, hidden_dim):
        super(MnistMLP, self).__init__()

        lin1 = nn.Linear(2 * 14 * 14, hidden_dim)
        lin2 = nn.Linear(hidden_dim, hidden_dim)

        self.classifier = (nn.Linear(hidden_dim, 1))
        for lin in [lin1, lin2, self.classifier]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

        self._main = nn.Sequential(
            lin1, nn.ReLU(True), lin2, nn.ReLU(True))
        self.alllayers = extend(
            nn.Sequential(lin1, nn.ReLU(True), lin2,
                          nn.ReLU(True), self.classifier)
        )

    @staticmethod
    def prepare_input(input):
        return input.view(input.shape[0], 2 * 14 * 14)

    def forward(self, input):
        out = self.prepare_input(input)
        features = self._main(out)
        logits = self.classifier(features)
        return features, torch.sigmoid(logits)


"""ICU MLP"""


class IcuMLP(nn.Module):

    def __init__(self, in_features):
        super(IcuMLP, self).__init__()

        lin1 = nn.Linear(in_features, 1024)
        lin2 = nn.Linear(1024, 1024)
        lin3 = nn.Linear(1024, 512)

        self.classifier = (nn.Linear(512, 1))
        for lin in [lin1, lin2, lin3, self.classifier]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

        self._main = nn.Sequential(
            lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)
        self.alllayers = extend(
            nn.Sequential(lin1, nn.ReLU(True), lin2,
                          nn.ReLU(True), lin3, self.classifier)
        )

    def forward(self, x):
        features = self._main(x)
        logits = self.classifier(features)
        return features, torch.sigmoid(logits)
