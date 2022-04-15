import torch
from torch import nn

import torchvision
from backpack import extend

"""CIFAR ResNet"""


class CifarResNet(nn.Module):

    def __init__(self, in_features, out_features):

        super(CifarResNet, self).__init__()
        self.network = torchvision.models.resnet18(pretrained=True)
        self.classifier = extend(nn.Linear(in_features=in_features,
                                           out_features=out_features))

    def forward(self, input):

        features = self.network(input)
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
        self.layer0 = nn.Linear(in_features, 32)
        self.layer1 = nn.Linear(32, 64)
        self.classifier = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.1)
        extend(self.classifier)

    def forward(self, x):
        x = torch.relu(self.layer0(x))
        x = self.dropout(x)
        x = torch.relu(self.layer1(x))
        features = self.dropout(x)
        logits = self.classifier(features)
        return features, torch.sigmoid(logits)
