import torch
import torch.nn as nn
import sys
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from resnet import BasicBlock, ResNet, Bottleneck
from torch import nn


import torch
import torch.nn as nn
import torch.nn.functional as F
import functools


sys.path.insert(0, "..")
from fedml.model.cv.resnet import resnet56


class AlexNetOne(nn.Module):
    def __init__(self, num_classes=10, input_channel=1):
        super(AlexNetOne, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, inputs):
        """Forward pass of the model."""
        inputs = self.features(inputs)
        # print(inputs.size())
        inputs = inputs.reshape(inputs.size(0), 256 * 4 * 4)
        outputs = self.classifier(inputs)
        return outputs


class LeNet(nn.Module):
    # this is for the medical mnist
    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class FullyConnectedModel(nn.Module):
    def __init__(
        self, input_size=6170, hidden_sizes=[1024, 512, 256, 128], output_size=100
    ):
        super(FullyConnectedModel, self).__init__()

        # Create a list to store the layers
        layers = []

        # Add the first hidden layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())

        # Add the remaining hidden layers
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            layers.append(nn.ReLU())

        # Add the output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        # Create the sequential container
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True
    )


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find("BatchNorm") != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=True
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class Wide_ResNet(nn.Module):
    """
    Implementation is from: https://github.com/meliketoy/wide-resnet.pytorch/blob/master/networks/wide_resnet.py
    """

    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert (depth - 4) % 6 == 0, "Wide-resnet depth should be 6n+4"
        n = (depth - 4) / 6
        k = widen_factor

        print("| Wide-Resnet %dx%d" % (depth, k))
        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(
            wide_basic, nStages[1], n, dropout_rate, stride=1
        )
        self.layer2 = self._wide_layer(
            wide_basic, nStages[2], n, dropout_rate, stride=2
        )
        self.layer3 = self._wide_layer(
            wide_basic, nStages[3], n, dropout_rate, stride=2
        )
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


class Net(nn.Module):
    """Simple CNN for CIFAR10 dataset."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, inputs):
        """Forward pass of the model."""
        inputs = self.pool(F.relu(self.conv1(inputs)))
        inputs = self.pool(F.relu(self.conv2(inputs)))
        # flatten all dimensions except batch
        inputs = inputs.reshape(-1, 16 * 5 * 5)
        inputs = F.relu(self.fc1(inputs))
        inputs = F.relu(self.fc2(inputs))
        outputs = self.fc3(inputs)
        return outputs


class AlexNet(nn.Module):
    """AlexNet model for CIFAR10 dataset."""

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, inputs):
        """Forward pass of the model."""
        inputs = self.features(inputs)
        inputs = inputs.reshape(inputs.size(0), 256 * 2 * 2)
        outputs = self.classifier(inputs)
        return outputs


class ConvNet(nn.Module):
    def __init__(self, nin, nclass, scales, filters, filters_max, pooling="max"):
        super(ConvNet, self).__init__()
        self.pooling = (
            nn.MaxPool2d(kernel_size=2, stride=2)
            if pooling == "max"
            else nn.AvgPool2d(kernel_size=2, stride=2)
        )

        def nf(scale):
            return min(filters_max, filters * (2**scale))

        self.layers = nn.ModuleList([nn.Conv2d(nin, nf(0), kernel_size=3, padding=1)])

        for i in range(scales):
            self.layers.append(nn.Conv2d(nf(i), nf(i), kernel_size=3, padding=1))
            self.layers.append(nn.Conv2d(nf(i), nf(i + 1), kernel_size=3, padding=1))
            self.layers.append(self.pooling)

        self.classifier = nn.Conv2d(nf(scales), nclass, kernel_size=3, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        for layer in self.layers:
            x = F.leaky_relu(layer(x))
        x = self.classifier(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten the output for the classifier
        return x


def get_model(model_type: str, num_classes: int = 10, dataset=None) -> nn.Module:
    """Instantiate the model based on the model_type

    Args:
        model_type (str): Name of the model
        num_classes (int): Number of classes
    Returns:
        torch.nn.Module: A model
    """
    if "purchase" in dataset and model_type == "nn":
        return FullyConnectedModel(
            input_size=600, hidden_sizes=[1024, 512, 256, 128], output_size=num_classes
        )
    elif "texas" in dataset and model_type == "nn":
        return FullyConnectedModel(
            input_size=6169, hidden_sizes=[1024, 512, 256, 128], output_size=num_classes
        )
    elif dataset == "texas" and model_type == "nn3":
        return FullyConnectedModel(
            input_size=6169, hidden_sizes=[512, 256, 128], output_size=num_classes
        )
    elif dataset == "texas" and model_type == "nn2":
        return FullyConnectedModel(
            input_size=6169, hidden_sizes=[256, 128], output_size=num_classes
        )
    elif dataset == "texas" and model_type == "nn1":
        return FullyConnectedModel(
            input_size=6169, hidden_sizes=[128], output_size=num_classes
        )
    elif model_type == "CNN":
        return Net(num_classes=num_classes)
    elif model_type == "CNN16":
        # Using functools.partial to preset some of the ConvNet arguments
        ConvNetPartial = functools.partial(
            ConvNet, scales=3, filters=16, filters_max=1024, pooling="max"
        )
        return ConvNetPartial(3, num_classes)
    elif model_type == "CNN32":
        ConvNetPartial = functools.partial(
            ConvNet, scales=3, filters=32, filters_max=1024, pooling="max"
        )
        return ConvNetPartial(3, num_classes)
    elif model_type == "CNN64":
        ConvNetPartial = functools.partial(
            ConvNet, scales=3, filters=64, filters_max=1024, pooling="max"
        )
        return ConvNetPartial(3, num_classes)

    elif model_type == "alexnet":
        if dataset == "pneumonia" or dataset == "retinal":
            return AlexNetOne(num_classes=num_classes)
        elif dataset == "skin" or dataset == "kidney":
            return AlexNetOne(num_classes=num_classes, input_channel=3)
        else:
            return AlexNet(num_classes=num_classes)
    elif model_type == "wrn28-2":
        return Wide_ResNet(
            depth=28, widen_factor=2, dropout_rate=0, num_classes=num_classes
        )
    elif model_type == "wrn28-1":
        return Wide_ResNet(
            depth=28, widen_factor=1, dropout_rate=0, num_classes=num_classes
        )
    elif model_type == "wrn28-10":
        return Wide_ResNet(
            depth=28, widen_factor=10, dropout_rate=0, num_classes=num_classes
        )
    elif model_type == "resnet18":
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
    elif model_type == "resnet34":
        return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)
    elif model_type == "resent50":
        return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)
    elif model_type == "resnet101":
        return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)
    elif model_type == "resnet152":
        return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)
    elif model_type == "resnet56":
        return resnet56(class_num=num_classes)
    elif model_type == "lenet":
        return LeNet(num_classes)

    raise NotImplementedError(f"{model_type} is not implemented")


def count_learnable_layers(model_type, dataset=None):
    model = get_model(model_type=model_type, dataset=dataset)
    layer_count = 0
    for module in model.modules():
        if any(p.requires_grad for p in module.parameters()):
            layer_count += 1
    return layer_count
