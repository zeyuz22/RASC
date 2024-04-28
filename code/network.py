import torch
import torch.nn as nn
import torchvision
from torchvision import models
import math
import torch.nn.functional as F


class ResNet50Fc(nn.Module):

    def __init__(self, bottleneck_dim=512, class_num=1000):
        super(ResNet50Fc, self).__init__()
        self.model_resnet = models.resnet50(pretrained=True)

        model_resnet = self.model_resnet
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool

        self.classifier = Classifier(inc=model_resnet.fc.in_features, num_emb=512, num_class=class_num)

    def forward(self, x, getemb=False, getfeat=False, justclf=False):
        if not justclf:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
        else:
            x = x
        feat = torch.flatten(x, 1)
        out = self.classifier(feat, getemb=getemb)
        out = (feat, out) if getfeat else out
        return out

    def trainable_parameters(self):
        backbone_layers = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        backbone_params = []
        for layer in backbone_layers:
            backbone_params += [param for param in layer.parameters()]
        classifier_params = list(self.classifier.parameters())

        return backbone_params, classifier_params


class Classifier(nn.Module):
    def __init__(self, inc=2048, num_emb=512, num_class=64):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(inc, num_emb)
        self.bn = nn.BatchNorm1d(num_emb)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(num_emb, num_class)
        self.num_class = num_class

    def forward(self, x, getemb=False):
        featwonorm = self.fc1(x)
        featwonorm = self.bn(featwonorm)
        emb = self.relu(featwonorm)
        x_out = self.fc2(emb)
        if getemb:
            remb = emb
            return remb, x_out
        else:
            return x_out
