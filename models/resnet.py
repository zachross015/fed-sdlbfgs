from torchvision import *
import torch

def resnet18(num_classes=None):
    if num_classes is not None:
        model = models.resnet18(pretrained=False, num_classes=num_classes)
        return model
    else:
        return models.resnet18(pretrained=True)

