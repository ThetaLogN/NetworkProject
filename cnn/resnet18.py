import torch
import torch.nn as nn
import torchvision.models as models

def get_standard_resnet18(pretrained=True, num_classes=2):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=None)
    
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model
