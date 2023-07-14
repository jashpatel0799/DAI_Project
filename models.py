from typing import List
import torch
from torch import nn
import numpy as np
from torchvision import models
from torchvision.models import ResNet18_Weights,ResNet50_Weights,VGG16_Weights


class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta: 
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True
                
                

class Resnet18(nn.Module):
    def __init__(self,out_shape:int = 1000) -> None:
        super().__init__()
        self.resnet = models.resnet18()
        self.resnet.fc = nn.Linear(512,out_shape)
    
    def forward(self,x):
        return self.resnet(x)
    
    
class PretrainedResnet18(nn.Module):
    def __init__(self,out_shape:int = 1000) -> None:
        super().__init__()
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # freeze all layers except last fc layer
        for parms in self.resnet.parameters():
            parms.requires_grad = False
        
        self.resnet.fc = nn.Linear(512,out_shape)
    
    def forward(self,x):
        return self.resnet(x)
    

class Resnet50(nn.Module):
    def __init__(self,out_shape:int = 1000) -> None:
        super().__init__()
        self.resnet = models.resnet50()
        self.resnet.fc = nn.Linear(2048,out_shape)
    
    def forward(self,x):
        return self.resnet(x)
    
    
class PretrainedResnet50(nn.Module):
    def __init__(self,out_shape:int = 1000) -> None:
        super().__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # freeze all layers except last fc layer
        for parms in self.resnet.parameters():
            parms.requires_grad = False
        
        self.resnet.fc = nn.Linear(2048,out_shape)
    
    def forward(self,x):
        return self.resnet(x)

class EfficentNetB0(nn.Module):
    def __init__(self,out_shape:int = 1000) -> None:
        super().__init__()
        self.effnet = models.efficientnet_b0()
        self.effnet.classifier = nn.Linear(1280,out_shape)
    
    def forward(self,x):
        return self.effnet(x)


class MobileNetV2(nn.Module):
    def __init__(self,out_shape:int = 1000) -> None:
        super().__init__()
        self.mobilenet = models.mobilenet_v2()
        self.mobilenet.classifier[1] = nn.Linear(1280,out_shape)
    
    def forward(self,x):
        return self.mobilenet(x)
    


class VGG16(nn.Module):
    def __init__(self,out_shape:int = 1000) -> None:
        super().__init__()
        self.vgg = models.vgg16()
        self.vgg.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=out_shape, bias=True),
        )
    
    def forward(self,x):
        return self.vgg(x)
    
    
class PretrainedVGG16(nn.Module):
    def __init__(self,out_shape:int = 1000) -> None:
        super().__init__()
        self.vgg = models.vgg16(weights=VGG16_Weights.DEFAULT)
        
        # freeze all layers except last clf layer
        for parms in self.vgg.parameters():
            parms.requires_grad = False
        
        self.vgg.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=out_shape, bias=True),
        )
    
    def forward(self,x):
        return self.vgg(x)
    


class VITBase16(nn.Module):
    def __init__(self,out_shape:int = 1000) -> None:
        super().__init__()
        self.vit = models.vit_b_16()
        self.vit.heads = nn.Sequential(
            nn.Linear(in_features=768, out_features=out_shape, bias=True),
        )
    
    def forward(self,x):
        return self.vit(x)