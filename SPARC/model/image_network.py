from platform import release
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from os.path import join, exists, isfile, realpath, dirname
import numpy as np

from torchvision.models import resnet18,vgg16,resnet50

class Classification_network(nn.Module):
    def __init__(self,config,device):
        super(Classification_network, self).__init__()
        self.config = config
        self.pretrained_model = self.load_model()
        # self.pretrained_model.classifier[6] = nn.Linear(4096, 1)
        # print(self.pretrained_model)

        self.adjust_model()
        self.pretrained_model = self.pretrained_model.to(device)


        self.final_layer = nn.Linear(self.embedding_dim, 11).to(device)

        # w = model.conv1.weight
        # w = torch.cat([w, torch.full((64, 1, 7, 7), 0.01)], dim=1)
        # model.conv1.weight = nn.Parameter(w)

    def get_number_copies(self):
        #n_copies = np.sum([self.config["data"]["use_rgb"],self.config["data"]["use_normals"],self.config["data"]["use_alpha"]])
        n_copies = 3
        return n_copies
        
    def load_model(self):
        if self.config["model"]["type"] == 'vgg16':
            pretrained_model = vgg16(pretrained=self.config["model"]["pretrained"])
            self.embedding_dim = 4096

        elif self.config["model"]["type"] == 'resnet50':
            pretrained_model = resnet50(pretrained=self.config["model"]["pretrained"])
            self.embedding_dim = 2048

        elif self.config["model"]["type"] == 'resnet18':
            pretrained_model = resnet18(pretrained=self.config["model"]["pretrained"])
            self.embedding_dim = 512

        return pretrained_model

    def adjust_model(self):
        if self.config["model"]["type"] == 'vgg16':
            weights_first_layer = self.pretrained_model.features[0].weight
            self.pretrained_model.features[0].weight = nn.Parameter(torch.cat([weights_first_layer]*(self.get_number_copies() + 1),dim=1))
            self.pretrained_model.classifier = nn.Sequential(*list(self.pretrained_model.classifier.children())[:-1])
        elif self.config["model"]["type"] == 'resnet50':
            weights_first_layer = self.pretrained_model.conv1.weight
            self.pretrained_model.conv1.weight = nn.Parameter(torch.cat([weights_first_layer]*(self.get_number_copies() + 1),dim=1))
            self.pretrained_model.fc = Identity()
        elif self.config["model"]["type"] == 'resnet18':
            weights_first_layer = self.pretrained_model.conv1.weight
            self.pretrained_model.conv1.weight = nn.Parameter(torch.cat([weights_first_layer]*(self.get_number_copies() + 1),dim=1))
            self.pretrained_model.fc = Identity()

    def set_device(self,device):
        self.pretrained_model = self.pretrained_model.to(device)
        self.final_layer = self.final_layer.to(device)



    def forward(self,x):
        embedding = self.pretrained_model(x)
        output = self.final_layer(embedding)
        return output


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x