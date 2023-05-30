import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
import re
from os import walk


for (dirpath, dirnames, filenames) in walk("."):
    if '___' in dirpath:
        print("Directory path: ", dirpath)
        print("Folder name: ", dirnames)
        #print('filenames: ', filenames)


# perform transformation before reading the dataset
transform = transforms.Compose([transforms.Resize(255),
                                 transforms.CenterCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomRotation(30),
                                 transforms.ToTensor()])

dataset = datasets.ImageFolder('new-plant-diseases-dataset', transform=transform)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

iter_dataset = iter(dataloader)

batch_img, batch_labels = next(iter_dataset)
print(batch_labels)