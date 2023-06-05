import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
import re
from os import walk


# for (dirpath, dirnames, filenames) in walk("."):
#     if '___' in dirpath:
#         print("Directory path: ", dirpath)
#         print("Folder name: ", dirnames)


def read_data(directory='.',
              batch_size=32,
              shuffle=True,
              test=False,
              seed=1)->torch.utils.data.dataloader.DataLoader:
    '''
    Read images from directory.
    The label associated with each image is the directory name.
    
    ------------------------------------------------------------
    Returns:
        Dataloader associated with the directory.
    '''
    # initialize random seed
    torch.manual_seed(seed)

    # transformations to be applied to the dataset
    transform = transforms.Compose([transforms.Resize(256),
                                    #transforms.CenterCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(10),
                                    transforms.ToTensor()])

    # read images from folder
    dataset = datasets.ImageFolder(directory, transform=transform)

    # dataloader
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def show_image(image:torch.tensor,
              label:int,
              cols=4):
    '''
    Given an image or a batch as a torch.tensor, show the image.
    '''
    # if single image
    if len(image.shape) == 3:
        plt.title(f"{label}", fontsize=14)
        plt.imshow(torch.permute(image, dims=(1,2,0)))
        plt.axis('off')
        plt.show()

    # else if batch is passed
    elif len(image.shape) == 4:
        batch_size = image.shape[0]

        fig, axis = plt.subplots(int(np.ceil(batch_size/cols)), cols, sharex=True, sharey=True, figsize=(20,10))
        for i in range(len(axis.ravel())):
            axis.ravel()[i].imshow(torch.permute(image[i], dims=(1,2,0)))
            axis.ravel()[i].axis('off')
        plt.axis('off')
        plt.show()
