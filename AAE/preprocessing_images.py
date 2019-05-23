# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 00:34:14 2019

@author: Adam
"""

import csv
import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from preprocessing_labels import training_data,top_1000_concepts,top_1000_indices,idx_to_umls,umls_to_idx,idx_to_name,name_to_idx,idx_to_name_v,name_to_idx_v


from torchvision import transforms

from torch.utils.data import Dataset, DataLoader

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

preprocessing = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])


def create_samples(filenames):
    
    samples=[]
    
    for file in filenames:
        img=np.asarray(img)
        samples.append(img)
    return samples

class Data(Dataset):
    def __init__(self, data, directory, transform=None):
        self.data = data
        self.transform = transform
        self.directory=directory
        if 'training' in directory:
            self.idx_to_name_=idx_to_name
        if 'validation' in directory:
            self.idx_to_name_=idx_to_name_v
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        fullname = self.directory+self.idx_to_name_[idx]+'.jpg'
        image = Image.open(fullname).convert('RGB')

        y_vector=np.zeros(1000)
        labels_present = np.array(self.data[idx])
        
        labels=np.in1d(top_1000_indices,labels_present)*1
        
        labels=np.where(labels==1)[0]
        
        np.put(y_vector,labels,1)
        
        
        if self.transform:
            image = self.transform(image)
        return [image, y_vector]


