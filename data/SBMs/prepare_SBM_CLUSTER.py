import matplotlib as matplotlib
import numpy as np
import torch
import pickle
import time
import os

import matplotlib.pyplot as plt

import os
os.chdir('../../') # go to root folder of the project
print(os.getcwd())

import pickle



from data.SBMs import SBMsDatasetDGL

from data.data import LoadData
from torch.utils.data import DataLoader
from data.SBMs import SBMsDataset

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self



start = time.time()

DATASET_NAME = 'SBM_CLUSTER'
dataset = SBMsDatasetDGL(DATASET_NAME)

print('Time (sec):',time.time() - start)

print(len(dataset.train))
print(len(dataset.val))
print(len(dataset.test))

print(dataset.train[0])
print(dataset.val[0])
print(dataset.test[0])

start = time.time()

with open('C:/Users/Surface/PycharmProjects/Student_job/data/SBMs/SBM_CLUSTER.pkl', 'wb') as f:
    pickle.dump([dataset.train, dataset.val, dataset.test], f)

print('Time (sec):', time.time() - start)

DATASET_NAME = 'SBM_CLUSTER'
dataset = LoadData(DATASET_NAME)
trainset, valset, testset = dataset.train, dataset.val, dataset.test

start = time.time()

batch_size = 10
collate = SBMsDataset.collate
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=collate)

print('Time (sec):',time.time() - start)
















