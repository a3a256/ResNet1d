import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings
warnings.filterwarnings("ignore")

from datasets import load_dataset

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from copy import deepcopy

from create_dataset import generate_dataset
from data_preprocessing import PreProcess

from resnet1d import *

def training_model_whole(train_dl, val_dl, model, criterion, optimizer, scheduler, epochs):
    EPOCHS = epochs
    best_mode = deepcopy(model)
    best_acc = 0

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    for i in range(1, EPOCHS+1):
        model.train()
        
        diff = 0
        acc = 0
        total = 0
        
        for data, target in train_dl:
            optimizer.zero_grad()
            
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            
            out = model(data)
            loss = criterion(out, target)
            diff += loss.item()
            acc += (out.argmax(1) == target).sum().item()
            total += out.size(0)
            loss.backward()
            optimizer.step()
            
        train_loss += [diff/total]
        train_acc += [acc/total]
        
        model.eval()
        
        diff = 0
        acc = 0
        total = 0
        with torch.no_grad():
            for data, target in val_dl:

                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()

                out = model(data)
                loss = criterion(out, target)
                diff += loss.item()
                acc += (out.argmax(1) == target).sum().item()
                total += out.size(0)
        val_loss += [diff/total]
        val_acc += [acc/total]
        
        if val_acc[-1] >= best_acc:
            best_acc = val_acc[-1]
            best_model = deepcopy(model)
            
        print("Epoch {} train loss {} acc {} val loss {} acc {}".format(i, train_loss[-1], train_acc[-1],
                                                                    val_loss[-1], val_acc[-1]))
        scheduler.step()

    return model, best_model

def training(epochs, lr, step, gamma, batch, out_size=4):
    images, labels = generate_dataset()


    transform = transforms.Compose([transforms.ToPILImage(),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.5],
                                                   std=[0.5])])
    
    train_images, val_images, train_labels, val_labels = train_test_split(images, labels, random_state=42, test_size=0.2)

    EPOCHS = epochs
    LR = lr
    GAMMA = gamma
    STEP = step
    BATCH = batch
    OUT_SIZE = out_size

    train_ds = PreProcess(train_images, train_labels, transform)
    val_ds = PreProcess(val_images, val_labels, transform)

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training device used:")
    print(device)

    model = ResNet(ResBlock, 1, OUT_SIZE, 3, 1, 1, [2, 2, 3, 2])
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP, gamma=GAMMA)