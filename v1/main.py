import torch

import torch.nn as nn
import torch.nn.functional as F
from models.model import Model
from criterion.criterion import Criterion
from data.data import get_dataloader, get_dataset
from train import model_train


# tst

### VAR ######
NUM_EPOCH = 10
BASIC_CONFIG = {"Model":"Single Task","Tasks":{"Segmen":1}}
MLT_CONFIG = {"Model":"Multi Task", "Tasks":{"Segmen":1, "Class":2, "BB":2} }
MINI_BATCH_SIZE = 32
NUM_EPOCH = 10

""

CONFIG = MLT_CONFIG 

def evaluate():

    # get model 
    net = Model(CONFIG)

    # get losses 
    criterion = Criterion(CONFIG)

    # get optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, momentum=0.9)

    # transform and get data
    train_dataset = get_dataset(CONFIG,"train")
    val_dataset = get_dataset(CONFIG,"val")
    train_dataloader = get_dataloader(train_dataset,MINI_BATCH_SIZE)
    val_dataloader = get_dataloader(val_dataset,MINI_BATCH_SIZE)

    # train loop

    for epoch in range(NUM_EPOCH):

        model_eval = model_train(CONFIG,net,criterion,optimizer,train_dataloader,val_dataloader)

