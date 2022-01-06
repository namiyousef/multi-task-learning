import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.model import Model, ResUnet
from criterion.criterion import Criterion
from data.data import get_dataloader, get_dataset
from train import model_train

# Base network
BASIC_CONFIG = {"Model":"Single Task","Tasks":{"Segmen":1}}
# Full MTL
#MLT_CONFIG = {"Model":"Multi Task", "Tasks":{"Segmen":1, "Class":2, "BB":4} }
# MTL with the random layer
MLT_CONFIG = {"Model":"Multi Task", "Tasks":{"Segmen":1, "RNL":1} }

MINI_BATCH_SIZE = 32
NUM_EPOCH = 5
test_mini_batch = 32

CONFIG = MLT_CONFIG 

# get model 
net = Model(CONFIG)
#net.double()

# get losses 
criterion = Criterion(CONFIG, "train")
test_criterion = Criterion(CONFIG, "test")

# get optimizer
optimizer = torch.optim.Adam(net.parameters(),lr=0.006)

# transform and get data
train_dataset = get_dataset(CONFIG,"train")
val_dataset = get_dataset(CONFIG,"val")
test_dataset = get_dataset(CONFIG, "test")
train_dataloader = get_dataloader(train_dataset,MINI_BATCH_SIZE)
val_dataloader = get_dataloader(val_dataset,MINI_BATCH_SIZE)
test_dataloader = get_dataloader(test_dataset, test_mini_batch)

# train loop
for epoch in range(NUM_EPOCH):

    print ("epoch " + str(epoch))
    model = model_train(CONFIG,net,criterion,optimizer,train_dataloader,val_dataloader)

print("\nNow testing\n")
# test loop

model.eval()
#net.train()
loss_epoch_dict = {"Seg":[],"Class":[],"BB":[]}
#running_loss = 0.0
with torch.no_grad():
    for i,mini_batch in enumerate(test_dataloader):
            inputs = mini_batch["image"]
            inputs = inputs.permute([0, 3, 2, 1])
            mini_batch["Segmen"] = mini_batch["Segmen"].permute([0, 3, 2, 1])
            #mini_batch["Class"] = torch.reshape(mini_batch["Class"],(-1,)).type(torch.LongTensor)
            task_targets = {task:mini_batch[task].detach() for task in CONFIG["Tasks"].keys()}
            outputs = model(inputs)
            loss = test_criterion(outputs,task_targets)
            print(i)
            print(loss)
            #running_loss += loss["Segmen"].item()
            #print(f"running_loss: {running_loss}")
            loss_epoch_dict["Seg"].append(loss['Segmen'].detach().item())
            
            #loss_epoch_dict["Class"].append(loss['Class'].item())
            #loss_epoch_dict["BB"].append(loss['BB'].item())

seg_mean = np.mean(np.array(loss_epoch_dict["Seg"]))
print(f"Segmentation test loss: {seg_mean}")