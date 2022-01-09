import torch
import numpy as np
from data.data import OxfordPetDataset, _get_data
from utils import _prepare_data , _update_loss_dict

def model_train(config,net,criterion,optimizer,mini_batch_size,train_dataloader,val_dataloader):

     
    net.train()
    loss_epoch_dict = {"Seg":[],"Class":[],"BB":[]}
    
   
    for i,mini_batch in enumerate(train_dataloader):
        
        mini_batch = _prepare_data(mini_batch,config)
        inputs = mini_batch["image"]

        task_targets = {task:mini_batch[task]for task in config["Tasks"].keys()}
        optimizer.zero_grad()
        outputs = net(inputs)
        
        loss = criterion(outputs,task_targets)
        loss['total'].backward()
        optimizer.step()
        
        #backward
        print(loss['Class'].item())
        
        loss_epoch_dict = _update_loss_dict(loss_epoch_dict,loss, config)

    
    
    #seg_mean = np.mean(np.array(loss_epoch_dict["Seg"]))
    class_mean = np.mean(np.array(loss_epoch_dict["Class"]))
    
    # bb_mean = np.mean(np.array(loss_epoch_dict["BB"]))
    #print ("seg mean " + str(seg_mean) + " class mean " + str(class_mean) + " bb mean " + str(bb_mean))
    print (" class mean " + str(class_mean) )
    return net
        


