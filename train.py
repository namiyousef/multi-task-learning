import torch
import numpy as np


def model_train(config,net,criterion,optimizer,train_dataloader,val_dataloader):

    net.train()
    loss_epoch_dict = {"Seg":[],"Class":[],"BB":[]}
    for i,mini_batch in enumerate(train_dataloader):
        
        #forward
        inputs = mini_batch["image"]
        inputs = inputs.permute([0,3, 2, 1])
        mini_batch["Segmen"] = mini_batch["Segmen"].permute([0,3, 2, 1])
        mini_batch["Class"] = torch.reshape(mini_batch["Class"],(-1,)).type(torch.LongTensor)
        task_targets = {task:mini_batch[task]for task in config["Tasks"].keys()}
        optimizer.zero_grad()
        outputs = net(inputs)

        #loss
        loss = criterion(outputs,task_targets)
        loss['total'].backward()
        optimizer.step()
        #print("check")
        #print(loss)
        #backward
        #if i % 10 == 0:
            #print('[ %5d] loss:' %
                    #( i + 1))
            #print(loss)

        loss_epoch_dict["Seg"].append(loss['Segmen'])
        loss_epoch_dict["Class"].append(loss['Class'])
        loss_epoch_dict["BB"].append(loss['BB'])
    
    
    seg_mean = np.mean(np.array(loss_epoch_dict["Seg"]))
    class_mean = np.mean(np.array(loss_epoch_dict["Class"]))
    bb_mean = np.mean(np.array(loss_epoch_dict["BB"]))
    print ("seg mean " + str(seg_mean) + " class mean " + str(class_mean) + " bb mean " + str(bb_mean))
        #print("got output")

        #loss = criterion(outputs, task_labels)


