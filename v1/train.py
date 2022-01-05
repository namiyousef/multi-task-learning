import torch
import numpy as np
#from torchsummary import summary
from torchsummaryX import summary
#arch = summary(model, torch.rand((1, 3, 256, 256)))

def model_train(config, net, criterion, optimizer, train_dataloader, val_dataloader):

    net.train()
    #print(summary(net, (3, 256, 256)))
    #arch = summary(net, torch.rand((1, 3, 256, 256)))
    loss_epoch_dict = {"Seg":[],"Class":[],"BB":[]}
    for i,mini_batch in enumerate(train_dataloader):
        
        #forward
        inputs = mini_batch["image"]
        inputs = inputs.permute([0,3, 2, 1])
        mini_batch["Segmen"] = mini_batch["Segmen"].permute([0,3, 2, 1])
        #mini_batch["Class"] = torch.reshape(mini_batch["Class"],(-1,)).type(torch.LongTensor)
        task_targets = {task:mini_batch[task]for task in config["Tasks"].keys()}
        optimizer.zero_grad()
        outputs = net(inputs)

        #loss
        loss = criterion(outputs,task_targets)
        loss['total'].backward()
        optimizer.step()
        print(i)
        print(loss["Segmen"])

        loss_epoch_dict["Seg"].append(loss['Segmen'].item())
        #loss_epoch_dict["Class"].append(loss['Class'].item())
        #loss_epoch_dict["BB"].append(loss['BB'].item())
    
    
    seg_mean = np.mean(np.array(loss_epoch_dict["Seg"]))
    class_mean = np.mean(np.array(loss_epoch_dict["Class"]))
    bb_mean = np.mean(np.array(loss_epoch_dict["BB"]))
    print ("seg mean " + str(seg_mean) + " class mean " + str(class_mean) + " bb mean " + str(bb_mean))
    return net