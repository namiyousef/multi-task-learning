import torch
import numpy as np
#from torchsummary import summary
from torchsummaryX import summary
from torchvision.utils import save_image
#arch = summary(model, torch.rand((1, 3, 256, 256)))

def model_train(config, net, criterion, optimizer, train_dataloader, val_dataloader):

    net.train()
    #print(summary(net, (3, 256, 256)))
    #arch = summary(net, torch.rand((1, 3, 256, 256)))
    loss_epoch_dict = {"Seg":[],"Class":[],"BB":[]}
    for i,mini_batch in enumerate(train_dataloader):
        
        #forward
        inputs = mini_batch["image"]
       # print(inputs.shape)
        inputs = inputs.permute([0,3, 2, 1])
       # print((inputs[0]/255).shape)
        save_image(inputs[0]/255, 'input.png')
        mini_batch["Segmen"] = mini_batch["Segmen"].permute([0, 3, 2, 1])
        mini_batch["RNL"] = mini_batch["RNL"].permute([0, 3, 2, 1])
        #mini_batch["Class"] = torch.reshape(mini_batch["Class"],(-1,)).type(torch.LongTensor)
        task_targets = {task:mini_batch[task]for task in config["Tasks"].keys()}
        optimizer.zero_grad()
        outputs = net(inputs)

        #loss
        loss = criterion(outputs,task_targets)
        loss['total'].backward()
        optimizer.step()
        print(f"Batch {i}")
        print(f"loss segmentation{loss['Segmen']}")
        print(f"loss Random task {loss['RNL']}")
        loss_epoch_dict["Seg"].append(loss['Segmen'].item())
        #loss_epoch_dict["Class"].append(loss['Class'].item())
        #loss_epoch_dict["BB"].append(loss['BB'].item())
    
    
    seg_mean = np.mean(np.array(loss_epoch_dict["Seg"]))
    class_mean = np.mean(np.array(loss_epoch_dict["Class"]))
    bb_mean = np.mean(np.array(loss_epoch_dict["BB"]))
    print ("Seg " + str(seg_mean) + " class mean " + str(class_mean) + " bb mean " + str(bb_mean))
    return net