import torch
import numpy as np
from data.data import OxfordPetDataset, _get_data
from utils import _prepare_data , _update_loss_dict

# if GPU avail, set device to GPU (NOTE: this will only use a single GPU. For multi GPU may need changes)
CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    device = torch.device('cuda:0')
    print('CUDA device detected. Running on GPU.')
else:
    device = torch.device('cpu')
    print('CUDA device not detected. Running on CPU instead.')

def model_train(config,net,criterion,optimizer,mini_batch_size,train_dataloader,val_dataloader):

    net.to(device)
    net.train()

    # TODO should generate this dynamically
    # TODO also, saving is only useful for plotting purposes. We should actuall implement this, so add some code to export this data
    # TODO perhaps the keras API .history() might be a nice API style to use??
    loss_epoch_dict = {"Seg":[],"Class":[],"BB":[]}
    
   
    
    for i,mini_batch in enumerate(train_dataloader):
        
        #forward
        #inputs = mini_batch["image"]
        #inputs = inputs.permute([0,3, 2, 1])
        #mini_batch["Segmen"] = mini_batch["Segmen"].permute([0,3, 2, 1])
        #mini_batch["Class"] = torch.reshape(mini_batch["Class"],(-1,)).type(torch.LongTensor)

        mini_batch = _prepare_data(mini_batch,config)
        inputs = mini_batch["image"]
        inputs = inputs.to(device)

        task_targets = {task:mini_batch[task].to(device) for task in config["Tasks"].keys()}
        optimizer.zero_grad()
        outputs = net(inputs)
        
        loss = criterion(outputs,task_targets)
        loss['total'].backward()
        optimizer.step()


        print(loss['Class'].item())
        
        loss_epoch_dict = _update_loss_dict(loss_epoch_dict,loss, config)

    
    
    #seg_mean = np.mean(np.array(loss_epoch_dict["Seg"]))
    class_mean = np.mean(np.array(loss_epoch_dict["Class"]))
    #bb_mean = np.mean(np.array(loss_epoch_dict["BB"]))
    #print ("seg mean " + str(seg_mean) + " class mean " + str(class_mean) + " bb mean " + str(bb_mean))
    print (" class mean " + str(class_mean) )  # for testing purposes only. # TODO need to add verbose options!
    # TODO also, I would like an option for plotting alt metrics as well, e.g. accuracy metrics.
    # TODO for classification can use accuracy, for segmentation can perhaps pixel accuracy or Jaccard or F1 score??
    return net
        


