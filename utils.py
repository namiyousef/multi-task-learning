from math import nan
import torch
from torchvision import transforms
import numpy as np




def _prepare_data(data,config):

    transform = transforms.Compose(
        [
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    data["image"] = transform(data["image"].permute([0,3, 2, 1]))
    
    if "Segmen" in config["Tasks"].keys():
        data["Segmen"] = data["Segmen"].permute([0,3, 2, 1])

    if "Class" in config["Tasks"].keys():
        data["Class"] = torch.reshape(data["Class"],(-1,)).type(torch.LongTensor)

    return data

def _update_loss_dict(dict,loss,config):

    if "Segmen" in config["Tasks"].keys():
        dict["Seg"].append(loss['Segmen'].item())
    if "Class" in config["Tasks"].keys():
        dict["Class"].append(loss['Class'].item())
    if "BB" in config["Tasks"].keys():
        dict["BB"].append(loss['BB'].item())

    return dict

def _print_epoch_results(dict,config):

    seg_mean = nan
    class_mean = nan
    bb_mean = nan

    if "Segmen" in config["Tasks"].keys():
        seg_mean = np.mean(np.array(dict["Seg"]))
    if "Class" in config["Tasks"].keys():
        class_mean = np.mean(np.array(dict["Class"]))
    if "BB" in config["Tasks"].keys():
        bb_mean = np.mean(np.array(dict["BB"]))

    print ("seg mean " + str(seg_mean) + " class mean " + str(class_mean) + " bb mean " + str(bb_mean))

def _update_performance_dict(dict,loss,output,batch,config):

    if "Segmen" in config["Tasks"].keys():
        dict["Seg"].append(loss['Segmen'].item())
    if "Class" in config["Tasks"].keys():
        #test_class = torch.argmax(output["Class"], dim=1)
        #accuracy = (test_class == batch["Class"]).sum()/len(32)
        #dict["Class"].append(accuracy)
        dict["Seg"].append(loss['Class'].item())
    if "BB" in config["Tasks"].keys():
        dict["BB"].append(loss['BB'].item())

    return dict