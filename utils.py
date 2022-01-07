import torch

def _prepare_data(data,config):

    data["image"] = data["image"].permute([0,3, 2, 1])
    
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