import torch
from utils import _prepare_data, _update_loss_dict
from models.model import Model
from criterion.criterion import Criterion
from data.data import get_dataloader, get_dataset
from train import model_train



def main(config, epochs=3, batch_size=32):
    model_config = config['model'] # TODO currently only supports RESU. Possibly generalise? (low priority)
    task_config = config['mtl']
    net = Model(task_config, model_config)
    criterion = Criterion(task_config)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-04)

    train_dataset = get_dataset(task_config, "train")
    val_dataset = get_dataset(task_config, "val")
    test_dataset = get_dataset(task_config, "test")

    train_dataloader = get_dataloader(train_dataset, batch_size)
    val_dataloader = get_dataloader(val_dataset, batch_size)
    test_dataloader = get_dataloader(test_dataset, batch_size)

    # train loop

    print("Train loop started...")

    for i, epoch in enumerate(range(epochs)):
        print(f"Epoch {i+1}") # TODO beautify this with verbose later
        model_eval = model_train(task_config, net, criterion, optimizer, batch_size, train_dataloader, val_dataloader)

    print("Test loop started...")
    model_eval.eval()
    loss_epoch_dict = {"Seg": [], "Class": [], "BB": []}

    # hard coded image accuracy, need to generalise later!
    accuracy = 0

    with torch.no_grad():
        for i, mini_batch in enumerate(test_dataloader):
            # inputs = mini_batch["image"]
            # inputs = inputs.permute([0,3, 2, 1])
            # mini_batch["Segmen"] = mini_batch["Segmen"].permute([0,3, 2, 1])
            # mini_batch["Class"] = torch.reshape(mini_batch["Class"],(-1,)).type(torch.LongTensor)
            mini_batch = _prepare_data(mini_batch, task_config)
            inputs = mini_batch["image"]

            task_targets = {task: mini_batch[task] for task in task_config["Tasks"].keys()}
            test_output = model_eval(inputs)

            test_class = torch.argmax(test_output, dim=1)
            accuracy += (test_class == test_output).sum()/len(mini_batch)
            loss = criterion(test_output, task_targets)

            loss_epoch_dict = _update_loss_dict(loss_epoch_dict, loss, task_config)

    for loss in loss_epoch_dict:
        print(f'Loss for {loss}: {sum(loss_epoch_dict[loss])/len(sum(loss_epoch_dict[loss]))}')
    print(f'Test accuracy is: {accuracy}')
            # loss_epoch_dict["Seg"].append(loss['Segmen'].item())
            # loss_epoch_dict["Class"].append(loss['Class'].item())
            # loss_epoch_dict["BB"].append(loss['BB'].item())

    # seg_mean = np.mean(np.array(loss_epoch_dict["Seg"]))
    # class_mean = np.mean(np.array(loss_epoch_dict["Class"]))
    # bb_mean = np.mean(np.array(loss_epoch_dict["BB"]))
    # print ("seg mean " + str(seg_mean) + " class mean " + str(class_mean) + " bb mean " + str(bb_mean))
    # print ( " class mean " + str(class_mean))

### VAR ######
"""
#BASIC_CONFIG = {"Model":"Single Task","Tasks":{"Segmen":1}}
#MLT_CONFIG = {"Model":"Multi Task", "Tasks":{ "Class":2, "BB":2} }
MLT_CONFIG = {"Model":"Multi Task","Tasks":{"Class":2},"Loss Lambda":{ "Class":1} }
#MLT_CONFIG = {
# "Model":"Multi Task", "Tasks":{"Segmen":1, "Class":2, "BB":4}, "Loss Lambda":{
# "Segmen":1, "Class":1/100, "BB":1/100000
# }
# } # TODO this causes parametrisation issues... needs fixing!
#MLT_CONFIG = {"Model":"Multi Task", "Tasks":{"Segmen":1, "Class":2} }

MINI_BATCH_SIZE = 32
NUM_EPOCH = 3
RESU_FILTS = [32,32,64,128] # TODO why this choice?


CONFIG = MLT_CONFIG

    # get model 
net = Model(CONFIG,RESU_FILTS)


    # get losses 
criterion = Criterion(CONFIG)

    # get optimizer
optimizer = torch.optim.Adam(net.parameters(),lr=1e-04)

    # transform and get data
train_dataset = get_dataset(CONFIG,"train")
val_dataset = get_dataset(CONFIG,"val")
test_dataset = get_dataset(CONFIG,"test")

train_dataloader = get_dataloader(train_dataset,MINI_BATCH_SIZE)
val_dataloader = get_dataloader(val_dataset,MINI_BATCH_SIZE)
test_dataloader = get_dataloader(test_dataset,MINI_BATCH_SIZE)

    # train loop

print("train")

for epoch in range(NUM_EPOCH):

    print ("Epoch " + str(epoch))
    model_eval = model_train(CONFIG,net,criterion,optimizer,MINI_BATCH_SIZE,train_dataloader,val_dataloader)


print("Now test")
model_eval.eval()
loss_epoch_dict = {"Seg":[],"Class":[],"BB":[]}

with torch.no_grad():
    for i,mini_batch in enumerate(test_dataloader):

        #inputs = mini_batch["image"]
        #inputs = inputs.permute([0,3, 2, 1])
        #mini_batch["Segmen"] = mini_batch["Segmen"].permute([0,3, 2, 1])
        #mini_batch["Class"] = torch.reshape(mini_batch["Class"],(-1,)).type(torch.LongTensor)
        mini_batch = _prepare_data(mini_batch,CONFIG)
        inputs = mini_batch["image"]

        task_targets = {task:mini_batch[task]for task in CONFIG["Tasks"].keys()}
        test_output = model_eval(inputs)
        loss = criterion(test_output,task_targets)

        loss_epoch_dict = _update_loss_dict(loss_epoch_dict,loss, CONFIG)

        #loss_epoch_dict["Seg"].append(loss['Segmen'].item())
        #loss_epoch_dict["Class"].append(loss['Class'].item())
        #loss_epoch_dict["BB"].append(loss['BB'].item())
    
#seg_mean = np.mean(np.array(loss_epoch_dict["Seg"]))
#class_mean = np.mean(np.array(loss_epoch_dict["Class"]))
#bb_mean = np.mean(np.array(loss_epoch_dict["BB"]))
#print ("seg mean " + str(seg_mean) + " class mean " + str(class_mean) + " bb mean " + str(bb_mean))
#print ( " class mean " + str(class_mean))"""

if __name__ == '__main__':
    config = {
        'model': [32, 32, 64, 128],
        'mtl': {"Model":"Multi Task","Tasks":{"Class":2},"Loss Lambda":{ "Class":1}}
    }

    main(config=config, epochs=3, batch_size=32)