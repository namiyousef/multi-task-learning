import torch



def model_train(config,net,criterion,optimizer,train_dataloader,val_dataloader):

    for i,batch in enumerate(train_dataloader):
        
        #forward
        #batch = batch
        inputs = batch["image"]
        inputs = inputs.permute([0,3, 2, 1])
        batch["Segmen"] = batch["Segmen"].permute([0,3, 2, 1])
        task_targets = {task:batch[task]for task in config["Tasks"].keys()}
        outputs = net(inputs)

        #loss
        loss = criterion(outputs,task_targets)
        print("check")
        #backward



        print("got output")

        #loss = criterion(outputs, task_labels)


