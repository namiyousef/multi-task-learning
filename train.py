import torch



def model_train(config,net,criterion,optimizer,train_dataloader,val_dataloader):

    for i,batch in enumerate(train_dataloader):

        inputs = batch["image"]
        inputs = inputs.permute([0,3, 2, 1])
        #task_labels = 
        outputs = net(inputs)
        print("got output")

        #loss = criterion(outputs, task_labels)


