

def model_train(config,net,criterion,optimizer,train_dataloader,val_dataloader):

    for i,batch in enumerate(train_dataloader):

        inputs = batch["image"]
        #task_labels = 
        outputs = net(inputs)

        #loss = criterion(outputs, task_labels)


