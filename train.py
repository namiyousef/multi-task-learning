import torch



def model_train(config,net,criterion,optimizer,train_dataloader,val_dataloader):

    net.train()

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
        if i % 10 == 0:
            print('[ %5d] loss:' %
                    ( i + 1))
            print(loss)


        #print("got output")

        #loss = criterion(outputs, task_labels)


