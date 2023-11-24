import torch

 # for test code
from model.models import Conv4
import torch.optim as optim
from dataset_dir.datasets import datasetload
import torch.nn as nn
import copy

def custom_finetuning(model, layers=None):
    '''
    model : pretrained model
    layers : list of layers to freeze, containing the names of the layers
    '''

    # layer names for current model
    c_layers_name = []
    for name, param in model.named_parameters():
        param.requires_grad = False
        if name[-6:] == 'weight':
            name = name[:-7]
        elif name[-4:] == 'bias':
            name = name[:-5]
        if name not in c_layers_name:
            c_layers_name.append(name)
    
    # to enter layers
    if layers is None:
        print("Layers in current model\n")
        for layer in c_layers_name:
            print(layer, end="  ")
        str = input("\n\nselect layers to freeze : ")
        layers = str.strip().split(" ")

    # check layer names & freeze input layers
    for layer in layers:
        if layer in c_layers_name:
            for name, param in model.named_parameters():
                if layer in name:
                    param.requires_grad = True
        else:
            raise ValueError(f"\nWrong name of layer : {layer}.")
        
    return model

def optimal_finetuning(model):
    '''
    model : pretraied model
    '''

    '''
    optimal fine tuning layer select and freeze TODO
    '''

    return model

def control_lr_finetuning(model, layers_lr=None):
    '''
    model : pretrained model
    layers_lr : learning rate per layer
    '''

    # layer names for current model
    c_layers_name = []
    for name, param in model.named_parameters():
        print(param)
        if name[-6:] == 'weight':
            name = name[:-7]
        elif name[-4:] == 'bias':
            name = name[:-5]
        if name not in c_layers_name:
            c_layers_name.append(name)
    
    # to enter lr
    print("Layers in current model\n")
    for layer in c_layers_name:
        print(layer, end="  ")
    str = input("\n\nset learning rate for each layer : ")
    layer_lrs = str.strip().split(" ")
    if len(layer_lrs)!= len(c_layers_name):
        raise ValueError('Wrong number of learning rates')

    # check layer names & set learning rate
    lr_list = []
    for lr, layer in zip(layer_lrs, c_layers_name):
        lr = float(lr)
        if lr<0:
            raise ValueError("learning rate cannot be negative")
        for name, param in model.named_parameters():
            if layer in name:
                lr_list.append({'params':param, 'lr': lr})
        lr_list.append({'params', param})

    return lr_list

def select_test():
    # test code 
    origin_model = Conv4().cuda()
    update_model = custom_finetuning(copy.deepcopy(origin_model))

    for name, param in update_model.named_parameters():
        print(f"name: {name}\t grad: {param.requires_grad}")

    trainloader, validloader, testloader = datasetload('cifar10')
    criterion = nn.CrossEntropyLoss().to('cuda')
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, update_model.parameters()), lr=0.1)

    train_loss, train_acc = train(update_model, criterion, optimizer, trainloader, 'cuda')

    for (name1, param1), (name2, param2) in zip(origin_model.named_parameters(), update_model.named_parameters()):
        print(f"name: {name1}\tupdate: {not torch.equal(param1, param2)}")

def lr_test():
    origin_model = Conv4().cuda()
    update_model = copy.deepcopy(origin_model)
    lr_list = control_lr_finetuning(update_model)

    trainloader, validloader, testloader = datasetload('cifar10')
    criterion = nn.CrossEntropyLoss().to('cuda')
    optimizer = optim.SGD(lr_list, momentum=0.9, weight_decay=5e-4)

    train_loss, train_acc = train(update_model, criterion, optimizer, trainloader, 'cuda')

    for (name1, param1), (name2, param2) in zip(origin_model.named_parameters(), update_model.named_parameters()):
        print(f"name: {name1}\tupdate: {not torch.equal(param1, param2)}")

def train(model, criterion, optimizer, trainloader, device):
    model.train()
    
    train_loss = 0
    loss = 0
    train_acc = 0
    for data, target in trainloader:
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        train_acc += pred.eq(target.view_as(pred)).sum().item()
        loss.backward()
        optimizer.step()
    
    train_loss = train_loss / len(trainloader.dataset)
    train_acc = train_acc / len(trainloader.dataset) * 100
    
    return train_loss, train_acc


if __name__ == '__main__':
    select_test()
    # lr_test()
    
