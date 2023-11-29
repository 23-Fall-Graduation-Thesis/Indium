import torch

import torch.optim as optim
from dataset_dir.datasets import datasetload
import torch.nn as nn
import copy
import torchvision.models as models

def custom_finetuning(model, layers=None):
    '''
    model : pretrained model
    layers : list of layers to update, containing the index of the layers
    '''

    # layer names for current model
    c_layers_name = []
    for name, param in model.named_parameters():
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
        layers = str.strip() # binary

    if len(layers)!=len(c_layers_name):
        raise ValueError("Number of layers entered is different from the number of layers in the model")

    # check layer names & freeze input layers
    for idx, (name, param) in enumerate(model.named_parameters()):
        if layers[int(idx//2)] == "1":
            param.requires_grad = True
        else:
            param.requires_grad = False
        
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
    layer_lrs = str.strip()
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

# test code
def select_test():
    # test code 
    weights = models.AlexNet_Weights.IMAGENET1K_V1
    origin_model = models.alexnet(weights=weights).to('cuda')
    update_model = custom_finetuning(copy.deepcopy(origin_model)).to('cuda')

    for name, param in update_model.named_parameters():
        print(f"name: {name}\t grad: {param.requires_grad}")

    trainloader, _, _ = datasetload('cifar10')
    criterion = nn.CrossEntropyLoss().to('cuda')
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, update_model.parameters()), lr=0.1)

    # _, _ = train(update_model, criterion, optimizer, trainloader, 'cuda')

    for (name1, param1), (_, param2) in zip(origin_model.named_parameters(), update_model.named_parameters()):
        print(f"name: {name1}\tupdate: {not torch.equal(param1, param2)}")

if __name__ == "__main__":
    select_test()