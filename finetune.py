import torch

 # for test code
from model.models import Conv4
import torch.optim as optim
from main import train
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
    for name, _ in model.named_parameters():
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
                    param.requires_grad = False
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


if __name__ == '__main__':
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
