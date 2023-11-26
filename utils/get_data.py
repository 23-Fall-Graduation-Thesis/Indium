import torch
import torch.nn as nn
import pandas as pd

def get_all_feature_maps(model, image):
    feature_maps = []
    hooks = []

    def hook_fn(module, input, output):
        if isinstance(module, torch.nn.Conv2d):
            feature_maps.append(output)

    for layer in model.children():
        if isinstance(layer, torch.nn.Conv2d):
            hook = layer.register_forward_hook(hook_fn)
            hooks.append(hook)

    with torch.no_grad():
        model(image)

    for hook in hooks:
        hook.remove()

    return feature_maps

def get_feature_map(model, image):
    with torch.no_grad(): 
        feature_map = model(image)
    return feature_map


def get_numerical_weight(model):
    means = []
    variances = []
    weights_data = []
    layer_index = []

    conv_layer_num = 0
    for layer in model.children():
        if isinstance(layer, nn.Conv2d):
            conv_layer_num += 1
            weights = layer.weight.data.cpu().numpy().flatten()
            means.append(weights.mean().item())
            variances.append(weights.var().item())
            weights_data.extend(weights)
            layer_index.extend([conv_layer_num] * len(weights))

    weight_df = pd.DataFrame({'Layer': layer_index, 'Weights': weights_data})

    return means, variances, weight_df

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

#
#
#

def hook_fn(module, input, output):
    module.output = output

def addHook(model, hook):
    for layer in model.children():
        if len(list(layer.children())) > 0: # if module has children -> sequential
            addHook(layer, hook)

        else:
            layer.register_forward_hook(hook)

def collectOutput(model, output_list):
    for layer in model.children():
        if len(list(layer.children())) > 0: # if module has children -> sequential
            collectOutput(layer, output_list)
        
        elif type(layer) == nn.ParameterList:
            return

        else:
            output = layer.output
            output_list.append(output)

def getName(model, name_list):
    for layer in model.children():
        if len(list(layer.children())) > 0: # if module has children -> sequential
            getName(layer, name_list)

        else:
            layer_name = str(layer)[:str(layer).index("(")] + str(len(list(filter(lambda x:str(layer)[:str(layer).index("(")] in x, name_list))))
            name_list.append(layer_name)