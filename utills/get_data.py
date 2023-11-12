import torch
import torch.nn as nn
import pandas as pd

def get_feature_maps(model, image):
    layers = []
    feature_maps = []

    def hook_fn(module, input, output):
        if isinstance(module, nn.Conv2d):
            feature_maps.append(output)

    for layer in model.children():
        if isinstance(layer, nn.Conv2d):
            layers.append(layer)
            layer.register_forward_hook(hook_fn)

    return layers, feature_maps

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