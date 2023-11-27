import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader

def get_weights(model, layer_name):
    flag = False
    for name, layer in model.named_modules():
        if name == layer_name:
            flag = True
            weights = layer.weight.data.clone()
            break
    if not flag :
        print(f'Undefined Layer.')
        return -1
    return weights


def get_feature_map(activation, model, input, layer_name, idx):
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    model.eval()
    model.features[idx].register_forward_hook(get_activation(layer_name))
    output = model(input)
    act = activation[layer_name].squeeze()
    return act


def get_numerical_weight(model):
    means = []
    variances = []
    weights_data = []
    layer_index = []

    conv_layer_num = 0

    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            conv_layer_num += 1
            weights = layer.weight.data.cpu().numpy().flatten()
            means.append(weights.mean().item())
            variances.append(weights.var().item())
            weights_data.extend(weights)
            layer_index.extend([conv_layer_num] * len(weights))

    weight_df = pd.DataFrame({'Layer': layer_index, 'Weights': weights_data})

    return means, variances, weight_df


def get_feature_from_dataset(MODEL, batch_size, testloader, layer_name, idx):
    
    features = None
    labels = None
    preds = None
    shuffle_testloader = DataLoader(testloader.dataset, batch_size=batch_size, shuffle=True)
    
    with torch.no_grad():
        MODEL.eval()
        count = 0
        for data, target in shuffle_testloader:
            count += 1
            activation = {}
            feature = get_feature_map(activation, MODEL, data, layer_name, idx)
            features = feature.view(batch_size, -1)
            labels = target
            output = MODEL(data)
            preds = output.argmax(dim=1, keepdim=True).squeeze()
            if count != 0 :
                break
    
    return features, labels, preds