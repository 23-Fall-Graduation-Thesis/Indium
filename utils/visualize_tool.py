import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import utils
import numpy as np
import torch
from utils.get_data import get_all_feature_maps, get_numerical_weight

def visTensor(tensor, layer_name, ncols=32 , showAll=False):
    # only use FIRST CHANNEL #TODO 
    n, c, w, h = tensor.shape
    nrows = n // ncols + (1 if n % ncols else 0)
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols, nrows))
    for i in range(n):
        ax = axes[i // ncols, i % ncols]
        kernel = tensor[i, 0, :, :] if n > 1 else tensor[i, :, :]
        ax.imshow(kernel, cmap='viridis')
        ax.axis('off')
    plt.title('Layer', layer_name)
    plt.show()


def filters_visualize(model, layer_name, ncols=32, showAll=False):
    flag = False
    for name, layer in model.named_modules():
        if name == layer_name:
            flag = True
            weights = layer.weight.data.clone()
            break
    if not flag :
        print(f'Undefined Layer.')
        return
    
    print(weights.shape)
    visTensor(weights, ncols, showAll)

    
def feature_map_visualize(model, image, cols=32):
    feature_maps = get_all_feature_maps(model, image)
    num_layers = len(feature_maps)    
    rows = num_layers // cols + (1 if num_layers % cols else 0)
    plt.figure(figsize=(20, rows * 3))
    for i, feature_map in enumerate(feature_maps):
        plt.subplot(rows, cols, i+1)
        plt.imshow(feature_map[0, 0].cpu().detach(), cmap='viridis')
        plt.title(f'Layer {i+1}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def weight_distribution_visualize(model):
    means, variances, weight_df = get_numerical_weight(model)
    
    plt.figure(figsize=(10, 5))
    plt.plot(means, label='Mean')
    plt.plot(variances, label='Variance')
    plt.xlabel('Kernel Index')
    plt.ylabel('Value')
    plt.title('Kernel Weights Statistics (Mean and Variance)')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.violinplot(x='Layer', y='Weights', data=weight_df)
    plt.xlabel('Convolution Layer Number')
    plt.ylabel('Weights')
    plt.title('Weight Distribution in Convolutional Layers')
    plt.show()