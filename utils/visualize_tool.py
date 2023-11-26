import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import utils
import numpy as np
import torch
from utils.get_data import *
from utils.util_functions import *

def visualize_tensor(tensor, isnomalized=True):
    numpy = tensor.numpy().transpose(1, 2, 0)
    if isnomalized :
        numpy = (numpy - numpy.min()) / (numpy.max() - numpy.min())
    plt.imshow(numpy)
    plt.show()


def plot_filters(tensor, layer_name, ncols=32 , nchannel=5):
    n, _, _, _ = tensor.shape
    nrows = n // ncols + (1 if n % ncols else 0)

    for c in range(nchannel):
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols, nrows))
        for i in range(n):
            ax = axes[i // ncols, i % ncols]
            kernel = tensor[i, c, :, :] if n > 1 else tensor[i, :, :]
            ax.imshow(kernel, cmap='viridis')
            ax.axis('off')
        alias = get_alias(layer_name, model_info)
        plt.suptitle('Layer: '+alias+" (# channel: "+str(c)+")")
        plt.show()


def visualize_filters(model, layer_name, ncols=32, nchannel=5, showAll=False):
    weights = get_weights(model, layer_name)
    print(weights.shape)
    n, c, w, h = weights.shape
    if not showAll:
        c = min(c, nchannel)
    plot_filters(weights, layer_name, ncols, c)



def visualize_feature_map(model, input, layer_name, idx, ncols=32):
    act = get_feature_map(model, input, layer_name, idx)
    print(act.shape)
    nrows = max(act.size(0) // ncols, 1) 
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols, nrows))
    for i in range(act.size(0)):
        ax = axes[i // ncols, i % ncols]
        kernel = act[i]
        ax.imshow(kernel, cmap='viridis')
        ax.axis('off')
    plt.suptitle('Layer: '+layer_name)
    plt.show()


def visualize_weight_distribution(model, violin_sample=1000):
    sns.set()
    means, variances, weight_df = get_numerical_weight(model)
    
    plt.figure(figsize=(7, 4))
    plt.plot(means, label='Mean')
    plt.plot(variances, label='Variance')
    plt.xlabel('conv layer idx')
    plt.ylabel('value')
    plt.title('Weights Statistics')
    plt.legend()
    plt.show()


    plt.figure(figsize=(7, 4))
    sampled_df = weight_df.groupby('Layer').apply(lambda x: x.sample(violin_sample)).reset_index(drop=True)
    plt.xlabel('conv #')
    plt.ylabel('weights')
    plt.xticks([1, 2, 3, 4, 5], ['1', '2', '3', '4', '5'])
    sns.violinplot(x='Layer', y='Weights', data=sampled_df)
    plt.show()
