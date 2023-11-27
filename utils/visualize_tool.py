import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import os
from utils.visualize_grad import VanillaBackprop
from utils.get_data import *
from utils.util_functions import *


def visualize_tensor(tensor, isnomalized=True):
    numpy = tensor.numpy().transpose(1, 2, 0)
    if isnomalized :
        numpy = (numpy - numpy.min()) / (numpy.max() - numpy.min())
    plt.imshow(numpy)
    plt.show()


def plot_filters(model_info, tensor, layer_name, ncols=32 , nchannel=5):
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


def visualize_filters(model_info, model, layer_name, ncols=32, nchannel=5, showAll=False):
    weights = get_weights(model, layer_name)
    print(weights.shape)
    n, c, w, h = weights.shape
    if not showAll:
        c = min(c, nchannel)
    plot_filters(model_info, weights, layer_name, ncols, c)



def visualize_feature_map(activation, model, input, layer_name, idx, ncols=32):
    act = get_feature_map(activation, model, input, layer_name, idx)
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


def visualize_class_activation_images(org_img, activation_map, dataset, freeze, layer_num, show=True, save=False):
    """
        Saves cam activation map and activation map on the original image

    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    if not os.path.exists('./results/LayerCAM'):
        os.makedirs('./results/LayerCAM')
        
    # Grayscale activation map
    heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map, 'hsv')
    
    if show:
        images = [heatmap, heatmap_on_image, activation_map]
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
        for ax, img in zip(axes, images):
            ax.imshow(img)
            ax.axis('off') 
        plt.show()
    
    if save :
        path_to_file = os.path.join('./results/LayerCAM', dataset+'_'+freeze+'_#'+str(layer_num)+'_Heatmap.png')
        save_image(heatmap, path_to_file)
        path_to_file = os.path.join('./results/LayerCAM', dataset+'_'+freeze+'_#'+str(layer_num)+'_On_Image.png')
        save_image(heatmap_on_image, path_to_file)
        path_to_file = os.path.join('./results/LayerCAM', dataset+'_'+freeze+'_#'+str(layer_num)+'_Grayscale.png')
        save_image(activation_map, path_to_file)


def visualize_gradXimage(prep_img, target_class,  model, dataset, freeze, show=True, save=False):
    VBP = VanillaBackprop(model)
    vanilla_grads = VBP.generate_gradients(prep_img, target_class)

    grad_times_image = vanilla_grads * prep_img.detach().numpy()[0]
    grayscale_vanilla_grads = convert_to_grayscale(grad_times_image)
    save_gradient_images(grayscale_vanilla_grads, dataset, freeze)
    print('Grad times image completed.')