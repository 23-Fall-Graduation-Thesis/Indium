
import matplotlib.pyplot as plt
from torchvision import utils
import numpy as np

"""def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1): 
    n,c,w,h = tensor.shape
    if allkernels: tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))    
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure( figsize=(nrow,rows) )
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    return"""

def visTensor(tensor, ncols=8 , showAll=False):
    # only use FIRST CHANNEL #TODO 
    n, c, w, h = tensor.shape
    nrows = n // ncols + (1 if n % ncols else 0)
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols, nrows))
    for i in range(n):
        ax = axes[i // ncols, i % ncols]
        kernel = tensor[i, 0, :, :] if n > 1 else tensor[i, :, :]
        ax.imshow(kernel, cmap='viridis')
        ax.axis('off')
    plt.show()

def filters_visualize(model, layer_name, nrows=1, ncols=6):
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
    visTensor(weights)

    
def activation_visualize(model):
    ""