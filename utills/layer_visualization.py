
import matplotlib.pyplot as plt
from torchvision import utils
import numpy as np

def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1): 
    n,c,w,h = tensor.shape
    if allkernels: tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))    
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure( figsize=(nrow,rows) )
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    return

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
    
    visTensor(weights)
    plt.axis('off')
    plt.ioff()
    plt.show()
    
def activation_visualize(model):
    ""