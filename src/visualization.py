from model import ForwardDiffusion, Denoiser, DiffusionSampler
import torch
from matplotlib import pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
from umap import UMAP
from PIL import Image
import numpy as np
from torchvision.transforms.functional import to_pil_image

### File for making visualizations for the final presentation
### If we want to make an activation atlas, we'll have to create a feature vector
### from the bottleneck layer of the UNet, and use that to place the images in 
### n dimensional space, then use PCA to plot all those images on the same plane




def process_hooks(arr: torch.Tensor):
    umap_arr = UMAP(n_neighbors=15, min_dist=0.1).fit_transform(arr)
    return umap_arr
        

def tensor_to_image(x):
    return to_pil_image(x.squeeze().permute(1, 2, 0).cpu().numpy(), mode='RGB')

def generate_image(name):
    torch.set_default_device('cuda')
    
    diffusor = ForwardDiffusion()
    model = Denoiser()
    
    hook_lst = []
    
    model.load_weights()
    model.eval()
    
    batch = 5    
    
    x = []
    hooks = []
    with torch.no_grad():
        sampler = DiffusionSampler(diffusor=diffusor)
        x_lst, hook_lst = sampler(model, batch, hook_timestep=5)
        
        hooks = torch.cat(hook_lst)
        coordinate = process_hooks(hooks)
        coordinate *= 400
        coordinate = np.round(coordinate, 0).astype(np.int32)
        
        atlas_mins = np.min(coordinate, axis=0)
        coordinate[:, 0] -= atlas_mins[0]
        coordinate[:, 1] -= atlas_mins[1]
       
        maxs = np.max(coordinate, axis=0)
        x_max = maxs[0]
        y_max = maxs[1]
        
        image = Image.new('RGB', (x_max, y_max))
        for elem, coord in zip(x_lst, coordinate):
            img = tensor_to_image(elem[-1, ...])
            image.paste(img, (coord[0], coord[1]))
        
        plt.imshow(image)
        plt.savefig(f'output/{name}.png')
        
        

if __name__ == '__main__':
    generate_image(f'noise')