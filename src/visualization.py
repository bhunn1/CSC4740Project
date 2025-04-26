from model import ForwardDiffusion, Denoiser, DiffusionSampler
import torch
from matplotlib import pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
from umap import UMAP
from PIL import Image
import numpy as np
from torchvision.transforms.functional import to_pil_image
import torchvision
### File for making visualizations for the final presentation
### If we want to make an activation atlas, we'll have to create a feature vector
### from the bottleneck layer of the UNet, and use that to place the images in 
### n dimensional space, then use PCA to plot all those images on the same plane


# Convert an activation hook into a 2D vector based on a manifold approximation
def process_hooks(arr: torch.Tensor):
    umap_arr = UMAP(n_neighbors=15, min_dist=0.1).fit_transform(arr)
    return umap_arr
        
def tensor_to_image(x):
    return x.squeeze().permute(1, 2, 0).cpu()

# Attempts to generate an activation atlas using the UMAP method, can be memory hungry
# in its current form
def generate_atlas(name):
    torch.set_default_device('cuda')
    
    diffusor = ForwardDiffusion()
    model = Denoiser()
    
    hook_lst = []
    
    model.load_weights()
    model.eval()
    
    batch = 3    
    
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
        
        image = Image.new('RGB', (x_max + 256, y_max + 256))
        for elem, coord in zip(x_lst, coordinate):
            img = tensor_to_image(elem[-1, ...])
            image.paste(img, (coord[0], coord[1]))
        
        plt.imshow(image)
        plt.savefig(f'output/{name}.png')
    
# Generates one image and shows the image at different noise levels
def generate_image(name="image"):
    torch.set_default_device('cuda')
    
    diffusor = ForwardDiffusion()
    model = Denoiser()
    
    
    model.load_weights()
    model.eval()     
    with torch.no_grad():
        sampler = DiffusionSampler(diffusor=diffusor)
        x_lst = reversed(sampler(model, 1))
        print(x_lst.size())
        for t in range(0, diffusor.timesteps, diffusor.timesteps // 10):
            img = tensor_to_image(x_lst[0, t, ...])
            plt.imshow(img)
            plt.show()
        img = tensor_to_image(x_lst[0, -1, ...])
        plt.imshow(img)
        plt.savefig(f'output/{name}.png')
        
# Tests the forward noising process on one sample image
def generate_tests(*args, **kwargs):
    torch.set_default_device('cuda')
    
    diffusor = ForwardDiffusion()
    sampler = DiffusionSampler(diffusor)
    
    path = 'data/resized/resized/William_Turner_44.jpg'
    raw_img = torchvision.io.decode_image(path, mode='RGB')
    img_resized = torchvision.transforms.functional.resize(raw_img, size=(256, 256)).permute((1, 2, 0)).float().cuda()
    img = img_resized / 255
    
    plt.imshow(img.cpu())
    plt.show()
    img = img * 2 - 1
    imgs = []
    
    epsilon = torch.randn_like(img).cuda()
    for t in range(diffusor.timesteps):
        xt = diffusor.sqrt_alpha_bar[t] * img + diffusor.sqrt_one_minus_alpha_bar[t] * epsilon
        imgs.append(sampler.denormalize(xt))
    
    for idx in range(0, len(imgs), 10):
        plt.imshow(imgs[idx].cpu())
        plt.show()
        
if __name__ == '__main__':
    generate_image(f'test')