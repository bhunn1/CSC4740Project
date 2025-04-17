from model import ForwardDiffusion, Denoiser, DiffusionSampler
import torch
from matplotlib import pyplot as plt
import matplotlib
### File for making visualizations for the final presentation
### If we want to make an activation atlas, we'll have to create a feature vector
### from the bottleneck layer of the UNet, and use that to place the images in 
### n dimensional space, then use PCA to plot all those images on the same plane


def tensor_to_image(x):
    return x.squeeze().permute(1, 2, 0).cpu().numpy()

def generate_image(name):
    torch.set_default_device('cuda')
    
    diffusor = ForwardDiffusion()
    model = Denoiser()
    
    #model.load_weights()
    model.eval()
    
    images = 1    
    
    #fig, axes = plt.subplots(10, 10, figsize=(10, 10)) # 10 rows, 10 columns

    # Flatten the 2D array of axes for easy iteration
    #axes = axes.flatten()
    
    with torch.no_grad():
        sampler = DiffusionSampler(diffusor=diffusor)
        x_lst = sampler(model, images)
        
        # n = x_lst.size(0) - 1
        # axe_count = len(axes)
        
        # print(x_lst.size())
        # for idx, ax in enumerate(axes):
        #     img = x_lst[(n-axe_count) + idx, ...].squeeze().permute(1, 2, 0).cpu().numpy()
        #     ax.imshow(img)
        plt.imshow(tensor_to_image(x_lst[-1]))
        plt.savefig(f'output/{name}.png')




if __name__ == '__main__':
    for i in range(1):
        generate_image(f'noise{i}')