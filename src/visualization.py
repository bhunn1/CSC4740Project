from model import ForwardDiffusion, Denoiser, DiffusionSampler
import torch
from matplotlib import pyplot as plt

### File for making visualizations for the final presentation
### If we want to make an activation atlas, we'll have to create a feature vector
### from the bottleneck layer of the UNet, and use that to place the images in 
### n dimensional space, then use PCA to plot all those images on the same plane



def generate_image():
    torch.set_default_device('cuda')
    
    diffusor = ForwardDiffusion()
    model = Denoiser()
    
    model.load_weights()
    model.eval()
    
    images = 1    
    
    with torch.no_grad():
        sampler = DiffusionSampler(diffusor=diffusor)
        predictions = sampler(model, images)
        for idx in range(predictions.size(0)):
            img = predictions[idx, ...].permute(1, 2, 0).cpu().numpy()
            plt.imshow(img)
            plt.show()




if __name__ == '__main__':
    generate_image()