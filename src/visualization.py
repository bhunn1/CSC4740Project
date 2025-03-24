from model import ForwardDiffusion, Denoiser, DiffusionSampler
import torch
from matplotlib import pyplot as plt


if __name__ == '__main__':
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