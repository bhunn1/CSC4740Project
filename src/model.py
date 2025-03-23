from pathlib import Path
import torch
from torch import nn
from time import time
import math
import torchvision.io
import torchvision.transforms.functional

# Noise generator
class ForwardDiffusion:
    def __init__(self, 
                 image_size=512, 
                 timesteps=1000,
                 schedule=(1e-4, 2e-2),
                 device='cuda'):
        self.device = device
        
        self.size = image_size
        self.timesteps = timesteps
        
        # Noise scheduler parameters
        self.beta = torch.linspace(schedule[0], schedule[1], timesteps).to(self.device)
        self.alpha = torch.cumprod(1.0 - self.beta, dim=0).to(self.device)
    
    def diffuse(self, x):
        # Gaussian noise of the same shape as our input
        epsilon = torch.rand_like(x).to(self.device)
        
        # Randomly selecting a number of timesteps equal to the batch size,
        # then getting the scheduler values at those timesteps
        batch_size = x.size(0)
        t = torch.randint(0, self.timesteps, (batch_size,)).to(self.device)
        alpha = self.alpha[t].reshape(batch_size, 1, 1, 1)
        
        # Applying the markov chain (independent) diffusions
        xt = torch.sqrt(alpha) * x + torch.sqrt(1 - alpha) * epsilon
        return xt, epsilon, t
    
    def __call__(self, x):
        return self.diffuse(x)

# Encodes sequence data into a vector
class SinEncoder(nn.Module):
    def __init__(self, dim, timesteps=1000, device='cuda'):
        super().__init__()
        self.dim = dim
        self.device = device
        self.timesteps = timesteps
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dim, 2).float().to(self.device) / self.dim))
    
    def forward(self, t):
        t = t / self.timesteps
        t = t.unsqueeze(-1) * self.inv_freq
        
        sin_encod = torch.sin(t).to(self.device)
        if self.dim % 2 == 0:
            cos_encod = torch.cos(t)
        else:
            cos_encod = torch.cos(t[..., :-1]).to(self.device)
        
        encoding = torch.cat((sin_encod, cos_encod), dim=-1).to(self.device)
        return encoding

# Encodes time data into a vector
class TimeEmbedder(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.encoder = SinEncoder(embedding_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
    def forward(self, t: torch.Tensor):
        pos_encoding = self.encoder(t)
        out = self.mlp(pos_encoding)
        
        return out

#Downsampling convolution UNet layer
class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, downsample=True):
        super().__init__()
        if downsample:
            initial_stride = 2
        else:
            initial_stride = 1
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=initial_stride),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.ReLU()
        )
    def forward(self, x):
        return self.cnn(x)

# Upsampling convolution UNet layer
class UpConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.up_cnn = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.up_cnn(x)

class Denoiser(nn.Module):
    def __init__(self, 
                 channels=3, 
                 depth=4,
                 channel_depth_base=64):

        super().__init__()
        
        
        channel_depth = [channels] + list(map(lambda i: channel_depth_base * 2**(i), range(depth)))
        in_out_pairs = list(zip(channel_depth[:-1], channel_depth[1:]))
        
        # Gives model intuition on how noisy the input is
        self.time_embed = TimeEmbedder(embedding_dim=channel_depth[-1])
        
        # UNet architecture
        # encoders downsample data and increase filter count
        # decoders upsample data and decrease filter count
        # conceptually this is a convolutional autoencoder with diffusion sampling
        self.bottleneck = ConvBlock(channel_depth[-1], channel_depth[-1], downsample=False)       
        self.encoders = nn.ModuleList([ConvBlock(in_ch, out_ch) for in_ch, out_ch in in_out_pairs])
        self.decoders = nn.ModuleList(list(reversed([UpConvBlock(in_ch, out_ch) for out_ch, in_ch in in_out_pairs])))
        self.output =  nn.Conv2d(channels, channels, kernel_size=1)
        
    def forward(self, x, t):        
        # Time vector embedding
        t = self.time_embed(t)
        
        # Downsampling with skips
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)

        # Largest feature map
        x = self.bottleneck(x)
        # Reshaping the time embedding to match feature map dimensions
        t = t.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.size(2), x.size(3))
        x = x + t 
        
        # Upsampling adding skips
        for idx, decoder in enumerate(self.decoders):
            x = x + skips[-(idx+1)]
            x = decoder(x)
            
        # Output layer (similar to a linear for logits)
        x = self.output(x)
        return x
        
    def save_weights(self):
        with open(Path('weights') / Path('unet_weights.pth')) as file:
            torch.save(self.state_dict(), f=file)
        
    def load_weights(self):
        with open(Path('weight') / Path('unet-weights.pth')) as file:
            state_dict = torch.load(file, weights_only=True)
            
        self.load_state_dict(state_dict)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_one_epoch(model: nn.Module, 
                    optim: torch.optim.Optimizer, 
                    loss_fn,
                    scheduler: torch.optim.lr_scheduler.LinearLR,
                    diffuser: ForwardDiffusion, 
                    dataloader: torch.utils.data.DataLoader):    
    total_loss = 0
    for x in dataloader:
        optim.zero_grad()
        
        x_t, noise, t = diffuser(x)
        noise_pred = model(x_t, t)
        loss = loss_fn(noise_pred, noise)
        
        loss.backward()
        optim.step()
        total_loss += loss.item()
        
    total_loss /= len(dataloader)
    scheduler.step(total_loss)
    return total_loss
    
def train_generative(dataloader, epochs=100):
    model = Denoiser()
    diffuser = ForwardDiffusion()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=5)
    
    time_start = time()
    print("Starting training")
    for epoch in range(epochs):
        epoch_time_start = time()
        epoch_loss = train_one_epoch(model, optim, loss_fn, scheduler, diffuser, dataloader)
        time_end = time()
        
        print(f"Epoch {epoch + 1}: Loss {epoch_loss}, Epoch Time {round(time_end - epoch_time_start, 2)}, Running Time {round(time_end - time_start, 2)}")
    
    model.save_weights()        

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir='data/resized/resized/', size_limit=16*32):
        import os
        
        paths = os.listdir(image_dir)
        paths = list(map(lambda x: Path(image_dir) / Path(x), paths))
        
        self.images = []  
        for path in paths:
            raw_img = torchvision.io.decode_image(path, mode='RGB')
            img_resized = torchvision.transforms.functional.resize(raw_img, size=(512, 512)).float()
            img = img_resized / 255
            self.images.append(img)
            
            if len(self.images) >= size_limit:
                break
        
        self.images = torch.stack(self.images, dim=0).to('cuda')
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx]
        

if __name__ == '__main__':
    torch.set_default_device('cuda')
    data = DummyDataset()
    trainloader = torch.utils.data.DataLoader(
        data,
        batch_size=16,
        shuffle=True,
        generator=torch.Generator(device='cuda')
    )
    train_generative(trainloader)

    