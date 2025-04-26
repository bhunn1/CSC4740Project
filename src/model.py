from pathlib import Path
import torch
from torch import nn
from time import time
import math
import torchvision.io
import torchvision.transforms.functional
from sklearn.decomposition import PCA

TIMESTEPS = 1000 # Number of denoising timesteps the model samples from
UNET_CHANNEL_SCALE = 64 # Every layer in the UNet will have Channel_Scale * layer_number channels
UNET_BASE_DEPTH = 5 # Number of UNet down/up blocks (including bottleneck layer)
BETA_SCHEDULE = (1e-4, 1e-2)
LR = 1e-3 # Initial learning rate
BATCH_SIZE = 16
LR_PATIENCE=3 # Epoch patience of the learning rate scheduler, decreases lr after n epochs
EPOCHS = 100 
DECAY = 1e-6 # Weight decay for the optimizer
TIME_SCALE = 4 # Scale of time embedding dimensions, number of dims is time_scale * channel_scale
IMAGE_SIZE=256 # Side length for one dimension of the image

# Noise generator
class ForwardDiffusion:
    def __init__(self, 
                image_size=IMAGE_SIZE, 
                timesteps=TIMESTEPS,
                device='cuda',
                schedule_type='cosine'):
        self.device = device
        
        self.size = image_size
        self.timesteps = timesteps
        
        # Noise scheduler parameters
        if schedule_type == 'cosine':
            self.cosine_beta_schedule()
        elif schedule_type == 'linear':
            self.linear_beta_schedule()
        else:
            raise ValueError('Invalid schedule type argument, should be "cosine" or "linear"!')
        
        # Precalcuating commonly used coefficients
        self.sqrt_alpha_bar = self.alpha_hat.sqrt().view(-1, 1, 1, 1)
        self.sqrt_one_minus_alpha_bar = (1 - self.alpha_hat).sqrt().view(-1, 1, 1, 1)
    
    # Creates noise linearly from schedule[0] to schedule[1]
    def linear_beta_schedule(self, schedule=BETA_SCHEDULE):
        self.beta = torch.linspace(schedule[0], schedule[1], self.timesteps).to(self.device)
        self.alpha_hat = torch.cumprod(1.0 - self.beta, dim=0).to(self.device)
    
    # Creates noise in a sinusoidal pattern to help adjust model to noise at each step
    # This implementation is directly from the original paper
    def cosine_beta_schedule(self, s=0.008):
        steps = self.timesteps + 1
        x = torch.linspace(0, self.timesteps, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(((x / self.timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        
        alphas_cumprod = alphas_cumprod[:-1] / alphas_cumprod[0]
        
        self.alpha_hat = alphas_cumprod.to(self.device)
        self.beta = torch.clip(betas, 0, 0.9999).to(self.device)
        
    
    def diffuse(self, x):
        # Gaussian noise of the same shape as our input
        epsilon = torch.randn_like(x).to(self.device)
        
        # Randomly selecting a number of timesteps equal to the batch size,
        # then getting the scheduler values at those timesteps
        batch_size = x.size(0)
        t = torch.randint(0, self.timesteps, (batch_size,)).to(self.device)
        
        # Applying the markov chain (independent) diffusions
        xt = self.sqrt_alpha_bar[t] * x + self.sqrt_one_minus_alpha_bar[t] * epsilon
        return xt, epsilon, t
    
    def __call__(self, x):
        return self.diffuse(x)

# Encodes sequence data into a vector
class SinEncoder(nn.Module):
    def __init__(self, dim, timesteps=TIMESTEPS, device='cuda'):
        super().__init__()
        self.dim = dim
        self.device = device
        self.timesteps = timesteps
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dim, 2).float().to(self.device) / self.dim))
    
    def forward(self, t):
        t = t[:, None] * self.inv_freq[None, :]
        
        sin_encod = torch.sin(t).to(self.device)
        if self.dim % 2 == 0:
            cos_encod = torch.cos(t)
        else:
            cos_encod = torch.cos(t[..., :-1]).to(self.device)
        
        encoding = torch.cat((sin_encod, cos_encod), dim=-1).to(self.device)
        return encoding

# Encodes time data into a vector
class TimeEmbedder(nn.Module):
    def __init__(self, embedding_dim=TIME_SCALE*UNET_CHANNEL_SCALE, timesteps=TIMESTEPS):
        super().__init__()
        self.encoder = SinEncoder(embedding_dim, timesteps=timesteps)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 2 * embedding_dim),
            nn.SiLU(),
            nn.Linear(2*embedding_dim, embedding_dim)
        )
        
    def forward(self, t: torch.Tensor):
        embedding = self.encoder(t)
        return self.mlp(embedding)

#Downsampling convolution UNet layer
class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, embedding_dim, downsample=True):
        super().__init__()
        if downsample:
            initial_stride = 2
        else:
            initial_stride = 1
       
        self.time_proj = nn.Linear(embedding_dim, out_channel)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=initial_stride),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.ReLU()
        )
    def forward(self, x, t):
        t_proj = self.time_proj(t)
        t_proj = t_proj.unsqueeze(-1).unsqueeze(-1)
        return self.cnn(x) + t_proj

# Upsampling convolution UNet layer
class UpConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, embedding_dim):
        super().__init__()
        
        self.time_proj = nn.Linear(embedding_dim, out_channel)
        self.up_cnn = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 3, padding=1),
            nn.ReLU(),
        )
        
    def forward(self, x, t):
        t_proj = self.time_proj(t)
        t_proj = t_proj.unsqueeze(-1).unsqueeze(-1)
        return self.up_cnn(x) + t_proj

class Denoiser(nn.Module):
    def __init__(self, 
                 channels=3, 
                 depth=UNET_BASE_DEPTH,
                 channel_depth_base=UNET_CHANNEL_SCALE,
                 timesteps=TIMESTEPS,
                 embed_scale=TIME_SCALE):

        super().__init__()
        self.embedding_dim = embed_scale * channel_depth_base
        
        channel_depth = [channels] + list(map(lambda i: channel_depth_base * 2**(i), range(depth)))
        in_out_pairs = list(zip(channel_depth[:-1], channel_depth[1:]))
        
        # Gives model intuition on how noisy the input is
        self.time_embed = TimeEmbedder(embedding_dim=self.embedding_dim, timesteps=timesteps)
        
        # UNet architecture
        # encoders downsample data and increase filter count
        # decoders upsample data and decrease filter count
        # conceptually this is a convolutional autoencoder with diffusion sampling
        self.bottleneck = ConvBlock(channel_depth[-1], channel_depth[-1], self.embedding_dim, downsample=False)       
        self.encoders = nn.ModuleList([ConvBlock(in_ch, out_ch, self.embedding_dim) for in_ch, out_ch in in_out_pairs])
        self.decoders = nn.ModuleList(list(reversed([UpConvBlock(in_ch, out_ch, self.embedding_dim) for out_ch, in_ch in in_out_pairs])))
        self.output =  nn.Conv2d(channels, channels, kernel_size=1)
        
    def forward(self, x, t):        
        # Time vector embedding
        t = self.time_embed(t)
        
        # Downsampling with skips
        skips = [x]
        for encoder in self.encoders:
            x = encoder(x, t)
            skips.append(x)

        # Largest feature map
        x = self.bottleneck(x, t)
        
        # Upsampling adding skips
        for idx, decoder in enumerate(self.decoders):
            x = x + skips.pop()
            x = decoder(x, t)
            
        # Output layer (similar to a linear for logits)
        x = self.output(x) + skips.pop()
        return x
        
    def save_weights(self):
        path = str(Path('weights') / Path('unet-weights.pth'))
        torch.save(self.state_dict(), f=path)
        
    def load_weights(self):
        path = str(Path('weights') / Path('unet-weights.pth'))
        state_dict = torch.load(path)
            
        self.load_state_dict(state_dict)

class DiffusionSampler:
    def __init__(self, diffusor: ForwardDiffusion):
        self.diffusor = diffusor
        self.hook_lst = []
    
    # Saves a vector to plot output in an activation atlas
    def save_hook(self):
        def hook(module, input, output):
            vec = output.detach().cpu().flatten(start_dim=1)
            self.hook_lst.append(vec)
        return hook
    
    def __call__(self, model: Denoiser, batches=1, hook_timestep=None):
        # Random noise
        self.hook_lst = []
        handle = None
       
        x_t = torch.randn(batches, 3, self.diffusor.size, self.diffusor.size).to('cuda')
        
        x_lst = []
        for t in reversed(range(self.diffusor.timesteps)):
            if hook_timestep is not None:
                if t == hook_timestep:
                    handle = model.bottleneck.register_forward_hook(self.save_hook())
                elif handle is not None:
                    handle.remove()
                    handle = None
            
            t_tensor = torch.full((batches,), t, dtype=torch.long).to('cuda')
            
            beta = self.diffusor.beta[t]
            alpha = 1 - beta
            
            noise_pred = model(x_t, t_tensor)
            
            # Denoising process
            coef1 = 1 / torch.sqrt(alpha)
            coef2 = beta / self.diffusor.sqrt_one_minus_alpha_bar[t]
            x_pred = coef1 * (x_t - coef2 * noise_pred)
            
            if t > 0:
                noise = torch.randn_like(x_t).to('cuda')
                sigma_t = torch.sqrt(beta)
                x_t = x_pred + sigma_t * noise
            else:
                x_t = x_pred
                
            x_lst.append(x_t)
        x_lst = torch.stack(x_lst, dim=1)
        
        if hook_timestep is not None:
            return self.denormalize(x_lst), self.hook_lst
        else:
            return self.denormalize(x_lst)
    
    # Turns a [-1, 1] image into a [0, 1] image for plotting
    def denormalize(self, x):
        x_hat = torch.clamp((x + 1) / 2, 0, 1)
        return x_hat

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Standard training loop
def train_one_epoch(model: nn.Module, 
                    optim: torch.optim.Optimizer, 
                    loss_fn: torch.nn.MSELoss,
                    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
                    diffuser: ForwardDiffusion, 
                    dataloader: torch.utils.data.DataLoader):    
    total_loss = 0
    total_count = 0
    for x in dataloader:
        x = x.to('cuda', non_blocking=True)
            
        optim.zero_grad()
        
        x_t, noise, t = diffuser(x)
        noise_pred = model(x_t, t)
        loss = loss_fn(noise_pred, noise)
        
        loss.backward()
        optim.step()
        total_loss += loss.item()
        total_count += 1
    
    total_loss /= total_count
    scheduler.step(total_loss)
    return total_loss
    
def save_training_checkpoint(model, optim, scheduler, epoch):
    model_state_dict = model.state_dict()
    optim_state_dict = optim.state_dict()
    scheduler_state_dict = scheduler.state_dict()
    
    state = {
        'epoch':epoch,
        'model_state_dict':model_state_dict,
        'optim_state_dict':optim_state_dict,
        'scheduler_state_dict':scheduler_state_dict
    }
    
    path =str(Path('weights') / Path('unet-training-checkpoint.pth'))
    torch.save(state, path)
    
def load_training_checkpoint(load_lr=True):
    path = str(Path('weights') / Path('unet-training-checkpoint.pth'))
    state = torch.load(f=path)
    
    model = Denoiser()
    model.load_state_dict(
        state['model_state_dict']
    )

    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=DECAY)
    optim.load_state_dict(
        state['optim_state_dict']
    )

    if not load_lr:
        for group in optim.param_groups:
            group['lr'] = LR
        
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=LR_PATIENCE)
    scheduler.load_state_dict(
        state['scheduler_state_dict']
    )
    return model, optim, scheduler, state['epoch']
    
# Dummy trainer for one device training, not used for cluster computing
def train_generative(dataloader, epochs=EPOCHS, load_checkpoint=False, save_checkpoint=True):
    torch.set_default_device('cuda')
    
    if load_checkpoint:
        model, optim, scheduler, epoch_count = load_training_checkpoint()
        print(f'Starting from epoch {epoch_count+1}')
    else:
        model = Denoiser()
        optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=LR_PATIENCE)
        epoch_count = 0

    diffuser = ForwardDiffusion()
    loss_fn = nn.MSELoss()
    
    model.train()
    time_start = time()
    print("Starting training")
    for epoch in range(epochs):
        epoch_time_start = time()
        epoch_loss = train_one_epoch(model, optim, loss_fn, scheduler, diffuser, dataloader)
        time_end = time()
        
        current_lr = scheduler.get_last_lr()[0]
        print(f"""Epoch {epoch_count + epoch + 1}:
              Loss {round(epoch_loss, 4)},
              Epoch Time {round(time_end - epoch_time_start, 2)},
              Running Time {round(time_end - time_start, 2)},
              Current LR {format(current_lr, '.2e')}""")

        if save_checkpoint:   
            save_training_checkpoint(model=model, optim=optim, scheduler=scheduler, epoch=epoch_count + epoch)
            model.save_weights()
    model.save_weights()     
    
# Dataset loader for one device, used for testing not used for cluster computing
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir='data/resized/resized/'):
        import os
        
        paths = os.listdir(image_dir)
        self.paths = list(map(lambda x: Path(image_dir) / Path(x), paths))
         
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        raw_img = torchvision.io.decode_image(path, mode='RGB')
        img_resized = torchvision.transforms.functional.resize(raw_img, size=(IMAGE_SIZE, IMAGE_SIZE)).float()
        img = img_resized / 255
        img = img * 2 - 1
        return img
        
### One epoch ~ 150 seconds on my machine
### Haven't tested CPU - might be faster on smaller batches but I don't know if you can run 64 threads on CPU and matmult will be worse
### I'm training this model with the dummy dataset because its faster,
### we only get performance improvements from spark if we have distributed computing
### but we still need to implement it and train it for a little bit on spark

if __name__ == '__main__':
    torch.set_default_device('cuda')
    data = DummyDataset()
    trainloader = torch.utils.data.DataLoader(
        data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator(device='cuda')
    )
    train_generative(trainloader, epochs=EPOCHS, load_checkpoint=False)

    