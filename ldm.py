import torch.nn as nn
import torch, torchvision
import torch.nn.functional as F
import os, numpy, json
from modules import *
from diffusion import DiffusionModel
from ae import AutoEncoder

import torchshow
from time import time

from helpers import *
from config import *



"""

1. Freeze Autoencoder (both encoder and decoder)
2. Pass input image to the encoder 
3. Noise it and proceed as usual with your ddpm
4. Reverse it back with the decoder. 

doubts;
    1. whats the loss function apart from Reconstruction loss ?
    2. Lets ignore the D_phi model for now. 
    3. 


"""



class LatentDiffusion(nn.Module):
    def __init__(self, autoencoder_model_path, time_steps = 512) -> None:
        super().__init__()

        self.time_steps = time_steps
        self.autoencoder_model_path = autoencoder_model_path
        
        self.image_dims = image_dims
        self.latent_image_dims = diffusion_model_dims

        self.diffusion_model = DiffusionModel(time_steps = self.time_steps, output_channels = c)
        self.autoencoder = self.load_autoencoder_model()
        # call self.autoencoder.encoder and self.autoencoder.decoder individually to encode and decode


    def load_autoencoder_model(self):
        model_name = self.autoencoder_model_path.split("/")[-1]
        m, c, starting_filters, img_sz = model_name.split(".")[:-1][0].split("_")[1:]
        model = AutoEncoder(m = m, c = c, starting_filters = starting_filters, image_dims = (self.image_dims[0], img_sz, img_sz))
        model.eval() # eval mode

        for param in model.parameters():
            param.requires_grad = False # freeze model 

        return model


    def sample(self, ep, num_samples = batch_size):
        self.diffusion_model.model.eval()
        print(f"Sampling {num_samples} samples...")
        stime = time()
        with torch.no_grad():
            # x = torch.randn(num_samples, self.input_channels, self.img_size, self.img_size, device = device)
            x = torch.randn(num_samples, *self.latent_image_dims, device = device)
            for i, t in enumerate(range(self.time_steps - 1, 0 , -1)):
                alpha_t, alpha_t_hat, beta_t = self.diffusion_model.alphas[t], self.diffusion_model.alpha_hats[t], self.diffusion_model.betas[t]
                # print(alpha_t, alpha_t_hat, beta_t)
                t = torch.tensor(t, device = device).long()
                x = (torch.sqrt(1/alpha_t))*(x - (1-alpha_t)/torch.sqrt(1 - alpha_t_hat) * self.diffusion_model.model(x, t))
                if i > 1:
                    noise = torch.randn_like(x)
                    x = x + torch.sqrt(beta_t) * noise
        ftime = time()
        torchshow.save(x, os.path.join(img_save_dir, f"latent_sample_{ep}.jpeg"))
        print(f"Done denoising in {ftime - stime}s ")


    def forward(self, x):
        bs = x.shape[0]
        z = self.autoencoder.encoder(x) # get latent space
        ts = torch.randint(low = 1, high = self.time_steps, size = (bs, ), device = device)
        z_noised, noise = self.diffusion_model.add_noise(z, ts)

        return z_noised, noise, ts



def train_ldm():
    ldm = LatentDiffusion(autoencoder_model_path = "/mnt/d/work/projects/latent_diffusion/models/autoencodertanh_16_128_32_128.pt", time_steps = time_steps)
    c, h, w = diffusion_model_dims
    assert h == w, f"height and width must be same, got {h} as height and {w} as width"

    loader = get_dataloader(dataset_type="custom", img_sz = h, batch_size = batch_size)

    opt = torch.optim.Adam(ldm.diffusion_model.model.parameters(), lr = lr) # optimizing only unet parameters
    criterion = nn.MSELoss(reduction="mean")

    ldm.autoencoder.to(device)
    ldm.diffusion_model.model.to(device)

    for ep in range(epochs):
        ldm.diffusion_model.model.train()
        print(f"Epoch {ep}:")
        losses = []
        stime = time()
        
        # for i, x in enumerate(loader):
        for i, (x, _) in enumerate(loader):

            x = x.to(device)
            # ts = torch.randint(low = 1, high = ddpm.time_steps, size = (bs, ), device = device)            
            # x, target_noise = ddpm.add_noise(x, ts)
            
            z_noised, target_noise, ts = ldm(x)
            # print(x.shape)
            predicted_noise = ldm.diffusion_model.model(x, ts)
            loss = criterion(target_noise, predicted_noise)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            losses.append(loss.item())

            if i % 200 == 0:
                print(f"Loss: {loss.item()}; step {i}; epoch {ep}")

        ftime = time()
        print(f"Epoch trained in {ftime - stime}s; Avg loss => {sum(losses)/len(losses)}")

        if (ep + 5) % 1 == 0:
            ldm.sample(ep)
        
        print()



if __name__ == "__main__":
    train_ldm()

