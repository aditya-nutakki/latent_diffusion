import torch.nn as nn
import torch, torchvision
import torch.nn.functional as F
import os, numpy, json
from modules import *
from diffusion import DiffusionModel
from ae import AutoEncoder, train_ae
from vae import VAE

import torchshow
from time import time
from random import uniform

from helpers import *
from config import *

from sys import exit

"""

1. Freeze Autoencoder (both encoder and decoder)
2. Pass input image to the encoder 
3. Noise it and proceed as usual with your ddpm
4. Reverse it back with the decoder. 

"""



class LatentDiffusion(nn.Module):
    def __init__(self, m, c, inference_mode = False, autoencoder_model_path = autoencoder_model_path, time_steps = time_steps) -> None:
        super().__init__()

        self.time_steps = time_steps
        self.m, self.c = m, c
        self.autoencoder_model_path = autoencoder_model_path
        self.inference_mode = inference_mode
        self.image_dims = image_dims
        self.latent_image_dims = diffusion_model_dims

        self.diffusion_model = DiffusionModel(time_steps = self.time_steps, image_dims = diffusion_model_dims, output_channels = c)
        self.autoencoder = self.load_autoencoder_model()
    

    def load_autoencoder_model(self):
        model = VAE(m = self.m, c = self.c)
        
        if self.inference_mode:
            # load vae
            path = f"./vaemodels_m{m}c{c}/vaemodel_46.pt" # m, c = 16, 4
            model.load_state_dict(torch.load(path).state_dict())
            print(f"Loaded VAE model from {path}")
        
        model.eval() # eval mode
        # freeze model 
        for param in model.parameters():
            param.requires_grad = False

        return model


    def ddim_sample(self, ep, sample_steps, num_samples = 16, eta = 0.0):
        assert sample_steps < self.time_steps, f"sampling steps should be lesser than number of time steps"
        
        self.diffusion_model.model.eval()
        print(f"sampling {num_samples} examples with ddim... ")
        with torch.no_grad():
            times = torch.linspace(1, self.time_steps - 1, sample_steps).to(torch.long)
            times = list(reversed(times.int().tolist()))
            time_pairs = list(zip(times[:-1], times[1:]))
            x = torch.randn(num_samples, *self.latent_image_dims, device = device)
            stime = time()
            for t, t_minus_one in time_pairs:
                noise = torch.randn(num_samples, *self.latent_image_dims, device = device)
                alpha_t, alpha_t_minus_one = self.diffusion_model.alpha_hats[t], self.diffusion_model.alpha_hats[t_minus_one]
                
                t = torch.tensor(t, device = device).long()
                pred_noise = self.diffusion_model(x, t)

                sigma = eta * torch.sqrt((1-alpha_t_minus_one)/(1 - alpha_t) * (1 - (alpha_t/alpha_t_minus_one)))
                
                k = torch.sqrt(1 - alpha_t_minus_one - sigma**2)
                pred_x0 = torch.sqrt(alpha_t_minus_one) * (x - torch.sqrt(1 - alpha_t)*pred_noise)/torch.sqrt(alpha_t)

                x = pred_x0 + k * pred_noise + sigma * noise
            
            ftime = time()
            x = self.autoencoder.decode(x)
            # torchshow.save(x, os.path.join(img_save_dir, f"latent_ddim_sample_{ep}.jpeg"))
            print(f"Done denoising in {ftime - stime}s ")
        
        return x


    def sample(self, ep = None, num_samples = 16):
        self.diffusion_model.model.eval()
        print(f"Sampling {num_samples} examples...")
        stime = time()
        with torch.no_grad():
            
            x = torch.randn(num_samples, *self.latent_image_dims, device = device)
            for i, t in enumerate(range(self.time_steps - 1, 0, -1)):
                alpha_t, alpha_t_hat, beta_t = self.diffusion_model.alphas[t], self.diffusion_model.alpha_hats[t], self.diffusion_model.betas[t]
                t = torch.tensor(t, device = device).long()
                x = (torch.sqrt(1/alpha_t))*(x - (1-alpha_t)/torch.sqrt(1 - alpha_t_hat) * self.diffusion_model.model(x, t))

                if i > 1:
                    noise = torch.randn_like(x)
                    x = x + torch.sqrt(beta_t) * noise
            
        ftime = time()
        x = self.autoencoder.decode(x)
        print(f"decoded shape: {x.shape}")
        # torchshow.save(x, os.path.join(img_save_dir, f"latent_sample_{ep}.jpeg"))
        print(f"Done denoising in {ftime - stime}s")
        return x


    def save_model(self, ep):
        model_path = os.path.join(model_save_dir, f"ldm_{ep}.pt")
        torch.save(self.state_dict(), model_path)
        print(f"saved model to {model_path}")


    def forward(self, x):
        bs = x.shape[0]
        z, _, _ = self.autoencoder.encode(x) # get latent space
        ts = torch.randint(low = 1, high = self.time_steps, size = (bs, ), device = device)
        z_noised, noise = self.diffusion_model.add_noise(z, ts)
        return z_noised, noise, ts



def train_ldm(load_checkpoint = False, continue_from = 199 + 1):
    ldm = LatentDiffusion(m = m, c = c, autoencoder_model_path = autoencoder_model_path, time_steps = time_steps)
    
    if load_checkpoint:
        _model_path = os.path.join(model_save_dir, f"ldm_200_old.pt")
        ldm.load_state_dict(torch.load(_model_path))
        print(f"loaded ldm weights from {_model_path}")

    loader = get_dataloader(dataset_type="custom", img_sz = img_sz, batch_size = batch_size, limit = -1)

    opt = torch.optim.Adam(ldm.diffusion_model.model.parameters(), lr = lr) # optimizing only unet parameters
    criterion = nn.MSELoss(reduction="mean")

    ldm.autoencoder.to(device)
    ldm.diffusion_model.model.to(device)
    print(f"Model training on m = {m}, c = {c}, image_dims = {image_dims}, t = {time_steps}")
    global_losses = []

    for ep in range(epochs):
        ldm.diffusion_model.model.train()
        print(f"Epoch {ep + continue_from}:")
        losses = []
        stime = time()
        
        for i, x in enumerate(loader):
            x = x.to(device)
            z_noised, target_noise, ts = ldm(x)
            
            predicted_noise = ldm.diffusion_model(z_noised, ts)
            loss = criterion(target_noise, predicted_noise)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            losses.append(loss.item())
            global_losses.append(loss.item())

            if i % 100 == 0:
                print(f"Loss: {loss.item()}; step {i}; epoch {ep + continue_from}")

        plot_metrics(global_losses, title = "ldm_loss")
        ftime = time()
        print(f"Epoch trained in {ftime - stime}s; Avg loss => {sum(losses)/len(losses)}")

        if (ep + continue_from) % 5 == 0:
            ldm.sample(ep + continue_from)
            ldm.save_model(ep + continue_from)
            print("Saved model")

        print()



def test_noise():
    ldm = LatentDiffusion(autoencoder_model_path = autoencoder_model_path, time_steps = time_steps)
    loader = get_dataloader(dataset_type="custom", img_sz = img_sz, batch_size = batch_size)

    ldm.autoencoder.to(device)
    ldm.diffusion_model.model.to(device)

    for i, x in enumerate(loader):
        x = x.to(device)
        
        # vae 
        z, _, _ = ldm.autoencoder.encode(x)
        z_noised, target_noise, ts = ldm(x)
        noised_recons_image = ldm.autoencoder.decode(z_noised)
        recons_image = ldm.autoencoder.decode(z)


        print(ts)
        torchshow.save(x, f"init_image_{i}.jpeg")
        torchshow.save(recons_image, f"recons_image_{i}.jpeg")
        torchshow.save(noised_recons_image, f"noised_recons_image_{i}.jpeg")

        if i == 5: break

if __name__ == "__main__":
    train_ldm(load_checkpoint = True)
    # test_noise()

