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

from helpers import *
from config import *



"""

1. Freeze Autoencoder (both encoder and decoder)
2. Pass input image to the encoder 
3. Noise it and proceed as usual with your ddpm
4. Reverse it back with the decoder. 

"""



class LatentDiffusion(nn.Module):
    def __init__(self, autoencoder_model_path = autoencoder_model_path, time_steps = time_steps) -> None:
        super().__init__()

        self.time_steps = time_steps
        self.autoencoder_model_path = autoencoder_model_path
        
        # if not os.path.exists(autoencoder_model_path):
        #     print(f"Autoencoder model not found, training ...")
        #     train_ae(epochs = 40)

        self.image_dims = image_dims
        self.latent_image_dims = diffusion_model_dims

        # self.diffusion_model = DiffusionModel(time_steps = self.time_steps, output_channels = c)
        self.diffusion_model = DiffusionModel(time_steps = self.time_steps, image_dims = (16, 16, 16), output_channels = c)
        self.autoencoder = self.load_autoencoder_model()
        # call self.autoencoder.encoder and self.autoencoder.decoder individually to encode and decode


    def load_autoencoder_model(self):
        # model_name = self.autoencoder_model_path.split("/")[-1]
        # m, c, starting_filters, img_sz = model_name.split(".")[:-1][0].split("_")[1:]
        # m, c, starting_filters, img_sz = int(m), int(c), int(starting_filters), int(img_sz)
        # model = AutoEncoder(m = m, c = c, starting_filters = starting_filters, image_dims = (self.image_dims[0], img_sz, img_sz))

        # load vae
        print("Loading VAE model")
        path = "/mnt/d/work/projects/vae/vaemodels_m16c16/vae_model_retraining_145.pth"
        model = VAE(m = 16, c = 16)
        model.load_state_dict(torch.load(path).state_dict())
        print("Loaded VAE model")
        model.eval() # eval mode

        # freeze model 
        for param in model.parameters():
            param.requires_grad = False

        return model


    def sample(self, ep = None, num_samples = batch_size, use_ddim_sampling = use_ddim_sampling):
        self.diffusion_model.model.eval()
        print(f"Sampling {num_samples} samples...")
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
        # x = self.autoencoder.decoder(x)
        print(f"sampling ... xshape: {x.shape}")
        x = self.autoencoder.decode(x)
        print(f"decoded shape: {x.shape}")
        torchshow.save(x, os.path.join(img_save_dir, f"latent_sample_{ep}.jpeg"))
        print(f"Done denoising in {ftime - stime}s ")


    def save_model(self, ep):
        model_path = os.path.join(model_save_dir, f"ldm_{ep}.pt")
        torch.save(self.state_dict(), model_path)


    def forward(self, x):
        bs = x.shape[0]
        # z = self.autoencoder.encoder(x) # get latent space
        z, _, _ = self.autoencoder.encode(x) # get latent space
        # print(f"encoded shape: {z.shape}")
        ts = torch.randint(low = 1, high = self.time_steps, size = (bs, ), device = device)
        z_noised, noise = self.diffusion_model.add_noise(z, ts)
        # print(f"ldm shapes: {z_noised.shape}; {noise.shape}; {ts.shape}")
        return z_noised, noise, ts



def train_ldm(load_checkpoint = False):
    ldm = LatentDiffusion(autoencoder_model_path = autoencoder_model_path, time_steps = time_steps)
    # c, h, w = img_sz
    # assert h == w, f"height and width must be same, got {h} as height and {w} as width"
    
    if load_checkpoint:
        ldm.load_state_dict(torch.load("./models_vae/ldm_5.pt"))
        print("loaded ldm weights")

    # assert os.path.exists(autoencoder_model_path), f"{autoencoder_model_path} not found !"

    loader = get_dataloader(dataset_type="custom", img_sz = img_sz, batch_size = batch_size)

    opt = torch.optim.Adam(ldm.diffusion_model.model.parameters(), lr = lr) # optimizing only unet parameters
    criterion = nn.MSELoss(reduction="mean")

    ldm.autoencoder.to(device)
    ldm.diffusion_model.model.to(device)
    print(f"Model training on m = {m}, c = {c}, image_dims = {image_dims}")
    global_losses = []

    for ep in range(epochs):
        ldm.diffusion_model.model.train()
        print(f"Epoch {ep}:")
        losses = []
        stime = time()
        
        for i, x in enumerate(loader):
        # for i, (x, _) in enumerate(loader):
            x = x.to(device)
            # print(f"init x shape {x.shape}")
            z_noised, target_noise, ts = ldm(x)
            # print(f"znoised shape {z_noised.shape}")

            predicted_noise = ldm.diffusion_model(z_noised, ts)
            loss = criterion(target_noise, predicted_noise)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            losses.append(loss.item())
            global_losses.append(loss.item())

            if i % 100 == 0:
                print(f"Loss: {loss.item()}; step {i}; epoch {ep}")

        plot_metrics(global_losses, title = "ldm_loss")
        ftime = time()
        print(f"Epoch trained in {ftime - stime}s; Avg loss => {sum(losses)/len(losses)}")

        if (ep) % 5 == 0:
            ldm.sample(ep)
            ldm.save_model(ep)
            print("Saved model")

        print()



def test_noise():
    ldm = LatentDiffusion(autoencoder_model_path = autoencoder_model_path, time_steps = time_steps)
    loader = get_dataloader(dataset_type="custom", img_sz = img_sz, batch_size = batch_size)

    ldm.autoencoder.to(device)
    ldm.diffusion_model.model.to(device)

    for i, x in enumerate(loader):
        x = x.to(device)

        # z = ldm.autoencoder.encoder(x)
        # z_noised, target_noise, ts = ldm(x)
        # noised_recons_image = ldm.autoencoder.decoder(z_noised)
        # recons_image = ldm.autoencoder.decoder(z)

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

