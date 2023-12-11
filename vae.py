# based on https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
import torch
from torch import nn
from torchvision import transforms
from helpers import *

IMAGE_SIZE = 128
# LATENT_DIM = 1024 # m4; c64

m, c = 4, 128
LATENT_DIM = m * m * c
image_dim = 3 * IMAGE_SIZE * IMAGE_SIZE 

print('IMAGE_SIZE', IMAGE_SIZE, 
      'LATENT_DIM', LATENT_DIM, 
      'image_dim', image_dim)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # hidden_dims = [32, 64, 128, 128, 256] # m16 c64
        # hidden_dims = [32, 64, 128, c] # (64, 8, 8)
        # hidden_dims = [64, 128, c] # (16, 16, 16)
        hidden_dims = [32, 64, 128, 256, c] # (64, 8, 8)
        self.final_dim = hidden_dims[-1]
        in_channels = 3
        modules = []

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        out = self.encoder(torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE))
        print(f"out shape: {out.shape}")
        self.size = out.shape[2] # 4
        self.fc_mu = nn.Linear(hidden_dims[-1] * self.size * self.size, LATENT_DIM)
        self.fc_var = nn.Linear(hidden_dims[-1] * self.size * self.size, LATENT_DIM)

        # Build Decoder
        modules = []
        # self.decoder_input = nn.Linear(LATENT_DIM, hidden_dims[-1] * self.size * self.size)
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Sigmoid())


    def encode(self, x):
        result = self.encoder(x)
        # print(f"encoded shape: {result.shape}")
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        # print(f"mu: {mu.shape}; logvar: {log_var.shape}")
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z):
        # print(f"init z shape: {z.shape}")
        # result = self.decoder_input(z) # commenting this out for now; please check again !!!!!
        result = z.view(-1, self.final_dim, self.size, self.size)
        # print(f"re viewed shape: {result.shape}")
        result = self.decoder(result)
        # print(f"decoded shape: {result.shape}")
        result = self.final_layer(result)
        # print(f"final shape: {result.shape}")
        # result = torch.flatten(result, start_dim=1)
        result = torch.nan_to_num(result)
        return result

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var
    


##################################

import os
import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchvision.utils import save_image


EPOCHS = 500
BATCH_SIZE = 16
device = "cuda"
print('EPOCHS', EPOCHS, 'BATCH_SIZE', BATCH_SIZE, 'device', device)

directory = f'./vaemodels_m{m}c{c}'
os.makedirs(directory, exist_ok = True)
print(directory)

train_loader, test_loader = get_dataloader(batch_size=BATCH_SIZE), get_dataloader(batch_size=BATCH_SIZE)


model_path = "/mnt/d/work/projects/vae/vaemodels_sigmoid/vae_model_retraining_58.pth"
# ckpt = torch.load(model_path).state_dict()

model = VAE().to(device)
# model.load_state_dict(ckpt)

print(sum([p.numel() for p in model.parameters()]))
# print(f"model loaded from {model_path}")

optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, log_var):
    MSE =F.mse_loss(recon_x, x)
    KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    kld_weight = 0.00025
    loss = MSE + kld_weight * KLD  
    return loss


def train(epoch):
    model.train()
    train_loss = 0
    # for batch_idx, (data, _) in enumerate(train_loader):
    for batch_idx, data in enumerate(train_loader):
        torch.cuda.empty_cache()
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(data)

        # print(torch.min(data), torch.max(data))
        # print(torch.min(recon_batch), torch.max(recon_batch))
        # break

        log_var = torch.clamp_(log_var, -5, 5)
        loss = loss_function(recon_batch, data, mu, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data)))

    print('Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))
    print()

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        # for i, (data, _) in enumerate(test_loader):
        for i, data in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, log_var = model(data)
            test_loss += loss_function(recon_batch, data, mu, log_var).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)[:n]])
                save_image(comparison.cpu(),
                           f'{directory}/reconstruction_retraining_{str(continue_from + epoch)}.png', nrow=n)
            break

continue_from = 0

if __name__ == "__main__":
    print(f'epochs: {EPOCHS}')

    for epoch in range(1, EPOCHS + 1):
        train(epoch)
        torch.save(model, f'{directory}/vae_model_retraining_{continue_from + epoch}.pth')
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, LATENT_DIM).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 3, IMAGE_SIZE, IMAGE_SIZE),
                       f'{directory}/sample_retraining_{str(continue_from + epoch)}.png')




