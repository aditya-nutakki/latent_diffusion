import torch.nn as nn
import torch, torchvision
import torch.nn.functional as F
import os, numpy, json
from modules import Encoder, Decoder
from math import log

import torchshow
from time import time

from helpers import *
from config import *


class AutoEncoder(nn.Module):
    def __init__(self, m, c, image_dims = (3, 128, 128), starting_filters = 32) -> None:
        super().__init__()
        self.m, self.c = m, c
        self.latent_dims = (c, m, m)
        self.starting_filters = starting_filters 
        self.img_sz, self.input_channels = image_dims[-1], image_dims[0]
    
        self.dims = self.get_dims(starting_filters = self.starting_filters)
        
        self.encoder = Encoder(input_channels = self.input_channels, dims = self.dims)
        self.decoder = Decoder(output_channels = self.input_channels, dims = list(reversed(self.dims))) # in order to get back the same shape as the input


    def get_dims(self, starting_filters = 16):
        # input is expected to be of the shape (3, 128, 128) or a power of 2
        num_iters = int(log(self.img_sz // self.m, 2))
        dims = []
        
        for _ in range(num_iters - 1):
            dims.append(starting_filters)
            starting_filters *= 2

        dims.append(self.c)
        return dims


    def forward(self, x):
        x = self.encoder(x)
        # bs, c, m, _ = x.shape
        # print(f"latent dims = {x.shape}")
        # assert m == self.m and c == self.c, f"latent dimensions not as described as the given input, got {m} as height/wdith and {c} channels" 
        return self.decoder(x)



def eval(model, loader, ep):
    model.eval()
    save_path = os.path.join(img_save_dir, f"recons_images_{ep}.jpeg")
    with torch.no_grad():
        # for i, (images, _) in enumerate(loader):
        for i, images in enumerate(loader):
            images = images.to(device)
            recons_images = model(images)
            torchshow.save(recons_images, save_path)
            torchshow.save(images, os.path.join(img_save_dir, f"actual_images_{ep}.jpeg"))
            print(f"reconstructed image saved to {save_path}")
            break


def train_ae(epochs = epochs):
    # m, c = 8, 128
    print(f"Model training on m = {m}, c = {c}, image_dims = {image_dims}")
    model = AutoEncoder(m = m, c = c, image_dims = image_dims)
    print(f"Training on {sum([p.numel() for p in model.parameters()])} parameters")
    
    loader = get_dataloader(dataset_type="custom", img_sz = img_sz, batch_size = batch_size)

    opt = torch.optim.Adam(model.parameters(), lr = lr)
    criterion = nn.MSELoss()

    os.makedirs(model_save_dir, exist_ok = True)
    model_name = f"autoencoder_{m}_{c}_{model.starting_filters}_{img_sz}.pt"
    
    model.to(device)
    for ep in range(epochs):
        model.train()
        print(f"Epoch {ep}:")
        losses = []
        stime = time()

        # for i, (images, _) in enumerate(loader):
        for i, images in enumerate(loader):
            images = images.to(device)

            opt.zero_grad()

            # print(f"images min max: {torch.min(images), torch.max(images)}")

            reconstructed_images = model(images)
            loss = criterion(reconstructed_images, images)
            loss.backward()
            opt.step()

            losses.append(loss.item())

            if i % 200 == 0:
                print(f"Loss: {loss.item()}; step {i}; epoch {ep}")

        ftime = time()
        print(f"Epoch trained in {ftime - stime}s; Avg loss => {sum(losses)/len(losses)}")
        eval(model, loader, ep)

        if (ep + 1) % 1 == 0:
            
            model_save_path = os.path.join(model_save_dir, model_name)
            torch.save(model.state_dict(), model_save_path)
            print(f"model saved to {model_save_path}")

        print()


if __name__ == "__main__":
    train_ae()
    # img_dims = (3, 128, 128)
    # ae = AutoEncoder(m = 4, c = 128, image_dims=img_dims)
    # x = torch.randn(4, *img_dims)
    # y = ae(x)
    # print(y.shape)