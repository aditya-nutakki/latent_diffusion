import torch.nn as nn
import torch, torchvision
import torch.nn.functional as F
import os, numpy, json
from time import time

from ldm import LatentDiffusion, VAE

from helpers import *
from config import *



def _load_ldm_model(model_path, time_steps):
    ldm = LatentDiffusion(m = 16, c = 4, time_steps = time_steps)
    ldm.load_state_dict(torch.load(model_path))
    ldm.to(device)
    return ldm


def infer(model_path, time_steps, num_samples = 8, use_ddim_sampling = use_ddim_sampling):
    ldm = _load_ldm_model(model_path = model_path, time_steps = time_steps)
    print(f"Loaded LatentDiffusion model from {model_path}")

    if use_ddim_sampling:
        print("Using DDIM Sampling")
        for i in range(4):
            ldm.ddim_sample(ep = f"test50_{i}", num_samples = num_samples, sample_steps = 999, eta = 0)

    else:
        print("Proceeding without DDIM Sampling")
        for i in range(4):
            ldm.sample(ep = f"m{m}c{c}_{i}", num_samples = num_samples)



if __name__ == "__main__":
    model_path = "./vaemodels_m16c4t1000/models/ldm_200.pt"
    infer(model_path = model_path, num_samples = 64, use_ddim_sampling = False)


