import os
# only parameters to be in this file

img_sz = 128
# m, c = 16, 128
m, c = 16, 16
image_dims = (3, img_sz, img_sz) # c, h, w
starting_filters = 64
# diffusion_model_dims = (3, 32, 32) # c, m, m
diffusion_model_dims = (c, m, m)
time_steps = 200

autoencoder_model_path = f"/mnt/d/work/projects/latent_diffusion/models/autoencoder_{m}_{c}_{starting_filters}_{img_sz}.pt"
device = "cuda"
batch_size = 16
epochs = 10000
lr = 6e-4
dataset_path = "/mnt/d/work/datasets/bikes/bikes_clean"

base_path = f"./vaemodels_m{m}c{c}t{time_steps}"
os.makedirs(base_path, exist_ok = True)
model_save_dir = os.path.join(base_path, "models")
img_save_dir = os.path.join(base_path, "samples")
metrics_save_dir = os.path.join(base_path, "metrics")

use_ddim_sampling = False
ddim_sampling_steps = 200 # applies only if 'use_ddim_sampling' is set to True


