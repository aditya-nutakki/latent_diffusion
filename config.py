# only parameters to be in this file

img_sz = 128
# m, c = 16, 128
m, c = 32, 4
image_dims = (3, img_sz, img_sz) # c, h, w
starting_filters = 64
# diffusion_model_dims = (3, 32, 32) # c, m, m
diffusion_model_dims = (c, m, m)
time_steps = 512

autoencoder_model_path = f"/mnt/d/work/projects/latent_diffusion/models/autoencoder_{m}_{c}_{starting_filters}_{img_sz}.pt"
device = "cuda"
batch_size = 24
epochs = 1000
lr = 3e-4
dataset_path = "/mnt/d/work/datasets/bikes/white_bg"

model_save_dir = "./models"
img_save_dir = "./samples/"

use_ddim_sampling = False
ddim_sampling_steps = 200 # applies only if 'use_ddim_sampling' is set to True


