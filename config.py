# only parameters to be in this file

img_sz = 32
image_dims = (3, img_sz, img_sz) # c, h, w
# diffusion_model_dims = (3, 64, 64) # c, m, m
diffusion_model_dims = (3, 32, 32) # c, m, m

device = "cuda"
batch_size = 24
epochs = 1000
lr = 3e-4
dataset_path = "/mnt/d/work/datasets/bikes/white_bg"

save_dir = "./models"
img_save_dir = "./samples/"

use_ddim_sampling = False
ddim_sampling_steps = 200 # applies only if 'use_ddim_sampling' is set to True


