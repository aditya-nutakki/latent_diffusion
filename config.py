import os
# only parameters to be in this file

img_sz = 128
m, c = 16, 4

image_dims = (3, img_sz, img_sz) # c, h, w
starting_filters = 64

diffusion_model_dims = (c, m, m)
time_steps = 1000

device = "cuda"
batch_size = 16
epochs = 10000
lr = 1e-3
dataset_path = "/mnt/d/work/datasets/bikes/bikes_clean"

base_path = f"./vaemodels_m{m}c{c}t{time_steps}"
os.makedirs(base_path, exist_ok = True)
model_save_dir = os.path.join(base_path, "models")
img_save_dir = os.path.join(base_path, "samples")
metrics_save_dir = os.path.join(base_path, "metrics")

use_ddim_sampling = False


