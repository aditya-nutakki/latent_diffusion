# only parameters to be in this file


image_dims = (3, 128, 128) # c, h, w
# diffusion_model_dims = (3, 64, 64) # o, m, m
diffusion_model_dims = (3, 32, 32) # o, m, m

device = "cuda"
batch_size = 24
epochs = 1000
lr = 3e-4
dataset_path = "/mnt/d/work/datasets/bikes/white_bg"

save_dir = "./models"
img_save_dir = "./samples/"