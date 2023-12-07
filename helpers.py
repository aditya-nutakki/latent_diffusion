import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os, cv2
from torchvision.transforms import transforms

from torchvision.datasets.mnist import MNIST
from torchvision.datasets.cifar import CIFAR10

from config import *


# custom normalizing function to get into range you want
class NormalizeToRange(nn.Module):
    def  __init__(self, min_val, max_val) -> None:
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, images):
        # images could be a batch or individual
        return (self.max_val - self.min_val) * ((images - 0) / (1)) + self.min_val
    


class BikesDataset(Dataset):
    def __init__(self, dataset_path, limit = -1, _transforms = None, img_sz = 128) -> None:
        super().__init__()
        
        self.transforms = _transforms
        
        if not self.transforms:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((img_sz, img_sz)),
                # transforms.Normalize((0.5, ), (0.5,))
                NormalizeToRange(-1, 1)
            ])


        self.dataset_path, self.limit = dataset_path, limit
        self.valid_extensions = ["jpg", "jpeg", "png", "JPEG", "JPG"]
        
        self.images_path = dataset_path
        self.images = os.listdir(self.images_path)[:self.limit]

        self.images = [os.path.join(self.images_path, image) for image in self.images if image.split(".")[-1] in self.valid_extensions] 
        

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image)
        return image
        

    def preprocess_images(self, image_paths):
        # call this method incase you think there are invalid images
        clean_paths = []
        count = 0
        for index, image_path in enumerate(image_paths):
            try:
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # print(image.shape)
                image = self.transforms(image)

                clean_paths.append(image_path)
            except:
                print(f"failed at {image_path}")
                count += 1

        print(f"{count} number of invalid images")
        return clean_paths


def get_dataloader(dataset_type = "", batch_size = 8, img_sz = 128):

    if dataset_type == "mnist":
        ds = MNIST(root="./datasets", download=True,
                        transform=transforms.Compose([
                        transforms.Resize(img_sz), # or h
                        transforms.ToTensor(),
                        NormalizeToRange(-1, 1)
                        ])) 
        
    elif dataset_type == "cifar":
        ds = CIFAR10(root="./datasets", download=True,
                        transform=transforms.Compose([
                        transforms.Resize(img_sz),
                        transforms.ToTensor(),
                        NormalizeToRange(-1, 1)
                        ])) 
    else:
        ds = BikesDataset(dataset_path, img_sz = img_sz)
    
    print(f"Training on {len(ds)} samples; with batch size {batch_size}; image dims {image_dims} ...")
    return DataLoader(ds, batch_size = batch_size, shuffle = True, drop_last = True)



if __name__ == "__main__":
    ds = BikesDataset("/mnt/d/work/datasets/bikes/combined", img_sz=128)

