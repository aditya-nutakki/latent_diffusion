import torch.nn as nn
import torch, torchvision
import torch.nn.functional as F
import os, numpy, json
from modules import Encoder, Decoder
from math import log

import torchshow
from time import time

from ldm import LatentDiffusion

from helpers import *
from config import *



def _load_ldm_model():
    ldm = LatentDiffusion()
    ldm.load_state_dict(torch.load("./models/ldm_200.pt"))
    ldm.to(device)
    return ldm


def infer(num_samples = 24):
    ldm = _load_ldm_model()
    print("Loaded LatentDiffusion model...")

    for i in range(4):
        ldm.sample(ep = f"200_{i}", num_samples = num_samples)



if __name__ == "__main__":
    infer(num_samples = 24)


