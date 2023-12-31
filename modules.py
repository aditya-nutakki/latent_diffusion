import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

""" Convolutional block:
    It follows a two 3x3 convolutional layer, each followed by a batch normalization and a relu activation.
"""
class conv_block(nn.Module):
    def __init__(self, in_c, out_c, activation = "relu", embedding_dims = None):
        super().__init__()


        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        
        # using groupnorm instead of batchnorm as suggested in the paper
        self.gn1 = nn.GroupNorm(8, num_channels = out_c)
        
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, num_channels = out_c)

        self.embedding_dims = embedding_dims if embedding_dims else out_c
        self.embedding = nn.Embedding(time_steps, embedding_dim = self.embedding_dims) # temporarily say number of positions is 512 by default, change it later. Ideally it should be num_time_steps from the ddpm
        self.relu = nn.ReLU()
        self.act = nn.ReLU() if activation == "relu" else nn.SiLU()

        
    def forward(self, inputs, time = None):

        time_embedding = self.embedding(time).view(-1, self.embedding_dims, 1, 1)
        # print(f"time embed shape => {time_embedding.shape}")
        # print(inputs.shape)
        x = self.conv1(inputs)
        x = self.gn1(x)
        # x = self.relu(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.gn2(x)
        # x = self.relu(x)
        x = self.act(x)

        x = x + time_embedding
        # print(f"conv block {x.shape}")
        # print()
        return x

""" Encoder block:
    It consists of an conv_block followed by a max pooling.
    Here the number of filters doubles and the height and width half after every block.
"""
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c, activation = "relu"):
        super().__init__()

        self.conv = conv_block(in_c, out_c, activation = activation, embedding_dims = out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs, time = None):
        x = self.conv(inputs, time)
        p = self.pool(x)

        return x, p

""" Decoder block:
    The decoder block begins with a transpose convolution, followed by a concatenation with the skip
    connection from the encoder block. Next comes the conv_block.
    Here the number filters decreases by half and the height and width doubles.
"""
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c, activation = "relu"):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c, activation = activation, embedding_dims = out_c)

    def forward(self, inputs, skip, time = None):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x, time)

        return x


class UNet(nn.Module):
    def __init__(self, input_channels = 3, output_channels = 3, num_steps = 512, down_factor = 1):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.down_factor = down_factor

        self.num_steps = num_steps
        print(f"UNET with {self.num_steps}")
        # self.embedding = nn.Embedding(self.num_steps, 512)

        self.e1 = encoder_block(self.input_channels, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        self.b = conv_block(512, 1024) # bottleneck

        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)
    
        self.outputs = nn.Conv2d(64, self.output_channels, kernel_size=1, padding=0)


    def forward(self, inputs, t = None):
        # downsampling block
        # print(inputs.shape)
        # print(embedding.shape)
        s1, p1 = self.e1(inputs, t)
        # print(f"p1 shape {p1.shape}")
        # print(s1.shape, p1.shape)
        s2, p2 = self.e2(p1, t)
        # print(s2.shape, p2.shape)
        s3, p3 = self.e3(p2, t)
        # print(s3.shape, p3.shape)
        s4, p4 = self.e4(p3, t)
        # print(s4.shape, p4.shape)

        b = self.b(p4, t)
        # print(b.shape)
        # print()
        # upsampling block
        d1 = self.d1(b, s4, t)
        # print(d1.shape)
        # # print(f"repeat {d1.shape}")
        d2 = self.d2(d1, s3, t)
        # print(d2.shape)
        d3 = self.d3(d2, s2, t)
        # print(d3.shape)
        d4 = self.d4(d3, s1, t)
        # print(d4.shape)

        outputs = self.outputs(d4)
        # print(f"ddpm output {outputs.shape}")
        # print()
        # print(f"output minmax: {torch.min(outputs)}, {torch.max(outputs)}")
        return outputs


class ResnetBlock(nn.Module):
    def __init__(self, in_c, out_c) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels = in_c, out_channels = out_c, kernel_size = 3, padding = 1)
        self.norm = nn.GroupNorm(num_groups = 4, num_channels = out_c)

        self.pool = nn.MaxPool2d((2,2))
        self.act = nn.ReLU()


    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.norm(x)
        # print(f"down block x: {x.shape}")
        return self.act(x)


class Encoder(nn.Module):
    # VAE's Encoder Module
    def __init__(self, input_channels = 3, dims = [32, 64, 128]) -> None:
        super().__init__()
        # self.input_channels = input_channels
        self.dims = dims

        self.init_block = ResnetBlock(in_c = input_channels, out_c = self.dims[0])
        self.blocks = nn.ModuleList([ResnetBlock(in_c = self.dims[i], out_c = self.dims[i + 1]) for i in range(len(self.dims) - 1)])

    def forward(self, x):
        x = self.init_block(x)
        
        for _block in self.blocks:
            x = _block(x)

        return x


class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, num_groups = 4, activation = "leaky_relu") -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = nn.Conv2d(out_c, out_c, kernel_size = 3, padding = 1)
        self.norm = nn.GroupNorm(num_groups = num_groups, num_channels = out_c)
        self.act = nn.LeakyReLU() if activation == "leaky_relu" else nn.Tanh()

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        x = self.norm(x)
        # print(f"upblock: {x.shape}")
        return self.act(x)


class Decoder(nn.Module):
    # VAE's Decoder Module
    def __init__(self, output_channels = 3, dims = [128, 64, 32]) -> None:
        super().__init__()
        self.dims = dims # dims to be the reverse of dims passed in the encoder block
        self.up_blocks = nn.ModuleList([UpBlock(in_c = dims[i], out_c = self.dims[i + 1]) for i in range(len(self.dims) - 1)])
        self.final_block = UpBlock(dims[-1], output_channels, num_groups = 1, activation = "tanh") # equivalent to layer norm here because we set num_groups to 1

    def forward(self, x):

        for _block in self.up_blocks:
            x = _block(x)

        x = self.final_block(x)
        return x



if __name__ == "__main__":    
    # device = "cuda:0"
    # batch_size = 2
    # in_channels, w = 3, 128
    # inputs = torch.randn((batch_size, in_channels, w, w), device=device)
    # randints = torch.randint(1, 512, (batch_size, ), device=device)
    # model = UNet().to(device)
    # print(f"model has {sum([p.numel() for p in model.parameters()])} params")
    # y = model(inputs, randints)
    # print(y.shape)
    

    # x = torch.randn(4, 3, 128, 128)
    # enc = Encoder(m = 4, latent_channels= 64)
    # y = enc(x)
    # print(y.shape, y.mean(), y.std())
    # print()
    
    # dec = Decoder()
    # # y = torch.randn(4, 128, 4, 4)
    # z = dec(y)
    # print(z.shape, z.mean(), z.std())

    pass