import torch
import torch.nn as nn
import math

class DynamicGenerator(nn.Module):
    def __init__(self, config):
        super(DynamicGenerator, self).__init__()
        self.nz = config.nz
        self.bottleneck_size = config.bottleneck_size

        needed_downsamples = int(math.log2(config.image_size / config.bottleneck_size))

        # --- 1. Content Encoder (source_encoder) ---
        layers = []
        in_ch, out_ch = config.nc, config.ngf

        for i in range(config.fixed_layers):
            is_last = (i == config.fixed_layers - 1)
            if i < needed_downsamples:
                k, s, p = config.kernel_size, 2, config.padding
            else:
                k, s, p = 3, 1, 1

            layers.append(nn.Conv2d(in_ch, out_ch, k, s, p, bias=False))
            if not is_last:
                layers.append(nn.BatchNorm2d(out_ch))
                layers.append(nn.LeakyReLU(0.2, inplace=True))

            in_ch = out_ch
            if i < needed_downsamples: out_ch = min(out_ch * 2, 512)

        self.source_encoder = nn.Sequential(*layers)
        self.enc_out_ch = in_ch

        # --- 2. Decoder (main_branch) ---
        layers = []
        dec_in_ch = self.enc_out_ch + config.nz
        num_flat_layers = config.fixed_layers - needed_downsamples

        for i in range(config.fixed_layers):
            is_last = (i == config.fixed_layers - 1)
            is_flat = i < num_flat_layers

            if is_flat:
                k, s, p = 3, 1, 1
                dec_out_ch = dec_in_ch
            else:
                k, s, p = config.kernel_size, 2, config.padding
                dec_out_ch = dec_in_ch // 2

            if is_last: dec_out_ch = config.nc

            if is_flat:
                layers.append(nn.Conv2d(dec_in_ch, dec_out_ch, k, s, p, bias=False))
            else:
                layers.append(nn.ConvTranspose2d(dec_in_ch, dec_out_ch, k, s, p, bias=False))

            if not is_last:
                layers.append(nn.BatchNorm2d(dec_out_ch))
                layers.append(nn.ReLU(True))
                dec_in_ch = dec_out_ch
            else:
                layers.append(nn.Sigmoid())

        self.main_branch = nn.Sequential(*layers)

    def forward(self, noise, src):
        feat = self.source_encoder(src)
        if noise.dim() == 2:
            noise = noise.view(noise.size(0), noise.size(1), 1, 1)
        noise_exp = noise.expand(-1, -1, feat.size(2), feat.size(3))
        combined = torch.cat((feat, noise_exp), dim=1)
        return self.main_branch(combined)

