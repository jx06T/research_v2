import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    """ Self attention Layer for spatial dimension """
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, max(1, in_channels // 8), 1)
        self.key = nn.Conv2d(in_channels, max(1, in_channels // 8), 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch, C, W, H = x.size()
        proj_query = self.query(x).view(batch, -1, W*H).permute(0, 2, 1)
        proj_key = self.key(x).view(batch, -1, W*H)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value(x).view(batch, -1, W*H)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch, C, W, H)
        return self.gamma * out + x

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        return torch.cat((x, skip_input), 1)

class AttnUNetGenerator(nn.Module):
    def __init__(self, config):
        super(AttnUNetGenerator, self).__init__()
        self.nz = config.nz
        self.bottleneck_size = config.bottleneck_size
        
        needed_downsamples = int(math.log2(config.image_size / config.bottleneck_size))
        
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.attns_down = nn.ModuleDict()
        self.attns_up = nn.ModuleDict()
        
        in_ch = config.nc
        out_ch = config.ngf
        
        # --- Encoder ---
        self.down_channels = []
        current_res = config.image_size
        
        for i in range(needed_downsamples):
            self.down_blocks.append(UNetDown(in_ch, out_ch, normalize=(i > 0)))
            self.down_channels.append(out_ch)
            in_ch = out_ch
            out_ch = min(out_ch * 2, 512)
            current_res //= 2
            
            # Add attention at 32x32 and 16x16
            if current_res in [32, 16]:
                self.attns_down[str(i)] = SelfAttention(in_ch)
                
        # --- Bottleneck Flat Layers ---
        self.flat_layers = nn.ModuleList()
        num_flat = config.fixed_layers - needed_downsamples
        for i in range(num_flat):
            self.flat_layers.append(nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, 1, 1, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.LeakyReLU(0.2, inplace=True)
            ))
            
        # --- Decoder ---
        # Input to decoder is bottleneck output + style noise
        dec_in_ch = in_ch + config.nz
        
        # Decoder Flat Layers
        self.dec_flat_layers = nn.ModuleList()
        for i in range(num_flat):
            self.dec_flat_layers.append(nn.Sequential(
                nn.Conv2d(dec_in_ch, dec_in_ch, 3, 1, 1, bias=False),
                nn.BatchNorm2d(dec_in_ch),
                nn.ReLU(inplace=True)
            ))
            
        # Up blocks
        for i in reversed(range(needed_downsamples)):
            if i == 0:
                # Final layer
                dec_out_ch = config.nc
                self.final_up = nn.Sequential(
                    nn.ConvTranspose2d(dec_in_ch, dec_out_ch, 4, 2, 1, bias=False),
                    nn.Sigmoid()
                )
            else:
                dec_out_ch = self.down_channels[i-1]
                self.up_blocks.append(UNetUp(dec_in_ch, dec_out_ch))
                dec_in_ch = dec_out_ch + self.down_channels[i-1] # after concat
                
                current_res *= 2
                # Add attention at 16x16 and 32x32
                if current_res in [16, 32]:
                    self.attns_up[str(i)] = SelfAttention(dec_in_ch)

    def forward(self, noise, src):
        skips = []
        x = src
        
        # Encoder
        for i, down in enumerate(self.down_blocks):
            x = down(x)
            if str(i) in self.attns_down:
                x = self.attns_down[str(i)](x)
            skips.append(x)
            
        # Bottleneck Flat
        for flat in self.flat_layers:
            x = flat(x)
            
        # Inject Style Noise
        if noise.dim() == 2:
            noise = noise.view(noise.size(0), noise.size(1), 1, 1)
        noise_exp = noise.expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat((x, noise_exp), dim=1)
        
        # Decoder Flat
        for dec_flat in self.dec_flat_layers:
            x = dec_flat(x)
            
        # Decoder Up
        for idx, up in enumerate(self.up_blocks):
            i_down = len(self.down_blocks) - 1 - idx
            skip = skips[i_down - 1]
            x = up(x, skip)
            
            if str(i_down) in self.attns_up:
                x = self.attns_up[str(i_down)](x)
                
        # Final Up
        x = self.final_up(x)
        return x
