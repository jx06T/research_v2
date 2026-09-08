import torch.nn as nn
import math

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        nc = config.nc
        ndf = config.ndf
        image_size = config.image_size

        # 計算需要多少次下採樣才能將尺寸降至 4x4
        # 假設 image_size 是 2 的次方 (如 64, 128, 256)
        n_downsampling = int(math.log2(image_size)) - 2
        
        layers = []
        in_ch = nc
        out_ch = ndf

        for i in range(n_downsampling):
            layers.append(nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_ch = out_ch
            out_ch = min(out_ch * 2, 512)  # 最大通道數限制在 512 避免顯存耗盡

        # Output decision: 4 -> 1
        layers.append(nn.Conv2d(in_ch, 1, 4, 1, 0, bias=False))
        self.main = nn.Sequential(*layers)

    def forward(self, input):
        return self.main(input).view(-1)

