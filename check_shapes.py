import torch
import sys

from src.config.config_parser import Config
from src.models.generator.dynamic_gen import DynamicGenerator
from src.models.discriminator.patch_gan import Discriminator
from src.models.loss import compute_gradient_penalty

def main():
    try:
        config = Config.from_yaml('configs/l1_100_n8_size128.yaml')
        device = torch.device('cpu')
        
        G = DynamicGenerator(config).to(device)
        D = Discriminator(config).to(device)
        
        print("Generator instantiated.")
        print("Discriminator instantiated.")
        
        bs = 2
        src_imgs = torch.randn(bs, config.nc, config.image_size, config.image_size).to(device)
        tgt_imgs = torch.randn(bs, config.nc, config.image_size, config.image_size).to(device)
        z = torch.randn(bs, config.nz, config.bottleneck_size, config.bottleneck_size).to(device)
        
        # Generator Forward
        gen_imgs = G(z, src_imgs)
        print(f"Generator output shape: {gen_imgs.shape}")
        assert gen_imgs.shape == tgt_imgs.shape, "Generator output shape mismatch!"
        
        # Discriminator Forward
        d_out = D(gen_imgs)
        print(f"Discriminator output shape: {d_out.shape}")
        assert d_out.shape == (bs,), f"Discriminator output shape mismatch! Expected {(bs,)}, got {d_out.shape}"
        
        # Loss check (Gradient Penalty)
        gp = compute_gradient_penalty(D, tgt_imgs, gen_imgs, device)
        print(f"Gradient Penalty: {gp.item()}")
        
        print("All shape checks passed successfully!")
    except Exception as e:
        print(f"Error during shape check: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
