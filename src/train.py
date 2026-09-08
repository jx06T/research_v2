import argparse
import os
import torch
import torch.optim as optim
import torch.nn.functional as F

from config.config_parser import Config
from data.builder import build_dataloader
from models.generator.dynamic_gen import DynamicGenerator
from models.discriminator.patch_gan import Discriminator
from models.loss import compute_gradient_penalty
from utils.logger import ExperimentLogger
from utils.visualize import plot_training_losses, test_specific_chars

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--src_font', type=str, required=True)
    parser.add_argument('--tgt_font', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./runs')
    parser.add_argument('--checkpoint_dir', type=str, default='./runs_checkpoints')
    parser.add_argument('--save_interval', type=int, default=100)
    return parser.parse_args()

def main():
    args = get_args()
    config = Config.from_yaml(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset & DataLoader
    dataset, dataloader = build_dataloader(config, args.src_font, args.tgt_font)
    
    # Logger
    logger = ExperimentLogger(config, output_dir=args.output_dir)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Models
    G = DynamicGenerator(config).to(device)
    D = Discriminator(config).to(device)
    print(f"Generator params: {sum(p.numel() for p in G.parameters())}")
    print(f"Discriminator params: {sum(p.numel() for p in D.parameters())}")

    opt_G = optim.Adam(G.parameters(), lr=config.lr, betas=config.betas)
    opt_D = optim.Adam(D.parameters(), lr=config.lr, betas=config.betas)

    print(f"[{logger.run_id}] Starting Training Loop...")
    
    for epoch in range(config.num_epochs):
        for i, (src_imgs, tgt_imgs, _) in enumerate(dataloader):
            bs = src_imgs.size(0)
            src_imgs = src_imgs.to(device)
            tgt_imgs = tgt_imgs.to(device)

            # 1. Train Discriminator
            for _ in range(config.n_critic):
                opt_D.zero_grad()
                z = torch.randn(bs, config.nz, config.bottleneck_size, config.bottleneck_size).to(device)
                fake_imgs = G(z, src_imgs).detach()

                loss_D = -torch.mean(D(tgt_imgs)) + torch.mean(D(fake_imgs))
                gp = compute_gradient_penalty(D, tgt_imgs, fake_imgs, device)
                loss_D += config.lambda_gp * gp

                loss_D.backward()
                opt_D.step()

            # 2. Train Generator
            opt_G.zero_grad()
            z = torch.randn(bs, config.nz, config.bottleneck_size, config.bottleneck_size).to(device)
            gen_imgs = G(z, src_imgs)

            loss_G_adv = -torch.mean(D(gen_imgs))
            loss_G_l1 = F.l1_loss(gen_imgs, tgt_imgs)
            loss_G = loss_G_adv + config.lambda_l1 * loss_G_l1

            loss_G.backward()
            opt_G.step()

            logger.log_metrics(loss_G.item(), loss_D.item(), loss_G_l1.item(), loss_G_adv.item())

        print(f"[Epoch {epoch}/{config.num_epochs}] G_loss: {loss_G.item():.4f} | L1: {loss_G_l1.item():.4f} | loss_G_adv: {loss_G_adv.item():.4f}")

        # Visualization
        if epoch % 30 == 0 or epoch == config.num_epochs - 1:
            test_specific_chars(G, dataset, config, epoch, device=device)

        if epoch % 50 == 0 or epoch == config.num_epochs - 1:
            plot_training_losses(logger.history, config)

        # Save Checkpoint
        if (epoch + 1) % args.save_interval == 0 or epoch == config.num_epochs - 1:
            # 1. 永遠覆寫最新版本 (供 demo_qt.py 或自動化下載使用)，加上 run_id 避免併發覆蓋
            latest_path = os.path.join(args.output_dir, f"G_latest_{logger.run_id}.pth")
            torch.save(G.state_dict(), latest_path)
            
            # 2. 儲存推論專用輕量檔 (僅包含 Generator 權重，檔案很小)
            # 加上 run_id 避免不同實驗互相覆蓋
            epoch_model_path = os.path.join(args.output_dir, f"G_{logger.run_id}_epoch_{epoch+1}.pth")
            torch.save(G.state_dict(), epoch_model_path)
            
            # 3. 儲存完整訓練狀態 (含 Optimizer 等)，隔離到 checkpoint_dir 避免被下載
            ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint_{logger.run_id}_epoch_{epoch+1}.pth")
            torch.save({
                'config': config.to_dict(),
                'epoch': epoch,
                'G_state_dict': G.state_dict(),
                'D_state_dict': D.state_dict(),
                'opt_G_state_dict': opt_G.state_dict(),
                'opt_D_state_dict': opt_D.state_dict(),
                'loss_history': logger.history
            }, ckpt_path)

if __name__ == '__main__':
    main()
