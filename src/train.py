import argparse
import yaml
import os
import glob
import re
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data.font_dataset import PairedFontDataset
# 暫時沿用搬運過來的架構
from models.mm_legacy import DynamicGenerator

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--src_font', type=str, required=True)
    parser.add_argument('--tgt_font', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./runs')
    parser.add_argument('--save_interval', type=int, default=10, help='Save model every N epochs')
    parser.add_argument('--keep_checkpoints', type=int, default=3, help='Number of recent epoch checkpoints to keep (0 for keeping all)')
    return parser.parse_args()

def main():
    args = get_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset
    dataset = PairedFontDataset(args.src_font, args.tgt_font, config, num_samples=1000)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2)

    # Models
    # Note: mm_legacy expects an object with attributes for config, so we wrap the dict
    class Dict2Obj:
        def __init__(self, in_dict):
            for k, v in in_dict.items():
                setattr(self, k, v)
                
    cfg_obj = Dict2Obj(config)
    G = DynamicGenerator(cfg_obj).to(device)
    
    # 簡化版的 Optimizer
    optimizer_G = optim.Adam(G.parameters(), lr=config['lr'], betas=tuple(config['betas']))

    # 簡單 Training Loop 示意
    print("Starting Training Loop...")
    for epoch in range(config['num_epochs']):
        G.train()
        for i, (real_src, real_tgt, _) in enumerate(dataloader):
            real_src = real_src.to(device)
            real_tgt = real_tgt.to(device)
            
            # Dummy loss for demonstration 
            # (In reality, we would have Discriminator training and L1 loss here)
            noise = torch.randn(real_src.size(0), config['nz'], 1, 1).to(device)
            fake_tgt = G(noise, real_src)
            loss_G = torch.nn.functional.l1_loss(fake_tgt, real_tgt)
            
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()
            
        print(f"Epoch [{epoch+1}/{config['num_epochs']}] Loss: {loss_G.item():.4f}")
        
        # 儲存 Generator 權重供推論使用，並限制保存數量避免硬碟空間爆炸
        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == config['num_epochs']:
            # 1. 永遠儲存/覆蓋最新的一份，方便外部自動同步與測試
            latest_path = os.path.join(args.output_dir, "G_latest.pth")
            torch.save(G.state_dict(), latest_path)
            
            # 2. 儲存當前輪次 checkpoint
            save_path = os.path.join(args.output_dir, f"G_epoch_{epoch+1}.pth")
            torch.save(G.state_dict(), save_path)
            print(f"Saved weights to {save_path} and {latest_path}")

            # 3. 滾動清理：若超過 keep_checkpoints 限制，刪除最舊的 checkpoint
            if args.keep_checkpoints > 0:
                epoch_files = glob.glob(os.path.join(args.output_dir, "G_epoch_*.pth"))
                def extract_epoch(fname):
                    match = re.search(r'G_epoch_(\d+)\.pth$', fname)
                    return int(match.group(1)) if match else -1
                
                sorted_files = sorted(epoch_files, key=extract_epoch)
                while len(sorted_files) > args.keep_checkpoints:
                    oldest_file = sorted_files.pop(0)
                    try:
                        os.remove(oldest_file)
                        print(f"Cleaned up old checkpoint to save disk space: {os.path.basename(oldest_file)}")
                    except OSError as e:
                        print(f"Failed to delete {oldest_file}: {e}")

if __name__ == '__main__':
    main()

