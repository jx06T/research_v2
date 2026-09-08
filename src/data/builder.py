import torch
from torch.utils.data import DataLoader
from .font_dataset import PairedFontDataset

def build_dataloader(config, src_font_path, tgt_font_path):
    # 此處保留擴充性，未來可根據 config 切換到 real_dataset
    dataset = PairedFontDataset(src_font_path, tgt_font_path, config)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=2
    )
    return dataset, dataloader
