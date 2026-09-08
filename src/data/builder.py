import torch
from torch.utils.data import DataLoader
from .font_dataset import PairedFontDataset

import os

def build_dataloader(config, src_font_path, tgt_font_path):
    # 此處保留擴充性，未來可根據 config 切換到 real_dataset
    dataset = PairedFontDataset(src_font_path, tgt_font_path, config)
    
    # 動態取得系統核心數，避免 Colab 出現 Worker 過多的警告與效能瓶頸
    num_workers = min(12, os.cpu_count() or 2)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    return dataset, dataloader

