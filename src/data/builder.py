import torch
from torch.utils.data import DataLoader
from .font_dataset import PairedFontDataset
from .scanned_glyph_dataset import ScannedGlyphDataset

import os

def build_dataloader(config, src_font_path, tgt_font_path=None, manifest_path=None, writer_id=""):
    # 此處保留擴充性，未來可根據 config 切換到 real_dataset
    if manifest_path:
        dataset = ScannedGlyphDataset(manifest_path, src_font_path, config, writer_id=writer_id)
    else:
        dataset = PairedFontDataset(src_font_path, tgt_font_path, config)
    
    # 動態取得系統核心數，避免 Colab 出現 Worker 過多的警告與效能瓶頸
    num_workers = min(4 if manifest_path else 12, os.cpu_count() or 2)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    print(f"DataLoader initialized with {num_workers} workers.")
    return dataset, dataloader

