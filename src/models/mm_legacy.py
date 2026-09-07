# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import os
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import string
import math
from scipy import linalg
from torchvision.models import inception_v3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class EvalFontDataset(Dataset):
    """
    專門用於評估/測試的 Dataset。
    特點：
    1. 循序取樣 (Deterministic)，不隨機。
    2. 強制關閉所有 Data Augmentation (確保評估標準一致)。
    3. 只需傳入要測試的字符列表 (target_chars)。
    4. 可指定 num_samples 進行重複採樣 (Repeating)。
    """
    def __init__(self, font_path_a, font_path_b, config, target_chars, invert=True, num_samples=None, content_scale=0.8):
        """
        Args:
            num_samples (int, optional): 指定要生成的總樣本數。
                                         如果不填 (None)，預設為 len(target_chars)，即每個字只測一次。
                                         如果填了 (例如 20000)，會循環重複測試這些字。
        """
        self.img_size = config.image_size
        self.invert = invert
        self.chars = target_chars # 直接使用傳入的列表
        self.content_scale = content_scale

        # 設定資料集長度
        if num_samples is None:
            self.total_samples = len(self.chars)
        else:
            self.total_samples = num_samples
            
        print(f"Eval Dataset Created. Total samples: {self.total_samples} (Unique chars: {len(self.chars)})")
        
        # 字體設置
        try:
            self.font_a = ImageFont.truetype(font_path_a, size=int(config.image_size * 1.1))
            self.font_b = ImageFont.truetype(font_path_b, size=int(config.image_size * 1.06))
        except IOError:
            raise RuntimeError("字體文件未找到，請檢查路徑。")

    def _render_char_to_tensor(self, char_str: str, font: ImageFont) -> torch.Tensor:
        # (保持原有的渲染邏輯不變，直接複製過來即可)
        canvas_size = (int(self.img_size * 2), int(self.img_size * 2))
        img_pil = Image.new("L", canvas_size, color=255)
        draw = ImageDraw.Draw(img_pil)

        bbox = draw.textbbox((0, 0), char_str, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        position = ((canvas_size[0] - text_width) / 2, (canvas_size[1] - text_height) / 2)
        draw.text(position, char_str, font=font, fill=0)

        img_tensor = transforms.ToTensor()(img_pil)

        non_white = torch.where(img_tensor < 1.0)
        if non_white[0].numel() == 0: return torch.ones(1, self.img_size, self.img_size)

        top, bottom = torch.min(non_white[1]), torch.max(non_white[1])
        left, right = torch.min(non_white[2]), torch.max(non_white[2])

        pad = 5
        img_cropped = img_tensor[:, max(0, top-pad):min(canvas_size[1], bottom+pad),
                                    max(0, left-pad):min(canvas_size[0], right+pad)]

        final_canvas = torch.ones(1, self.img_size, self.img_size)
        c_h, c_w = img_cropped.shape[1], img_cropped.shape[2]
        ratio = min((self.img_size * self.content_scale) / c_h, (self.img_size * self.content_scale) / c_w)
        new_h, new_w = int(c_h * ratio), int(c_w * ratio)

        resized_img = F.interpolate(img_cropped.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)

        y_off = (self.img_size - new_h) // 2
        x_off = (self.img_size - new_w) // 2
        final_canvas[:, y_off:y_off+new_h, x_off:x_off+new_w] = resized_img

        return final_canvas

    def __len__(self):
        # 回傳設定好的總樣本數
        return self.total_samples

    def __getitem__(self, idx):
        # 1. 循序且循環地取出字符 (Deterministic Round-Robin)
        # 使用 idx % len(self.chars) 確保 idx 超過字數時會回到第一個字
        char_str = self.chars[idx % len(self.chars)]
        
        # 2. 渲染 (無增強)
        img_a = 1.0 - self._render_char_to_tensor(char_str, self.font_a)
        img_b = 1.0 - self._render_char_to_tensor(char_str, self.font_b)

        # 3. 處理反色
        if not self.invert:
            return 1.0 - img_a, 1.0 - img_b, 0 # label 設為 0
        return img_a, img_b, 0


class PairedFontDataset(Dataset):
    def __init__(self, font_path_a, font_path_b, config, num_samples=5000, invert=True):
        self.num_samples = num_samples
        self.img_size = config.image_size
        self.invert = invert
        self.cfg = config

        # 字體設置
        try:
            self.font_a = ImageFont.truetype(font_path_a, size=int(config.image_size * 1.1))
            self.font_b = ImageFont.truetype(font_path_b, size=int(config.image_size * 1.06))
        except IOError:
            raise RuntimeError("字體文件未找到，請檢查路徑。")

         # --- 1. 定義全集 (數字 + 大寫 + 小寫) ---
        digits = [str(i) for i in range(10)]
        upper = list(string.ascii_uppercase)
        lower = list(string.ascii_lowercase)
        all_candidates = digits + upper + lower

        # --- 2. 硬排除 (Hard Exclude) ---
        # 這些是「渲染效果很差」或「過寬」的字，完全不使用
        hard_exclude = {}

        valid_chars = [ch for ch in all_candidates if ch not in hard_exclude]

        # --- 3. 根據 Config 分組 ---
        # Missing Set: 雖然是有效字，但在訓練時假裝沒有 Target (Mask=0)
        self.missing_set = [c for c in valid_chars if c in config.missing_chars]

        # Common Set: 有效且有 Target 的普通訓練字 (Mask=1)
        self.characters = [c for c in valid_chars if c not in self.missing_set]

        print(f"總有效字符: {len(valid_chars)}")
        print(f"├── 普通訓練字符 (Common): {len(self.characters)} 個")
        print(f"└── 缺失測試字符 (Missing/Zero-shot): {self.missing_set}")

        self.char_to_label = {char: i for i, char in enumerate(self.characters)}

    def _render_char_to_tensor(self, char_str: str, font: ImageFont) -> torch.Tensor:
        # 畫布稍大一點以免邊緣裁切
        canvas_size = (int(self.img_size * 2), int(self.img_size * 2))
        img_pil = Image.new("L", canvas_size, color=255)
        draw = ImageDraw.Draw(img_pil)

        bbox = draw.textbbox((0, 0), char_str, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        position = ((canvas_size[0] - text_width) / 2, (canvas_size[1] - text_height) / 2)
        draw.text(position, char_str, font=font, fill=0)

        img_tensor = transforms.ToTensor()(img_pil)

        # 裁剪內容邏輯 (保持原樣或微調)
        non_white = torch.where(img_tensor < 1.0)
        if non_white[0].numel() == 0: return torch.ones(1, self.img_size, self.img_size)

        top, bottom = torch.min(non_white[1]), torch.max(non_white[1])
        left, right = torch.min(non_white[2]), torch.max(non_white[2])

        # 增加一點 padding
        pad = 5
        img_cropped = img_tensor[:, max(0, top-pad):min(canvas_size[1], bottom+pad),
                                    max(0, left-pad):min(canvas_size[0], right+pad)]

        # Resize and Center on Final Canvas
        final_canvas = torch.ones(1, self.img_size, self.img_size)

        # 保持長寬比縮放
        c_h, c_w = img_cropped.shape[1], img_cropped.shape[2]
        ratio = min((self.img_size * 0.8) / c_h, (self.img_size * 0.8) / c_w)
        new_h, new_w = int(c_h * ratio), int(c_w * ratio)

        resized_img = F.interpolate(img_cropped.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)

        y_off = (self.img_size - new_h) // 2
        x_off = (self.img_size - new_w) // 2
        final_canvas[:, y_off:y_off+new_h, x_off:x_off+new_w] = resized_img

        return final_canvas

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        char_str = random.choice(self.characters)
        label = self.char_to_label[char_str]
        # label = 0 # 簡化，本任務不依賴標籤分類

        img_a = 1.0 - self._render_char_to_tensor(char_str, self.font_a)
        img_b = 1.0 - self._render_char_to_tensor(char_str, self.font_b)

        # 同步增強
        params = transforms.RandomAffine.get_params(
            degrees=(-self.cfg.aug_degrees, self.cfg.aug_degrees), translate=self.cfg.aug_translate,
            scale_ranges=self.cfg.aug_scale, shears=None,
            img_size=[self.img_size, self.img_size]
        )

        aug_a = TF.affine(img_a, *params, interpolation=transforms.InterpolationMode.BILINEAR, fill=0)
        aug_b = TF.affine(img_b, *params, interpolation=transforms.InterpolationMode.BILINEAR, fill=0)

        # 隨機遮罩
        if random.random() < self.cfg.aug_mask_prob:
             # 簡單實作：隨機遮蔽一塊
             mask_size = int(self.img_size * 0.4)
             mx = random.randint(0, self.img_size - mask_size)
             my = random.randint(0, self.img_size - mask_size)
             aug_a[:, my:my+mask_size, mx:mx+mask_size] = 0
             aug_b[:, my:my+mask_size, mx:mx+mask_size] = 0

        if not self.invert:
            return 1.0 - aug_a, 1.0 - aug_b, label
        return aug_a, aug_b, label

class DynamicGenerator(nn.Module):
    def __init__(self, config):
        super(DynamicGenerator, self).__init__()
        self.nz = config.nz
        self.bottleneck_size = config.bottleneck_size

        # 計算需要縮小幾次 (Stride=2)
        needed_downsamples = int(math.log2(config.image_size / config.bottleneck_size))

        # --- 1. Content Encoder (source_encoder) ---
        layers = []
        in_ch, out_ch = config.nc, config.ngf

        for i in range(config.fixed_layers):
            is_last = (i == config.fixed_layers - 1)

            # 優先進行下採樣，次數夠了就改用 stride=1 保持特徵圖大小
            if i < needed_downsamples:
                # 下採樣層 (使用 cfg 的 k=6, p=2)
                k, s, p = config.kernel_size, 2, config.padding
            else:
                # 保持層 (使用標準 ResNet 設定 k=3, s=1, p=1 以維持穩定)
                k, s, p = 3, 1, 1

            layers.append(nn.Conv2d(in_ch, out_ch, k, s, p, bias=False))
            if not is_last:
                layers.append(nn.BatchNorm2d(out_ch))
                layers.append(nn.LeakyReLU(0.2, inplace=True))

            in_ch = out_ch
            if i < needed_downsamples: out_ch = min(out_ch * 2, 512)

        self.source_encoder = nn.Sequential(*layers) # <--- 命名回歸
        self.enc_out_ch = in_ch

        # --- 2. Decoder (main_branch) ---
        layers = []
        dec_in_ch = self.enc_out_ch + config.nz
        num_flat_layers = config.fixed_layers - needed_downsamples

        for i in range(config.fixed_layers):
            is_last = (i == config.fixed_layers - 1)
            is_flat = i < num_flat_layers # 先做保持層，最後才上採樣

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
                layers.append(nn.BatchNorm2d(dec_out_ch)); layers.append(nn.ReLU(True))
                dec_in_ch = dec_out_ch
            else:
                layers.append(nn.Sigmoid())

        self.main_branch = nn.Sequential(*layers) # <--- 命名回歸

    def forward(self, noise, src):
        # 1. 提取特徵
        feat = self.source_encoder(src)

        # 2. 處理噪聲
        # if noise.dim() == 2: noise = noise.view(noise.size(0), noise.size(1), 1, 1)
        # noise_exp = noise.expand(-1, -1, self.bottleneck_size, self.bottleneck_size)
        noise_exp = noise
        # 3. 合併
        combined = torch.cat((feat, noise_exp), dim=1)

        # 4. 生成 (確保不用再手動 interpolate)
        return self.main_branch(combined)
    

def plot_gan_results(source_imgs, target_imgs, fake_imgs, labels, epoch, config, title_suffix):
    """
    繪製 GAN 生成結果 (適配 Config 物件)
    """
    # 確保數據在 CPU 上並轉為 numpy，且去掉多餘維度
    source_imgs = source_imgs.detach().cpu()
    target_imgs = target_imgs.detach().cpu()
    fake_imgs = fake_imgs.detach().cpu()

    n_samples = source_imgs.size(0)
    # 防呆：如果只有一張圖
    if n_samples == 1:
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        axes = axes[np.newaxis, :] # 增加維度以便下面統一處理
        # 這裡需要手動調整一下結構，但為了通用性，建議測試時至少 batch=2
    else:
        fig, axes = plt.subplots(3, n_samples, figsize=(2 * n_samples, 6))

    main_title = (f"{title_suffix} | Epoch {epoch}/{config.num_epochs}\n"
                  f"n={config.bottleneck_size}, L1_w={config.lambda_l1}\n"
                  f"lr={config.lr}, bs={config.batch_size}")

    for j in range(n_samples):
        # 處理 labels (如果是 Tensor 就轉文字，如果是 list 就直接用)
        lbl = labels[j].item() if isinstance(labels, torch.Tensor) else labels[j]

        # 來源圖像
        ax_src = axes[0, j] if n_samples > 1 else axes[0]
        ax_src.imshow(source_imgs[j].squeeze(), cmap='gray')
        ax_src.set_title(f"Source ({lbl})", fontsize=8)
        ax_src.axis('off')

        # 目標圖像
        ax_tgt = axes[1, j] if n_samples > 1 else axes[1]
        ax_tgt.imshow(target_imgs[j].squeeze(), cmap='gray')
        ax_tgt.set_title(f"Target ({lbl})", fontsize=8)
        ax_tgt.axis('off')

        # 生成圖像
        ax_gen = axes[2, j] if n_samples > 1 else axes[2]
        ax_gen.imshow(fake_imgs[j].squeeze(), cmap='gray')
        ax_gen.set_title(f"Generated", fontsize=8)
        ax_gen.axis('off')

    plt.suptitle(main_title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.9])
    plt.show()

def preprocess_char(dataset, char_str):
    """
    統一的前處理邏輯：確保測試時的輸入跟訓練時完全一樣 (包含反色邏輯)
    """
    # 1. 取得基礎渲染圖 (0=字, 1=背景)
    # 注意：這裡假設 _render_char_to_tensor 回傳的是 PIL 預設 (白底黑字 render 出來 0=黑色字)
    rendered = dataset._render_char_to_tensor(char_str, dataset.font_a)

    # 2. 模擬 Dataset.__getitem__ 的邏輯
    # 在 Dataset 中，第一步是 img_a = 1.0 - img_a_base (變成 1=字, 0=背景)
    img_tensor = 1.0 - rendered

    # 3. 處理反色配置
    if not dataset.invert:
        # 如果設定為白底黑字 (invert=False)，需要再反轉一次回來
        # 變成 0=字 (黑), 1=背景 (白)
        img_tensor = 1.0 - img_tensor

    # 如果 dataset.invert=True，則保持步驟 2 的結果 (1=字, 0=背景)

    return img_tensor

def test_specific_chars(G, dataset, config, epoch, chars_to_test=None):
    if chars_to_test is None:
        # chars_to_test = ["R", "G", "E", "Z", "A", "3","B","H", "7"]
        chars_to_test = ["a","t","R", "G", "E", "Z", "A", "3","B","H","U", "7"]


    # G.eval() # 切換到評估模式 (影響 BatchNorm)
    source_list = []
    target_list = []

    # 使用與訓練一致的處理邏輯
    for char in chars_to_test:
        # 處理來源圖 (Input)
        img_a = preprocess_char(dataset, char)
        source_list.append(img_a)

        # 處理目標圖 (Target) - 僅供畫圖參考用
        # 注意：我們通常想看目標圖的樣子，所以也要用同樣的字元去 render font_b
        # 但這裡其實 font_b 的處理邏輯要參考 font_a，通常是一樣的
        # 我們直接借用 preprocess_char，只是要把內部的 font 改成 font_b
        # 為了方便，我們手動做一次 font_b 的渲染
        raw_b = dataset._render_char_to_tensor(char, dataset.font_b)
        img_b = 1.0 - raw_b
        if not dataset.invert:
            img_b = 1.0 - img_b
        target_list.append(img_b)

    # 堆疊成 Batch 並送入 GPU
    src_batch = torch.stack(source_list).to(device)
    tgt_batch = torch.stack(target_list).to(device)

    # 生成
    with torch.no_grad():
        z = torch.randn(len(chars_to_test), config.nz, config.bottleneck_size, config.bottleneck_size).to(device)
        fake_batch = G(z, src_batch)

    # 呼叫繪圖函數
    plot_gan_results(src_batch, tgt_batch, fake_batch, chars_to_test,
                     epoch, config, "Specific Test Characters")

    G.train() # 記得切回訓練模式


def preview_dataset_samples(dataset, n=6):
    """
    預覽數據集：顯示 Source, Target 以及兩者的疊加圖 (Overlay)
    Overlay 有助於檢查幾何增強是否同步應用在成對圖像上。
    """
    # 建立 3 行 n 列的圖表 (Source, Target, Overlay)
    fig, axes = plt.subplots(3, n, figsize=(n * 2.5, 6))

    # 隨機抽樣
    indices = [random.randint(0, len(dataset)-1) for _ in range(n)]

    for i, idx in enumerate(indices):
        # 取得資料
        item = dataset[idx]
        if isinstance(item, tuple) and len(item) == 3:
            src, tgt, label = item
        else:
            # 防呆
            src, tgt = item[0], item[1]
            label = ""

        # 轉為 Numpy
        src_np = src.squeeze().numpy()
        tgt_np = tgt.squeeze().numpy()

        # 1. 第一行：Source (Times)
        axes[0, i].imshow(src_np, cmap='gray')
        axes[0, i].set_title(f"Source", fontsize=9)
        axes[0, i].axis('off')

        # 2. 第二行：Target (Garamond/Handwritten)
        axes[1, i].imshow(tgt_np, cmap='gray')
        axes[1, i].set_title(f"Target", fontsize=9)
        axes[1, i].axis('off')

        # 3. 第三行：Overlay (疊加比較)
        # 底層用 Target (灰色)，上層用 Source (紅色半透明)
        # 這樣可以清楚看到結構對齊的情況
        axes[2, i].imshow(tgt_np, cmap='gray', alpha=1.0)
        axes[2, i].imshow(src_np, cmap='Reds', alpha=0.5) # 使用紅色 colormap 區分
        axes[2, i].set_title(f"Overlay", fontsize=9)
        axes[2, i].axis('off')

    plt.suptitle("Dataset Preview: Source (Top), Target (Mid), Overlay (Bottom)")
    plt.tight_layout()
    plt.show()


from skimage.metrics import structural_similarity as ssim
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model_ssim(model, dataloader, model_cfg, device):
    """
    計算模型在測試集上的平均 SSIM 分數。
    
    Args:
        model: 生成器模型
        dataloader: 測試集的 DataLoader
        model_cfg: 模型的配置 (用來決定 noise 的大小)
        device: GPU/CPU
        
    Returns:
        avg_ssim: 平均 SSIM 分數 (0~1, 越高越好)
    """
    model.eval()
    ssim_scores = []
    
    with torch.no_grad():
        for i, (src_imgs, tgt_imgs, _) in enumerate(dataloader):
            bs = src_imgs.size(0)
            src_imgs = src_imgs.to(device)
            # tgt_imgs 不需要上 GPU，因為 SSIM 在 CPU 上算比較方便
            
            # 1. 生成圖像
            # 依據該模型的 bottleneck_size 生成 noise
            z = torch.randn(bs, model_cfg.nz, model_cfg.bottleneck_size, model_cfg.bottleneck_size).to(device)
            fake_imgs = model(z, src_imgs)
            
            # 2. 轉回 CPU numpy 用於計算 SSIM
            # 假設輸出是 [0, 1] (Sigmoid)，skimage 需要知道 data_range
            fake_np = fake_imgs.cpu().squeeze(1).numpy() # [B, H, W]
            tgt_np = tgt_imgs.squeeze(1).numpy()         # [B, H, W]
            
            # 3. 逐張計算 SSIM
            for j in range(bs):
                # data_range=1.0 表示像素值在 0-1 之間
                score = ssim(tgt_np[j], fake_np[j], data_range=1.0)
                ssim_scores.append(score)
    
    avg_ssim = sum(ssim_scores) / len(ssim_scores)
    return avg_ssim

def evaluate_model_l1(model, dataloader, model_cfg, device):
    """
    計算模型在測試集上的平均 L1 Loss (Pixel-wise Error)。
    
    Args:
        model: 生成器模型
        dataloader: 測試集的 DataLoader (建議關閉 Augmentation)
        model_cfg: 模型的配置
        device: GPU/CPU
        
    Returns:
        avg_l1: 平均 L1 Loss (越低越好)
    """
    model.eval()
    l1_scores = []
    
    # 定義 L1 Loss 函數 (reduction='mean' 會算出該 batch 的平均)
    criterion_l1 = torch.nn.L1Loss()
    
    with torch.no_grad():
        for i, (src_imgs, tgt_imgs, _) in enumerate(dataloader):
            bs = src_imgs.size(0)
            src_imgs = src_imgs.to(device)
            tgt_imgs = tgt_imgs.to(device) # L1 需要在 GPU 上計算比較快
            
            # 1. 生成圖像
            z = torch.randn(bs, model_cfg.nz, model_cfg.bottleneck_size, model_cfg.bottleneck_size).to(device)
            fake_imgs = model(z, src_imgs)
            
            # 2. 計算 L1 Loss
            loss = criterion_l1(fake_imgs, tgt_imgs)
            l1_scores.append(loss.item())
    
    # 計算整個 DataLoader 的平均
    avg_l1 = sum(l1_scores) / len(l1_scores)
    return avg_l1

# =================================================================



class InceptionV3FeatureExtractor(nn.Module):
    """
    包裝 InceptionV3 模型，只提取最後一層的特徵 (2048維)
    """
    def __init__(self):
        super().__init__()
        # 載入預訓練模型
        inception = inception_v3(pretrained=True, transform_input=False)
        # 我們只需要最後一層全連接層之前的特徵 (Mixed_7c 之後, Average Pooling 之前)
        # 但 pytorch-fid 標準做法是取 avgpool 之後的 2048 維向量
        # 這裡我們直接修改 inception 的 forward 邏輯
        self.blocks = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3, inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d,
            inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c, inception.Mixed_6d, inception.Mixed_6e,
            inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c,
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.blocks.eval() # 設定為評估模式

    def forward(self, x):
        # x: [B, 3, 299, 299] (需 resize 到 299)
        # 為了省顯存，我們可以在外部 resize，這裡只做 forward
        x = self.blocks(x)
        return x.view(x.size(0), -1) # Flatten [B, 2048]

def get_activations(images, model, batch_size=50, dims=2048, device='cuda'):
    """
    計算一組圖片的 Inception 特徵
    Args:
        images: list of tensor or huge tensor [N, 3, H, W]
        model: InceptionV3FeatureExtractor
    Returns:
        pred_arr: numpy array [N, 2048]
    """
    model.eval()
    
    # 確保是 tensor
    if isinstance(images, list):
        images = torch.stack(images)
        
    n_samples = images.size(0)
    pred_arr = np.empty((n_samples, dims))

    start_idx = 0
    
    with torch.no_grad():
        while start_idx < n_samples:
            end_idx = min(start_idx + batch_size, n_samples)
            batch = images[start_idx:end_idx].to(device)
            
            # Inception 需要 299x299 的輸入
            batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
            
            # 正規化到 [-1, 1] 如果原本是 [0, 1]
            # Inception 預期輸入大約是 [-1, 1] 範圍
            batch = (batch - 0.5) * 2
            
            pred = model(batch)
            pred_arr[start_idx:end_idx] = pred.cpu().numpy()
            
            start_idx = end_idx

    return pred_arr

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Numpy 實作的 FID 公式計算
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ("fid calculation produces singular product; "
               "adding %s to diagonal of cov estimates") % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.isclose(np.diagonal(covmean).imag, 0, atol=1e-3).all():
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

def evaluate_model_fid(model, dataloader, model_cfg, device, real_stats=None):
    """
    計算模型生成圖片的 FID 分數
    
    Args:
        real_stats: (mu_real, sigma_real) tuple, 如果已經算過真實圖片的統計量就傳入，節省時間
    """
    # 1. 準備 Inception 模型
    inception = InceptionV3FeatureExtractor().to(device)
    
    # 收集真實圖片 (如果沒有預算好的 stats) 與 生成圖片
    real_imgs_list = []
    fake_imgs_list = []
    
    print("正在生成圖片以計算 FID...", end=" ")
    
    with torch.no_grad():
        for i, (src_imgs, tgt_imgs, _) in enumerate(dataloader):
            bs = src_imgs.size(0)
            src_imgs = src_imgs.to(device)
            
            # 生成
            z = torch.randn(bs, model_cfg.nz, model_cfg.bottleneck_size, model_cfg.bottleneck_size).to(device)
            fake_imgs = model(z, src_imgs) # [B, 1, 64, 64] in [0, 1]
            
            # 轉成 3 通道 (Inception 需要 RGB)
            fake_rgb = fake_imgs.repeat(1, 3, 1, 1)
            fake_imgs_list.append(fake_rgb.cpu())
            
            # 如果還沒算過真實圖片的統計量，就收集真實圖片
            if real_stats is None:
                tgt_imgs = tgt_imgs.to(device)
                tgt_rgb = tgt_imgs.repeat(1, 3, 1, 1)
                real_imgs_list.append(tgt_rgb.cpu())
                
            # 為了節省時間，Demo 時可以只取前 1000 張，但正式評估建議用整個測試集
            # if len(fake_imgs_list) * bs > 1000: break 

    fake_imgs_tensor = torch.cat(fake_imgs_list, dim=0)
    print(f"共收集 {fake_imgs_tensor.size(0)} 張生成圖片。")
    
    # 2. 計算生成圖片的統計量 (mu_fake, sigma_fake)
    act_fake = get_activations(fake_imgs_tensor, inception, device=device)
    mu_fake = np.mean(act_fake, axis=0)
    sigma_fake = np.cov(act_fake, rowvar=False)
    
    # 3. 計算真實圖片的統計量 (mu_real, sigma_real)
    if real_stats is None:
        print("正在計算真實圖片的統計特徵...", end=" ")
        real_imgs_tensor = torch.cat(real_imgs_list, dim=0)
        act_real = get_activations(real_imgs_tensor, inception, device=device)
        mu_real = np.mean(act_real, axis=0)
        sigma_real = np.cov(act_real, rowvar=False)
        real_stats = (mu_real, sigma_real)
        print("完成。")
    else:
        mu_real, sigma_real = real_stats

    # 4. 計算 FID
    fid_score = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    
    return fid_score, real_stats

# =================================================================

import lpips

# 初始化 LPIPS 模型 (建議放在迴圈外面，只載入一次)
# net='alex' 是最常用的輕量級版本，效果通常最好
lpips_loss_fn = lpips.LPIPS(net='alex').to(device)

def evaluate_model_lpips(model, dataloader, model_cfg, device):
    """
    計算模型在測試集上的平均 LPIPS 分數 (Perceptual Distance)。
    
    Returns:
        avg_lpips: 平均 LPIPS 分數 (越低越好，0 代表完全一樣)
    """
    model.eval()
    lpips_scores = []
    
    with torch.no_grad():
        for i, (src_imgs, tgt_imgs, _) in enumerate(dataloader):
            bs = src_imgs.size(0)
            src_imgs = src_imgs.to(device)
            tgt_imgs = tgt_imgs.to(device)
            
            # 1. 生成圖像
            z = torch.randn(bs, model_cfg.nz, model_cfg.bottleneck_size, model_cfg.bottleneck_size).to(device)
            fake_imgs = model(z, src_imgs)
            
            # 2. 處理圖像格式給 LPIPS
            # LPIPS 預期輸入範圍是 [-1, 1]
            # 你的模型輸出如果是 Sigmoid [0, 1]，需要轉換
            # 轉換公式: (img - 0.5) * 2
            
            # 假設 fake_imgs 和 tgt_imgs 都是 [0, 1]
            fake_norm = (fake_imgs - 0.5) * 2
            tgt_norm = (tgt_imgs - 0.5) * 2
            
            # 如果是灰階 [B, 1, H, W]，需要轉成 RGB [B, 3, H, W]
            if fake_norm.shape[1] == 1:
                fake_norm = fake_norm.repeat(1, 3, 1, 1)
                tgt_norm = tgt_norm.repeat(1, 3, 1, 1)
            
            # 3. 計算 LPIPS
            # forward 回傳的是一個 batch 的分數 tensor
            batch_scores = lpips_loss_fn(fake_norm, tgt_norm)
            
            # 轉成 list 加入總分
            lpips_scores.extend(batch_scores.cpu().view(-1).tolist())
    
    avg_lpips = sum(lpips_scores) / len(lpips_scores)
    return avg_lpips
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

def visualize_metric_results(results_list):
    """
    通用視覺化評測結果函數。
    自動偵測輸入資料中的指標欄位，並繪製對應的長條圖。
    
    Args:
        results_list: list of dict, e.g. 
                      [{'Model': '1x1', 'SSIM': 0.85, 'L1': 0.12}, ...]
    """
    # 1. 轉成 DataFrame
    df = pd.DataFrame(results_list)
    
    # 2. 顯示表格 (Markdown 格式)
    print("\n=== Evaluation Metrics Table ===")
    try:
        print(df.to_markdown(index=False))
    except:
        print(df) # Fallback if markdown not installed

    # 3. 找出所有指標欄位 (排除 'Model' 欄位)
    metrics = [col for col in df.columns if col != 'Model']
    num_metrics = len(metrics)
    
    if num_metrics == 0:
        print("沒有可繪製的指標數據。")
        return df

    # 4. 設定繪圖版面 (自動計算行數與列數)
    # 例如：3個指標 -> 1行3列
    cols = min(num_metrics, 3) 
    rows = math.ceil(num_metrics / cols)
    
    plt.figure(figsize=(6 * cols, 5 * rows))
    sns.set_style("whitegrid")
    
    # 定義一組專業簡約的色票 (深藍, 深紅, 深綠, 紫...)
    # 讓每個指標有一種專屬的主題色
    palette_map = ['#4C72B0', '#C44E52', '#55A868', '#8172B3', '#CCB974']

    for i, metric in enumerate(metrics):
        plt.subplot(rows, cols, i + 1)
        
        # 選擇該指標的主題色
        color = palette_map[i % len(palette_map)]
        
        # 繪製長條圖 (使用單一顏色，不使用 hue)
        ax = sns.barplot(x="Model", y=metric, data=df, color=color)
        
        # 標示數值
        for p in ax.patches:
            height = p.get_height()
            if height == 0: continue
            
            # 智慧調整標籤位置 (避免數值太小時標籤擠在一起)
            ax.annotate(f'{height:.4f}', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='bottom', 
                        xytext=(0, 5), 
                        textcoords='offset points', 
                        fontsize=10, fontweight='bold', color='black')
            
        plt.title(f"{metric} Comparison", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Bottleneck Size", fontsize=11)
        plt.ylabel(f"Average {metric}", fontsize=11)
        
        # 微調 Y 軸範圍，讓標籤不會被切到
        # 對於 SSIM 這種 0-1 的指標，固定上限比較好看；其他的自動調整
        if metric.upper() == 'SSIM':
            plt.ylim(0, 1.1)
        else:
            # 自動擴展 15% 空間給標籤
            current_max = df[metric].max()
            plt.ylim(0, current_max * 1.15)

    plt.tight_layout()
    plt.savefig("metrics_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return df