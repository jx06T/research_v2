# 字體生成對抗網路 (Font Generation WGAN-GP) 專案架構說明

本專案採用了高度模組化且具備 MLOps 擴展性的架構設計，旨在提供穩定、可複現、且能輕易擴展至不同實驗環境的 WGAN-GP 字體生成訓練流程。

## 📁 目錄結構與分工

```text
research_v2/
├── configs/                  # 統一超參數配置中心
│   ├── default.yaml          # 預設配置 (n=4, l1=1)
│   └── l1_100_n8.yaml        # 強效配置 (n=8, l1=100)
├── data/
│   └── fonts/                # 存放來源與目標 TTF 字體檔
├── runs/                     # 訓練產出物 (由 Logger 自動生成)
│   └── [Run_ID]/             # 包含權重檔、備份的 yaml 配置等
├── scripts/
│   ├── python_pull.py        # 解決 Colab SSH 隧道無法 SCP 問題的自動下載腳本
│   └── auto_pull.ps1         # PowerShell 版本的自動下載腳本
├── src/                      # 核心模組
│   ├── config/               
│   │   └── config_parser.py  # 將 yaml 轉為嚴格的 Dataclass 物件，支援點語法 (config.image_size)
│   ├── data/                 
│   │   ├── builder.py        # DataLoader 工廠，動態配置 num_workers 以最佳化 CPU
│   │   └── font_dataset.py   # 即時渲染 TTF 並進行資料幾何增強的 Dataset
│   ├── models/               
│   │   ├── discriminator/    
│   │   │   └── patch_gan.py  # 判別器 (PatchGAN 架構)
│   │   ├── generator/        
│   │   │   └── dynamic_gen.py# 生成器 (動態深度 U-Net 架構)
│   │   └── loss.py           # 損失函數集合 (含 Gradient Penalty 計算)
│   ├── utils/                
│   │   ├── logger.py         # 實驗追蹤器 (生成 RUN_ID、備份設定、儲存權重)
│   │   └── visualize.py      # 統一管理所有 Matplotlib 視覺化與繪圖
│   ├── export/               
│   │   └── font_builder.py   # (預留) 未來將生成結果向量化並打包回 TTF 的腳本
│   └── train.py              # 乾淨的訓練主流程 (僅負責流程控制)
├── demo_qt.py                # 本地端 PyQt 介面測試工具
└── requirements.txt          # Python 相依套件清單
```

## 🛠️ 核心機制說明

### 1. 實驗配置隔離 (Config & RUN_ID)
為了支援在不同帳號或機器同時測試不同配置：
- 所有超參數皆由 `configs/*.yaml` 獨立管理。
- 程式啟動時會產生一組獨一無二的 `RUN_ID` (例如 `20260908_1423_a1b2`)。
- 訓練當下的配置會被自動備份成 `config_[RUN_ID].yaml`。
- 所有的輸出檔案都會帶有這個 ID，完美防止不同實驗互相覆蓋。

### 2. 模型權重雙軌制 (Inference vs Checkpoint)
每經過設定的保存間隔（預設 100 輪），會產生兩種不同的檔案：
1. **推論輕量檔 (`G_[RUN_ID]_epoch_100.pth`)**
   - 內容：**僅包含 Generator 權重**。
   - 用途：檔案極小，專門供 `python_pull.py` 自動下載，用於本地 `demo_qt.py` 測試與未來推論使用。
   - （每次還會額外覆寫一份 `G_latest.pth` 方便腳本抓取最新版）。
2. **完整訓練狀態 (`checkpoint_[RUN_ID]_epoch_100.pth`)**
   - 內容：包含 Generator、Discriminator、雙 Optimizer 動量、以及完整的 config 字典。
   - 用途：檔案巨大，僅存放在雲端，專門用於 Colab 斷線時的**無縫接續訓練**。

### 3. CPU 效能自動調適 (num_workers)
`PairedFontDataset` 採用即時渲染與幾何轉換，對 CPU 極度消耗。
在 `src/data/builder.py` 中，我們採用了 `min(12, os.cpu_count() or 2)`。
這確保了在 Colab 免費版 (2 核心) 不會因為過載而死機，同時在 Colab Pro (高 RAM 模式) 或本地電腦 (12 核心) 能火力全開加速訓練。

## 🚀 典型開發工作流

1. **遠端訓練**：在 Colab 透過 `%run src/train.py --config configs/l1_100_n8.yaml ...` 啟動訓練。
2. **本地同步**：在本地電腦執行 `python scripts/python_pull.py`，它會在背景每 5 分鐘掃描遠端，並將新的 `G_*.pth` 下載到本地的 `runs/exp_01/`。
3. **本地測試**：開啟 `demo_qt.py`，讀取剛下載的推論權重，實時檢視字體生成效果。
