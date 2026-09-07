# Font GAN - 中文字體風格轉換 (v2)

本專案是基於 PyTorch 實作的字體風格轉換 GAN 模型。在 v2 架構中，我們捨棄了龐大的單一 Jupyter Notebook 開發模式，改為**模組化架構**，並高度整合 **Colab + Cloudflare Tunnel + VS Code** 的遠端開發流程。

---

## 📂 專案結構 (Project Structure)

```text
research_v2/
├── configs/                 # 設定檔放置處
│   └── default.yaml         # 模型與訓練超參數配置 (整合了 v7 最新參數)
├── data/                    # 靜態資源與資料
│   ├── corpus/              # 語料庫 Markdown 檔案
│   └── fonts/               # 存放作為 Source 與 Target 的字體檔案 (.ttf, .otf)
├── src/                     # 核心原始碼模組
│   ├── models/              
│   │   └── mm_legacy.py     # 網路架構定義 (Generator, Discriminator 等)
│   ├── data/                
│   │   └── font_dataset.py  # PyTorch Dataset 實作，包含中文字元集與擴增邏輯
│   ├── utils/               
│   │   └── font_checker.py  # 字體缺字檢測工具
│   └── train.py             # 核心訓練腳本
├── scripts/                 
│   └── setup_colab.sh       # Colab 初始化與 SSH 連線建立腳本
├── demo_qt.py               # PyQt6 圖形化展示介面 (本地端執行)
└── requirements.txt         # 依賴套件清單
```

---

## 🚀 環境建置與連線步驟 (Setup & Connection)

本專案建議在 **Google Colab (提供免費 GPU)** 運行訓練，並透過 **VS Code** 進行遠端編寫與控制。

### 第一步：設定 Colab 端
1. 開啟一個全新的 Google Colab 筆記本，將執行階段更改為 **T4 GPU**。
2. 建立一個新的程式碼區塊，將 `scripts/setup_colab.sh` 內的指令貼上。
   *(或者，直接在區塊執行以下 Python 程式碼)*：
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   !pip install colab-ssh --upgrade -q
   !pip install -r /content/drive/MyDrive/research_v2/requirements.txt -q
   !pip install pyyaml

   from colab_ssh import launch_ssh_cloudflared
   launch_ssh_cloudflared("請輸入您的自訂密碼")
   ```
3. 執行後，等待畫面顯示 `Cloudflare Tunnel` 的連線資訊（一組以 `.trycloudflare.com` 結尾的網址）。

### 第二步：設定本地 VS Code
1. 本地電腦請確保已安裝 `cloudflared` 工具以及 VS Code 的 `Remote - SSH` 擴充套件。
2. 開啟 VS Code 左下角的 `><` 遠端圖示，點擊 **Connect to Host**。
3. 選擇上面在 Colab 獲取的網址，輸入您剛才自訂的密碼。
4. 連線成功後，點擊「開啟資料夾」，輸入 `/content/drive/MyDrive/research_v2` (依您的實際存放路徑而定)，即可開始開發！

---

## 🏃 執行訓練 (Training)

所有的超參數（如 `lambda_l1`, `n_critic`, `batch_size`, `num_epochs`）皆已移至 `configs/default.yaml` 集中管理。
若要開始訓練，請在遠端 VS Code 終端機內執行：

```bash
python src/train.py \
    --config configs/default.yaml \
    --src_font data/fonts/NotoSansTC-Regular.ttf \
    --tgt_font data/fonts/kaiu.ttf \
    --output_dir /content/drive/MyDrive/research_v2_runs/exp_01 \
    --save_interval 10 \
    --keep_checkpoints 3
```
*(字體亦可替換為 `data/fonts/tegaki_zatsu.ttf`)*

### 💡 防硬碟爆滿與輕量化儲存設計
1. **推論權重輕量化**：訓練過程中**只會儲存推論用的 Generator 權重** (`G.state_dict()`)，捨棄了龐大的 Optimizer 與 Discriminator 狀態，單個檔案僅約 15~25MB。
2. **滾動保留 Checkpoints (`--keep_checkpoints 3`)**：自動刪除過舊的歷史輪次，硬碟最多只保留最新 3 份 checkpoints。
3. **即時最新檔 (`G_latest.pth`)**：每輪儲存時自動更新覆蓋 `G_latest.pth`。

---

## 🔄 本地自動同步腳本 (Auto-Pull Daemon)

若要在本機自動接收遠端產生的模型權重，免手動下載：
1. 在本地端 VS Code 開啟一個本機 PowerShell 終端機。
2. 執行自動拉取腳本：
   ```powershell
   .\scripts\auto_pull.ps1 -RemoteHost "colab" -RemoteDir "/content/drive/MyDrive/research_v2_runs/exp_01" -LocalDir "runs/exp_01"
   ```
3. 腳本會在背景每 60 秒自動透過 `scp` 抓取最新 `.pth` 權重至本地端！

---

## 🎨 預覽結果 (GUI Demo)

訓練完成後，我們可以在本地端電腦上使用 `demo_qt.py` 來查看生成成果：
1. 確保本地端裝有 `PyQt6`、`torch` 與 `Pillow`。
2. 修改 `demo_qt.py` 內的 `EXPERIMENTS` 字典，將 `model_path` 指向您剛剛訓練產生的 Generator 權重檔 (.pth)。
3. 在終端機執行：
   ```bash
   python demo_qt.py
   ```
4. 您將能透過視窗介面，即時拉動滑桿預覽來源字型轉換為目標字型的生成過程！
