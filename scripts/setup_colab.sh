#!/bin/bash
# ==============================================================================
# Google Colab 環境初始化與 Cloudflare SSH 隧道啟動腳本
# ==============================================================================

echo "=== [1/3] 掛載 Google 雲端硬碟 (用於保存模型權重與備份) ==="
python3 -c "from google.colab import drive; drive.mount('/content/drive')"

echo "=== [2/3] 安裝必要依賴套件 ==="
pip install colab-ssh --upgrade -q
# 若已 clone 專案到本地 /content/research_v2
if [ -f "/content/research_v2/requirements.txt" ]; then
    pip install -r /content/research_v2/requirements.txt -q
elif [ -f "/content/drive/MyDrive/research_v2/requirements.txt" ]; then
    pip install -r /content/drive/MyDrive/research_v2/requirements.txt -q
else
    pip install torch torchvision pillow pyyaml matplotlib -q
fi

echo "=== [3/3] 啟動 Cloudflare SSH Tunnel ==="
# 提示：請將密碼替換為您的個人自訂密碼
python3 -c "
from colab_ssh import launch_ssh_cloudflared
password = 'your_secure_password'  # 請替換為您的連線密碼
print(f'正在啟動 Cloudflare SSH Tunnel (密碼: {password})...')
launch_ssh_cloudflared(password)
"

echo "=============================================================================="
echo "連線初始化完成！請複製上方產生的 VS Code SSH 設定貼入本地端 ~/.ssh/config 即可連線。"
echo "=============================================================================="
