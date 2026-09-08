import subprocess
import time
import os

REMOTE_HOST = "Colab-WGAN"
REMOTE_DIR = "/content/drive/MyDrive/research_v2/runs"
LOCAL_DIR = "runs/exp_01"

def get_remote_files():
    try:
        # 取得遠端所有 G_*.pth 的檔案路徑清單
        result = subprocess.run(
            ["ssh", REMOTE_HOST, f"ls {REMOTE_DIR}/G_*.pth"],
            capture_output=True, text=True, check=True
        )
        # 過濾空字串並返回檔案名稱
        return [line.strip().split('/')[-1] for line in result.stdout.split('\n') if line.strip()]
    except subprocess.CalledProcessError:
        return []

def download_file(filename):
    remote_path = f"{REMOTE_DIR}/{filename}"
    local_path = os.path.join(LOCAL_DIR, filename)
    
    with open(local_path, "wb") as f:
        result = subprocess.run(
            ["ssh", REMOTE_HOST, f"cat {remote_path}"],
            stdout=f, stderr=subprocess.DEVNULL
        )
    return result.returncode == 0, local_path

def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    print(f"啟動自動同步腳本 (Auto-Pull Daemon)")
    print(f"遠端主機: {REMOTE_HOST}")
    print(f"將自動下載所有遠端的 G_*.pth 推論權重檔，忽略肥大的 checkpoint。")
    print(f"本地儲存至: {LOCAL_DIR}/")
    print("按下 Ctrl+C 終止腳本。")
    print("-" * 50)

    cycle = 1
    while True:
        timestamp = time.strftime("%H:%M:%S")
        remote_files = get_remote_files()
        
        new_downloads = 0
        for fname in remote_files:
            local_path = os.path.join(LOCAL_DIR, fname)
            
            # 如果是 G_latest 或者是本地還沒有的 epoch 檔，才抓取
            if "latest" in fname or not os.path.exists(local_path):
                # 只有最新版或未下載的會觸發下載
                if not "latest" in fname:
                    print(f"[{timestamp}] 發現新的權重檔: {fname}，正在下載...")
                
                success, downloaded_path = download_file(fname)
                if success:
                    if not "latest" in fname:
                        print(f"[{timestamp}] {fname} 下載完成！")
                    new_downloads += 1

        if new_downloads > 0:
            print(f"[{timestamp}] [Cycle #{cycle}] 同步完成，更新了 {new_downloads} 個檔案。")
        else:
            print(f"[{timestamp}] [Cycle #{cycle}] 檢查中... (遠端無新檔案)")

        cycle += 1
        time.sleep(300) # 每 5 分鐘檢查一次

if __name__ == "__main__":
    main()
