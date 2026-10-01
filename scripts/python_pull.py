import subprocess
import time
import os

REMOTE_HOST = "Colab-WGAN"
REMOTE_BASE_DIR = "/content/drive/MyDrive/research_v2/runs"
EXPERIMENTS = ['a1', 'a3']

def get_remote_files(exp_name):
    try:
        # 取得遠端該實驗資料夾下所有的 .pth 和 .yaml
        cmd = f"ls {REMOTE_BASE_DIR}/{exp_name}/G_*.pth {REMOTE_BASE_DIR}/{exp_name}/config_*.yaml 2>/dev/null"
        result = subprocess.run(
            ["ssh", REMOTE_HOST, cmd],
            capture_output=True, text=True, check=True
        )
        return [line.strip().split('/')[-1] for line in result.stdout.split('\n') if line.strip()]
    except subprocess.CalledProcessError:
        return []

def download_file(exp_name, filename, local_output_dir):
    remote_path = f"{REMOTE_BASE_DIR}/{exp_name}/{filename}"
    local_path = os.path.join(local_output_dir, filename)
    
    with open(local_path, "wb") as f:
        result = subprocess.run(
            ["ssh", REMOTE_HOST, f"cat {remote_path}"],
            stdout=f, stderr=subprocess.DEVNULL
        )
    return result.returncode == 0, local_path

def process_experiment(exp_name, timestamp):
    local_output = os.path.join("runs", exp_name)
    os.makedirs(local_output, exist_ok=True)
    
    remote_files = get_remote_files(exp_name)
    if not remote_files:
        return 0

    new_downloads = 0
    
    # 1. 先下載所有的 config_*.yaml
    yaml_files = [f for f in remote_files if f.endswith('.yaml')]
    for fname in yaml_files:
        local_path = os.path.join(local_output, fname)
        if not os.path.exists(local_path):
            print(f"[{timestamp}] [{exp_name}] 發現新設定檔: {fname}，正在下載...")
            success, _ = download_file(exp_name, fname, local_output)
            if success:
                new_downloads += 1

    # 2. 下載所有的 G_*.pth
    pth_files = [f for f in remote_files if f.endswith('.pth')]
    for fname in pth_files:
        local_path = os.path.join(local_output, fname)
        if "latest" in fname or not os.path.exists(local_path):
            if not "latest" in fname:
                print(f"[{timestamp}] [{exp_name}] 發現新權重檔: {fname}，正在經由 SSH 下載 (無檔案大小限制)...")
            success, _ = download_file(exp_name, fname, local_output)
            if success:
                if not "latest" in fname:
                    print(f"[{timestamp}] [{exp_name}] {fname} 下載完成！")
                new_downloads += 1

    return new_downloads

def main():
    print(f"啟動 SSH 自動同步腳本 (Bypass gdown limits)")
    print(f"遠端主機: {REMOTE_HOST}")
    print("按下 Ctrl+C 終止腳本。")
    print("-" * 50)

    cycle = 1
    while True:
        timestamp = time.strftime("%H:%M:%S")
        total_downloads = 0
        
        for exp in EXPERIMENTS:
            total_downloads += process_experiment(exp, timestamp)

        if total_downloads > 0:
            print(f"[{timestamp}] [Cycle #{cycle}] 同步完成，更新了 {total_downloads} 個檔案。開始執行本地歸檔與註冊...")
            subprocess.run(["python", "scripts/organize_runs.py"])
        else:
            print(f"[{timestamp}] [Cycle #{cycle}] 檢查中... (遠端無新檔案)")

        cycle += 1
        time.sleep(300)

if __name__ == "__main__":
    main()
