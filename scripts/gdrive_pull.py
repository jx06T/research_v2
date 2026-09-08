import os
import time
import argparse
import gdown

def main():
    parser = argparse.ArgumentParser(description="從多個 Google Drive 共用資料夾自動下載模型權重")
    parser.add_argument('--interval', type=int, default=300, help='檢查間隔 (秒)')
    args = parser.parse_args()

    # 格式: {"實驗名稱": "Google Drive 資料夾 ID"}
    # 您可以在這裡加入不同帳號、不同實驗的資料夾 ID
    DRIVE_FOLDERS = {
        "exp_default": "這裡填入資料夾的ID", 
        # "exp_large": "另一個帳號的資料夾ID"
    }

    print("啟動 Google Drive 自動下載腳本 (GDown Daemon)")
    print("--------------------------------------------------")
    for exp_name, folder_id in DRIVE_FOLDERS.items():
        print(f"監控目標: [{exp_name}] -> ID: {folder_id}")
    print(f"檢查間隔: {args.interval} 秒")
    print("請確保這些資料夾已在 Google 雲端硬碟設定為「知道連結的任何人皆可檢視」")
    print("按下 Ctrl+C 終止腳本。")
    print("-" * 50)

    cycle = 1
    while True:
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] [Cycle #{cycle}] 開始掃描雲端硬碟...")

        for exp_name, folder_id in DRIVE_FOLDERS.items():
            if folder_id == "這裡填入資料夾的ID":
                print(f"[{timestamp}] ⚠️ 請先將 {exp_name} 的真實資料夾 ID 填入腳本中！")
                continue

            local_output = os.path.join("runs", exp_name)
            os.makedirs(local_output, exist_ok=True)
            
            try:
                # gdown.download_folder 會自動比對本地檔案，只下載有更新的內容
                # 設定 quiet=True 減少洗頻
                gdown.download_folder(id=folder_id, output=local_output, quiet=True, use_cookies=False)
                print(f"[{timestamp}] ✅ [{exp_name}] 同步完成")
            except Exception as e:
                print(f"[{timestamp}] ❌ [{exp_name}] 同步失敗: {e}")

        cycle += 1
        time.sleep(args.interval)

if __name__ == "__main__":
    main()
