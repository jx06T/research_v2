import os
import argparse
import gdown
import yaml
import shutil
import csv
import glob

REGISTRY_FILE = "model_registry.csv"

def parse_yaml(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        # 因 config 中可能包含 !!python/tuple 等特定標籤，故需使用 UnsafeLoader
        return yaml.load(f, Loader=yaml.UnsafeLoader)

def update_registry(run_id, config, exp_name, arch_folder):
    fieldnames = ['run_id', 'experiment', 'arch_folder', 'image_size', 'bottleneck_size', 'nc', 'nz', 'lambda_l1', 'batch_size', 'src_font', 'tgt_font']
    
    file_exists = os.path.exists(REGISTRY_FILE)
    existing_runs = set()
    
    if file_exists:
        with open(REGISTRY_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_runs.add(row['run_id'])
                
    if run_id in existing_runs:
        return # Already registered

    with open(REGISTRY_FILE, 'a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
            
        writer.writerow({
            'run_id': run_id,
            'experiment': exp_name,
            'arch_folder': arch_folder,
            'image_size': config.get('image_size', 'N/A'),
            'bottleneck_size': config.get('bottleneck_size', 'N/A'),
            'nc': config.get('nc', 'N/A'),
            'nz': config.get('nz', 'N/A'),
            'batch_size': config.get('batch_size', 'N/A'),
            'src_font': config.get('src_font', 'N/A'),
            'tgt_font': config.get('tgt_font', 'N/A')
        })
    print(f"    [Registry] 已將 {run_id} 登錄至 {REGISTRY_FILE}")

def main():
    parser = argparse.ArgumentParser(description="Google Drive 智能歸檔與模型註冊工具")
    args = parser.parse_args()

    # 格式: {"實驗名稱": "Google Drive 資料夾 ID"}
    DRIVE_FOLDERS = {
        'a3': '12h4SSeCxbwxWME9hbdJ_Yv1pCXk-UabZ'
    }

    print("啟動 Google Drive 智能歸檔與分類工具")
    print("--------------------------------------------------")
    
    for exp_name, folder_id in DRIVE_FOLDERS.items():
        if folder_id == "這裡填入資料夾的ID":
            print(f"⚠️ 請先將 {exp_name} 的真實資料夾 ID 填入腳本中！")
            continue

        local_output = os.path.join("runs", exp_name)
        os.makedirs(local_output, exist_ok=True)
        
        print(f"正在同步 [{exp_name}] (ID: {folder_id}) ...")
        
        # 1. 下載更新 (gdown 會自動略過已存在的檔案)
        # 用 try-except 包起來，就算 gdown 因為病毒掃描大檔案失敗，我們也要繼續往下走
        try:
            gdown.download_folder(id=folder_id, output=local_output, quiet=True, use_cookies=False)
            print(f"✅ [{exp_name}] 下載完成，開始執行智能分類與註冊...")
        except Exception as e:
            print(f"⚠️ [{exp_name}] gdown 下載過程中發生錯誤 (可能是遇到舊版巨型檔案限制): {e}")
            print(f"    但將會繼續為本地已有的檔案進行歸檔與註冊...")
            
        try:
            # 2. 尋找所有 yaml 設定檔
            yaml_files = glob.glob(os.path.join(local_output, "config_*.yaml"))
            
            for yaml_path in yaml_files:
                filename = os.path.basename(yaml_path)
                # 檔名格式: config_{run_id}.yaml
                run_id = filename.replace("config_", "").replace(".yaml", "")
                
                config = parse_yaml(yaml_path)
                img_size = config.get('image_size', 64)
                btn_size = config.get('bottleneck_size', 4)
                
                arch_folder_name = f"arch_img{img_size}_btn{btn_size}"
                arch_folder_path = os.path.join("runs", arch_folder_name)
                os.makedirs(arch_folder_path, exist_ok=True)
                
                # 3. 複製 yaml 到架構資料夾
                dest_yaml = os.path.join(arch_folder_path, filename)
                if not os.path.exists(dest_yaml):
                    shutil.copy2(yaml_path, dest_yaml)
                
                # 4. 尋找屬於這個 run_id 的 G_*.pth
                pth_files = glob.glob(os.path.join(local_output, f"G_*{run_id}*.pth"))
                for pth_path in pth_files:
                    pth_filename = os.path.basename(pth_path)
                    dest_pth = os.path.join(arch_folder_path, pth_filename)
                    if not os.path.exists(dest_pth):
                        shutil.copy2(pth_path, dest_pth)
                        print(f"    [Archived] {pth_filename} -> {arch_folder_name}/")
                        
                # 5. 更新註冊表
                update_registry(run_id, config, exp_name, arch_folder_name)
                
        except Exception as e:
            print(f"❌ [{exp_name}] 處理失敗: {e}")

    print("--------------------------------------------------")
    print("所有同步與歸檔作業已完成！")

if __name__ == "__main__":
    main()
