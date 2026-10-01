import os
import yaml
import shutil
import csv
import glob

REGISTRY_FILE = "model_registry.csv"
EXPERIMENTS = ['a1', 'a3']

def parse_yaml(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.load(f, Loader=yaml.UnsafeLoader)

def update_registry(run_id, config, exp_name, arch_folder):
    fieldnames = ['run_id', 'experiment', 'arch_folder', 'image_size', 'bottleneck_size', 'nc', 'nz', 'lambda_l1', 'batch_size', 'src_font', 'tgt_font', 'gen_type']
    
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
            'tgt_font': config.get('tgt_font', 'N/A'),
            'gen_type': config.get('gen_type', 'dynamic')
        })
    print(f"    [Registry] 已將 {run_id} 登錄至 {REGISTRY_FILE}")

def main():
    print("啟動本地歸檔與註冊腳本 (Organize Runs)")
    print("--------------------------------------------------")
    
    for exp_name in EXPERIMENTS:
        local_output = os.path.join("runs", exp_name)
        if not os.path.exists(local_output):
            continue
            
        print(f"正在整理 [{exp_name}] ...")
        
        try:
            # 1. 尋找所有 yaml 設定檔
            yaml_files = glob.glob(os.path.join(local_output, "config_*.yaml"))
            
            for yaml_path in yaml_files:
                filename = os.path.basename(yaml_path)
                run_id = filename.replace("config_", "").replace(".yaml", "")
                
                config = parse_yaml(yaml_path)
                img_size = config.get('image_size', 64)
                btn_size = config.get('bottleneck_size', 4)
                
                arch_folder_name = f"arch_img{img_size}_btn{btn_size}"
                arch_folder_path = os.path.join("runs", arch_folder_name)
                os.makedirs(arch_folder_path, exist_ok=True)
                
                # 2. 複製 yaml 到架構資料夾
                dest_yaml = os.path.join(arch_folder_path, filename)
                if not os.path.exists(dest_yaml):
                    shutil.copy2(yaml_path, dest_yaml)
                
                # 3. 尋找屬於這個 run_id 的 G_*.pth
                pth_files = glob.glob(os.path.join(local_output, f"G_*{run_id}*.pth"))
                for pth_path in pth_files:
                    pth_filename = os.path.basename(pth_path)
                    dest_pth = os.path.join(arch_folder_path, pth_filename)
                    if not os.path.exists(dest_pth):
                        shutil.copy2(pth_path, dest_pth)
                        print(f"    [Archived] {pth_filename} -> {arch_folder_name}/")
                        
                # 4. 更新註冊表
                update_registry(run_id, config, exp_name, arch_folder_name)
                
        except Exception as e:
            print(f"❌ [{exp_name}] 整理失敗: {e}")

    print("--------------------------------------------------")
    print("本地歸檔與註冊作業已完成！")

if __name__ == "__main__":
    main()
