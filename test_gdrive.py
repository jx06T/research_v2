import os
import shutil
from scripts.gdrive_pull import parse_yaml, update_registry
import glob

def test_logic():
    # Setup dummy
    os.makedirs("runs/exp_test", exist_ok=True)
    with open("runs/exp_test/config_abcd.yaml", "w") as f:
        f.write("image_size: 128\nbottleneck_size: 8\nlambda_l1: 50\n")
    with open("runs/exp_test/G_abcd_epoch_100.pth", "w") as f:
        f.write("dummy pth")
        
    local_output = "runs/exp_test"
    exp_name = "exp_test"
    
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
        
        dest_yaml = os.path.join(arch_folder_path, filename)
        shutil.copy2(yaml_path, dest_yaml)
        
        pth_files = glob.glob(os.path.join(local_output, f"G_*{run_id}*.pth"))
        for pth_path in pth_files:
            pth_filename = os.path.basename(pth_path)
            dest_pth = os.path.join(arch_folder_path, pth_filename)
            shutil.copy2(pth_path, dest_pth)
                
        update_registry(run_id, config, exp_name, arch_folder_name)
        
    print("Test finished")

if __name__ == "__main__":
    test_logic()
