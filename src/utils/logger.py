import time
import random
import string
import os

import yaml

class ExperimentLogger:
    def __init__(self, config, output_dir="./runs"):
        self.config = config
        self.output_dir = output_dir
        self.run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
        self.history = {"G_loss": [], "D_loss": [], "L1_loss": [], "loss_G_adv": []}
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 自動將當次訓練的 config 備份成 yaml 檔，方便追蹤
        config_save_path = os.path.join(self.output_dir, f"config_{self.run_id}.yaml")
        with open(config_save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config.to_dict(), f, allow_unicode=True)
            
        print(f"[{self.run_id}] Logger initialized. Saving to {self.output_dir}")


    def log_metrics(self, loss_G, loss_D, loss_G_l1, loss_G_adv):
        self.history["G_loss"].append(loss_G)
        self.history["D_loss"].append(loss_D)
        self.history["L1_loss"].append(loss_G_l1)
        self.history["loss_G_adv"].append(loss_G_adv)

