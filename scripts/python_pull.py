import subprocess
import time
import os

remote_host = "Colab-WGAN"
remote_file = "/content/drive/MyDrive/research_v2/runs/G_latest.pth"
local_file = "runs/exp_01/G_latest.pth"

os.makedirs("runs/exp_01", exist_ok=True)
print(f"Starting auto-pull from {remote_host}...")
print(f"Target: {local_file}")
print("Press Ctrl+C to stop.")

cycle = 1
while True:
    timestamp = time.strftime("%H:%M:%S")
    try:
        with open(local_file, "wb") as f:
            result = subprocess.run(
                ["ssh", remote_host, f"cat {remote_file}"], 
                stdout=f, stderr=subprocess.DEVNULL
            )
        
        if result.returncode == 0:
            file_size = os.path.getsize(local_file)
            if file_size > 0:
                print(f"[{timestamp}] [Cycle #{cycle}] Successfully pulled {file_size} bytes.")
    except Exception as e:
        pass
    cycle += 1
    time.sleep(300)
