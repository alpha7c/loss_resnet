import os
import subprocess

config_prefix = "temp_cfg/"


def count_files_scandir(directory):
    return sum(1 for entry in os.scandir(directory) if entry.is_file())
    
    
num_configs = count_files_scandir(config_prefix)

print(f"File Number (scandir): {num_configs}")


for i in range(num_configs):
    config_file = f"{config_prefix}{i}.json"
    command = f"python3.8 online_expr.py -c {config_file}"
    print(f"Executing: {command}")
    subprocess.run(command, shell=True)