# read_binary_files_data.py
import pickle
import numpy as np
import torch

def read_data_file(file_path):
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        print("Keys in the file:", list(data.keys()))
        print("=" * 50)

        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                arr = v.cpu().numpy()
            elif isinstance(v, np.ndarray):
                arr = v
            else:
                try:
                    arr = np.array(v)
                except Exception:
                    print(f"{k}: type={type(v)} (not convertible to ndarray)")
                    continue
            print(f"{k}: shape {arr.shape}, dtype {arr.dtype}")

        # Preview accuracies
        if "accuracies" in data:
            acc = (
                data["accuracies"].cpu().numpy()
                if isinstance(data["accuracies"], torch.Tensor)
                else np.array(data["accuracies"])
            )
            print("\nFirst 10 accuracy values:")
            for i in range(min(10, len(acc))):
                print(f"Task {i}: {acc[i]:.6f}")

        # Stats
        print("\nData statistics:")
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                arr = v.cpu().numpy()
            elif isinstance(v, np.ndarray):
                arr = v
            else:
                continue
            if arr.ndim == 1 and arr.size > 0 and np.issubdtype(arr.dtype, np.number):
                print(
                    f"{k}: min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f}"
                )

    except FileNotFoundError:
        print(f"Error: file {file_path} not found")
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    file_path = "/home/wujy/PL_NC/lop/permuted_mnist/data/bp/std_net/0/0"
    read_data_file(file_path)
