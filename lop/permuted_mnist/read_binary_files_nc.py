# read_nc_file.py
import pickle
import os
import sys
from typing import List, Tuple

def read_incremental_records(path: str) -> List[Tuple[int, float, float, float, float]]:
    """
    Read a file saved by multiple pickle.dump(record, f) appends.
    Each record is expected to be a tuple:
      (task_idx:int, nc1:float, nc2:float, nc3:float, nc4:float)
    """
    records = []
    with open(path, "rb") as f:
        while True:
            try:
                obj = pickle.load(f)
            except EOFError:
                break
            if isinstance(obj, tuple) and len(obj) == 5:
                task_idx, nc1, nc2, nc3, nc4 = obj
                def to_float(x):
                    try:
                        return float(getattr(x, "item", lambda: x)())
                    except Exception:
                        return float(x)
                records.append((
                    int(task_idx),
                    to_float(nc1), to_float(nc2), to_float(nc3), to_float(nc4)
                ))
            else:
                # Skip non-record objects
                pass
    return records

def read_dict_format(path: str) -> List[Tuple[int, float, float, float, float]]:
    """
    Read a dict-format snapshot:
      {'nc1': ..., 'nc2': ..., 'nc3': ..., 'nc4': ..., 'task_indices': ...}
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    def to_list(v):
        try:
            import torch
            if isinstance(v, torch.Tensor):
                return v.cpu().numpy().tolist()
        except Exception:
            pass
        try:
            import numpy as np
            if isinstance(v, np.ndarray):
                return v.tolist()
        except Exception:
            pass
        return list(v)

    if not isinstance(data, dict):
        raise TypeError("This file is not a dict-format NC snapshot.")

    nc1 = to_list(data.get("nc1", []))
    nc2 = to_list(data.get("nc2", []))
    nc3 = to_list(data.get("nc3", []))
    nc4 = to_list(data.get("nc4", []))
    ti  = to_list(data.get("task_indices", range(len(nc1))))

    n = min(len(ti), len(nc1), len(nc2), len(nc3), len(nc4))
    out = []
    for i in range(n):
        out.append((
            int(ti[i]),
            float(nc1[i]), float(nc2[i]), float(nc3[i]), float(nc4[i])
        ))
    return out

def pretty_print(records: List[Tuple[int, float, float, float, float]]) -> None:
    print("task ID  NC1              NC2              NC3              NC4")
    print("-" * 74)
    for t, a, b, c, d in records:
        print(f"{t:<7d} {a:<16.6f} {b:<16.6f} {c:<16.6f} {d:<16.6f}")
    print(f"\nTotal records: {len(records)}")

def main():
    # Default file path (modify if needed)
    FILE_PATH = "/home/wujy/PL_NC/lop/permuted_mnist/data/bp/std_net/0_NC/0"

    if len(sys.argv) > 1:
        FILE_PATH = sys.argv[1]

    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        sys.exit(1)

    records: List[Tuple[int, float, float, float, float]] = []
    try:
        records = read_incremental_records(FILE_PATH)
    except Exception as e:
        print(f"[WARN] Incremental read failed: {e}")

    if not records:
        try:
            records = read_dict_format(FILE_PATH)
        except Exception as e:
            print(f"[WARN] Dict-format read failed: {e}")

    if not records:
        print("No NC records found in file (neither incremental tuples nor dict-snapshot).")
        sys.exit(2)

    records.sort(key=lambda r: r[0])
    pretty_print(records)

if __name__ == "__main__":
    main()
