import os, numpy as np, argparse
from collections import defaultdict

DEFAULT_WORK_DIR = os.environ.get("HYPERMOE_MRI_WORK_DIR", "./data/mri_work")

def inspect_mri_files(work_dir, num_cases=5):
    required = [
        "before_T1_FS.npy", "before_T1_ROI.npy",
        "before_T2_FS.npy", "before_T2_ROI.npy",
        "after_T1_FS.npy",  "after_T1_ROI.npy",
        "after_T2_FS.npy",  "after_T2_ROI.npy"
    ]
    cases = sorted([d for d in os.listdir(work_dir) if os.path.isdir(os.path.join(work_dir, d))])
    print(f"Total cases: {len(cases)}\n")
    shape_stats = defaultdict(list)
    ndim_stats = defaultdict(list)

    for i, case in enumerate(cases):
        cd = os.path.join(work_dir, case)
        for fn in required:
            fp = os.path.join(cd, fn)
            if not os.path.exists(fp):
                continue
            arr = np.load(fp)
            shape_stats[fn].append(arr.shape)
            ndim_stats[fn].append(arr.ndim)
        if i < num_cases:
            print(f"Sample {i + 1}:")
            for fn in required:
                fp = os.path.join(cd, fn)
                if not os.path.exists(fp):
                    print(f"  {fn:22s} MISSING")
                    continue
                arr = np.load(fp)
                print(f"  {fn:22s} shape={str(arr.shape):20s} dtype={str(arr.dtype):8s} "
                      f"min={arr.min():10.4f} max={arr.max():10.4f} mean={arr.mean():10.4f}")
            print()

    print("=== Shape Consistency ===")
    all_ok = True
    for fn in required:
        shapes = shape_stats.get(fn, [])
        if not shapes:
            continue
        us = set(shapes)
        if len(us) == 1:
            print(f"  {fn:22s} consistent: {list(us)[0]} ({len(shapes)} cases)")
        else:
            all_ok = False
            print(f"  {fn:22s} INCONSISTENT:")
            for s in sorted(us, key=lambda x: (len(x), x)):
                print(f"       {str(s):20s} {shapes.count(s)} cases")
    print(f"\nAll consistent: {all_ok}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--work_dir", default=DEFAULT_WORK_DIR)
    p.add_argument("--num_cases", type=int, default=5)
    a = p.parse_args()
    inspect_mri_files(a.work_dir, a.num_cases)
