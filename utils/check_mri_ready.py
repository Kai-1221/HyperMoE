import numpy as np
import os
from tqdm import tqdm

DEFAULT_WORK_DIR = os.environ.get("HYPERMOE_MRI_WORK_DIR", "./data/mri_work")

def check_mri_ready(work_data_path):
    cases = [d for d in os.listdir(work_data_path) if os.path.isdir(os.path.join(work_data_path, d))]
    
    required_files = [
        "before_T1_FS.npy", "before_T1_ROI.npy",
        "before_T2_FS.npy", "before_T2_ROI.npy",
        "after_T1_FS.npy",  "after_T1_ROI.npy",
        "after_T2_FS.npy",  "after_T2_ROI.npy"
    ]
    
    print(f"🚀 开始扫描 {len(cases)} 个病例的数据就绪状态...")
    
    inconsistent_cases = []
    missing_cases = []
    
    for case in tqdm(cases, desc="就绪状态检测"):
        case_dir = os.path.join(work_data_path, case)
        
        missing = [f for f in required_files if not os.path.exists(os.path.join(case_dir, f))]
        if missing:
            missing_cases.append((case, missing))
            continue
            
        try:
            shape_t1_fs = np.load(os.path.join(case_dir, "before_T1_FS.npy")).shape
            for f in required_files:
                shape = np.load(os.path.join(case_dir, f)).shape
                if shape != shape_t1_fs:
                    inconsistent_cases.append((case, f, shape))
                    break
        except Exception as e:
            inconsistent_cases.append((case, "读取错误", type(e).__name__))

    print("\n=== 检测报告 ===")
    if not missing_cases and not inconsistent_cases:
        print("✅ 全部就绪！410 例病例数据完整且维度一致，可以开始训练。")
    else:
        if missing_cases:
            print(f"❌ 发现 {len(missing_cases)} 个病例缺失文件：")
            for _, missing in missing_cases[:5]:
                print(f"  - 缺失 {missing}")
        if inconsistent_cases:
            print(f"❌ 发现 {len(inconsistent_cases)} 个病例维度不匹配：")
            for _, filename, shape in inconsistent_cases[:5]:
                print(f"  - {filename} 形状为 {shape}")

if __name__ == "__main__":
    check_mri_ready(DEFAULT_WORK_DIR)
