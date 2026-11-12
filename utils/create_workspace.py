import os

DATA_ROOT = os.environ.get("HYPERMOE_DATA_ROOT", "./data")
SRC_DIR = os.environ.get("HYPERMOE_MRI_SOURCE_DIR", os.path.join(DATA_ROOT, "mri_source"))
WORK_DIR = os.environ.get("HYPERMOE_MRI_WORK_DIR", os.path.join(DATA_ROOT, "mri_work"))

def create_work_space():
    if not os.path.isdir(SRC_DIR):
        raise FileNotFoundError("MRI source directory is unavailable")
    if not os.path.exists(WORK_DIR):
        print("创建 MRI 工作目录")
        os.makedirs(WORK_DIR, exist_ok=True)
    else:
        print("MRI 工作目录已存在，开始同步")

    cases = [d for d in os.listdir(SRC_DIR) if os.path.isdir(os.path.join(SRC_DIR, d))]
    
    for case in cases:
        src_case_dir = os.path.join(SRC_DIR, case)
        dst_case_dir = os.path.join(WORK_DIR, case)
        
        if not os.path.exists(dst_case_dir):
            os.makedirs(dst_case_dir)
            
        for filename in os.listdir(src_case_dir):
            src_file = os.path.join(src_case_dir, filename)
            dst_file = os.path.join(dst_case_dir, filename)
            
            if not os.path.exists(dst_file):
                try:
                    os.symlink(src_file, dst_file)
                except OSError as e:
                    print(f"链接失败: {filename} ({type(e).__name__})")
                    
    print("MRI 工作目录同步完成")

if __name__ == "__main__":
    create_work_space()
