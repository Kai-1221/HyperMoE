"""Defaults consumed directly by train_main; CLI arguments override them."""
import os

DATA_ROOT = os.environ.get("HYPERMOE_DATA_ROOT", "./data")
SPLIT_DIR = os.environ.get("HYPERMOE_SPLIT_DIR", os.path.join(DATA_ROOT, "splits_1015"))
WSI_DIR = os.environ.get("HYPERMOE_WSI_DIR", os.path.join(DATA_ROOT, "WSI"))
MRI_DIR = os.environ.get("HYPERMOE_MRI_DIR", os.path.join(DATA_ROOT, "MRI_old_0719_2"))
RESULTS_DIR = os.environ.get("HYPERMOE_RESULTS_DIR", "./results")
FEATURE_DIM = 512
NUM_CLASSES = 2
NUM_EXPERTS = 2
CLASS_WEIGHTS = [1.0, 2.5]
DEFAULT_EPOCHS = 150
DEFAULT_LR = 2.05e-4
DEFAULT_WEIGHT_DECAY = 8e-3
DEFAULT_ACCUM_STEPS = 16
PATIENCE = 12
LR_PATIENCE = 6
MOE_BALANCE_LAMBDA = 0.05
SEED = 42
