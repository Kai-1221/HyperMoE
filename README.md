# HyperMoE 1110

This repository contains the patient-level training code for a multimodal WSI-MRI
classification model. It combines hierarchical WSI features, global and ROI-pooled
MRI features, mixture-of-experts routing, and Lorentz-space objectives.

The code reports cross-validation results used for internal model selection. These
results are not an independent test-set evaluation.

## Repository contents

- `train_main.py`: training and evaluation entry point.
- `dataset_loader.py`: split validation and HDF5 feature loading.
- `feature/moe_wsi_mri.py`: multimodal HyperMoE model.
- `feature/unified_topology.py`: WSI-MRI hierarchy construction.
- `feature/mri_extractor.py`: paired-timepoint MRI feature extraction.
- `utils/lorentz_utils.py`: Lorentz geometry operations.

WSI feature extraction and split generation are not included in the current release.
Their outputs must be prepared before training.

## Environment

The main dependencies are Python, PyTorch, torchvision, NumPy, pandas, h5py,
scikit-learn, matplotlib, and tqdm. Record the exact Python, CUDA, PyTorch, and
torchvision versions used for the archived paper release.


## MRI feature extraction

From the repository root:

```bash
python -m feature.mri_extractor
```

The extractor uses the Kinetics-400 R3D-18 weights and a default random seed of 42.
The seed can be changed with `HYPERMOE_MRI_SEED`. Repeated extraction is deterministic
when the code, dependency versions, hardware, and input arrays are unchanged.

The feature heads are initialized by this script. If the study used trained MRI
feature-head weights, the corresponding checkpoint must be released and loaded;
fixing the random seed is not a substitute for a trained checkpoint.

## Training

Run from the repository root:

```bash
python train_main.py \
  --task hyper_final \
  --split_dir ./data/splits_1015 \
  --wsi_dir ./data/WSI \
  --mri_dir ./data/MRI \
  --results_dir ./results
```



