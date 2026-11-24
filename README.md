# HyperMoE

Reference implementation of **HyperMoE**, a multimodal model that predicts
pathological complete response (pCR) from a whole-slide image (WSI) and four
MRI acquisitions, and that learns the fusion in hyperbolic space.

The model is a five-level hierarchy — patch, region, slide, tumour ROI, full
fat-suppressed volume — embedded in the Lorentz model of hyperbolic space.
Each level is placed on a prescribed geodesic sphere, the level summaries are
propagated over the hierarchy graph, and two modality experts return to the
manifold, where their outputs are combined in the tangent space at the origin
and classified by distance to two class prototypes.

This repository contains the training pipeline only: no data, no cohort
identifiers, no analysis scripts.

## Layout

    config.py             hyper-parameters and data locations
    dataset_loader.py     h5 readers, region grid, fold splits, dataloader
    losses.py             classification, hyperbolic, entailment, cone, separation
    train.py              cross-validation entry point
    smoke_test.py         builds the model and runs one forward pass
    model/lorentz.py      Lorentz operators (exp, log, distance, cones, means)
    model/geometry.py     metric wrapper used by the losses and the head
    model/temporal_t2.py  temporal attention pooling and the difference branch
    model/topology.py     hierarchy graph and neighbourhood mixing
    model/hyperbolic.py   the network itself

## Data

Three kinds of files are expected. Paths are relative to the working
directory and can be overridden on the command line.

**WSI features** — `data/wsi/<wsi_id>.h5`, one dataset per file:

| key | shape | dtype | meaning |
| --- | --- | --- | --- |
| `coords` | `(N, 2)` | int32 | tile position in the slide grid |
| `features` | `(N, 512)` | float32 | frozen tile embedding |

**MRI tokens** — `data/mri/<mri_id>.h5`:

| key | shape | dtype | meaning |
| --- | --- | --- | --- |
| `fs_feat` | `(4, 512)` | float32 | embedding of the full fat-suppressed volume |
| `roi_feat` | `(4, 512)` | float32 | embedding of the tumour-ROI crop |
| `roi_state` | `(4,)` | int64 | `1` acquisition present, `0` absent |

The four rows are, in order, the two pre-treatment and the two post-treatment
acquisitions.

**Splits** — `data/splits/splits_<fold>_bool.csv`, one row per patient with
the columns `case_id`, `label`, `wsi_id`, `mri_id` and the boolean columns
`train`, `val`, `test`. `data/splits/holdout.csv` uses the same columns and
holds the patients that are never trained on.

Identifiers are only used to locate the two h5 files; nothing else about a
patient is read.

## Install

    python -m pip install -r requirements.txt

Python 3.10 and PyTorch 2.4 are what the experiments used; a single GPU with
about 20 GB is enough, and the pipeline also runs on CPU with the default
settings.

## Run

The defaults in `config.py` are the configuration reported in the paper, so a
plain call reproduces it:

    python train.py \
        --wsi-dir data/wsi --mri-dir data/mri --split-dir data/splits \
        --run-name foldset --folds 0,1,2,3,4

The ablation arms are reached by overriding single switches, for example
`--geometry euclidean`, `--level-radius-band 0` (no per-level placement),
`--cone-aperture-deg 20`, `--lambda-cone 0 --lambda-ent 0 --lambda-hyp 0`,
`--no-level-adaptor`, `--hierarchy-mode no_topo`, `--no-temporal-delta`,
`--fusion-weight-mode gate`, `--num-experts 1 --expert-split mixed`.

`python train.py --help` lists every flag. `python dataset_loader.py --fold 0`
prints the size and prevalence of each split.

## Outputs

Everything is written to `runs/<run-name>/`:

    config.json                 resolved configuration and command line
    fold_summary.csv            per-fold validation and held-out metrics
    fold<k>_best.pth            best checkpoint of fold k
    fold<k>_history.csv         per-epoch losses and validation metrics
    fold<k>_val_predictions.csv, fold<k>_holdout_predictions.csv
    oof_predictions.csv         out-of-fold predictions, one row per patient
    oof_metrics.json            AUC, AUPRC, Brier and confidence intervals on the out-of-fold set
    holdout_predictions.csv     held-out predictions, all folds
    holdout_metrics.json        the same metrics on the held-out patients

## Method notes

* Encoders are frozen: the model reads cached embeddings and trains only the
  hierarchy, the temporal block, the experts and the classification head.
* The two modalities are trained jointly; at inference either one can be
  masked out, and the fusion weights are renormalised over what remains.
* The class decision is a nearest-prototype rule on the manifold, so the
  logits are negative hyperbolic distances divided by a learned temperature.
* Batch size is fixed to one patient, with gradient accumulation of sixteen.

## License

Released for anonymous review; the license will be added on publication.
