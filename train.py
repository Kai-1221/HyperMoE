from __future__ import annotations

import argparse
import inspect
import json
import math
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn

from config import Config
from dataset_loader import make_loader, load_split_rows
from losses import compute_losses, router_balance
from model.hyperbolic import MODEL_VERSION, build_model

METRICS = ("auc", "auprc", "f1", "precision", "recall", "specificity", "acc", "brier")

WSI_KEYS = ("wsi_patch", "wsi_region", "wsi_slide")
MRI_KEYS = ("mri_fs", "mri_roi", "roi_state")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def forward_batch(model, item: dict, device: torch.device):
    return model(
        item["wsi_patch"].to(device),
        item["wsi_region"].to(device),
        item["wsi_slide"].to(device),
        item["wsi_p2r"].to(device),
        item["mri_fs"].to(device),
        item["mri_roi"].to(device),
        item["roi_state"].to(device),
    )


def mask_modality(item: dict, cfg: Config) -> dict:
    mode = str(getattr(cfg, "modality_mode", "both"))
    if mode == "mri":
        return {**item, **{k: torch.zeros_like(item[k]) for k in WSI_KEYS}}
    if mode == "wsi":
        return {**item, **{k: torch.zeros_like(item[k]) for k in MRI_KEYS}}

    p = float(getattr(cfg, "modality_dropout", 0.0))
    if p <= 0.0:
        return item
    u = torch.rand(2)
    if float(u[0]) < 0.5 * p:
        return {**item, **{k: torch.zeros_like(item[k]) for k in WSI_KEYS}}
    if float(u[1]) < 0.5 * p:
        return {**item, **{k: torch.zeros_like(item[k]) for k in MRI_KEYS}}
    return item


def metric_dict(y_true, y_prob, threshold: float) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    two = len(np.unique(y)) == 2
    return {
        "auc": float(roc_auc_score(y, p)) if two else float("nan"),
        "auprc": float(average_precision_score(y, p)) if two else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "acc": float((tp + tn) / len(y)) if len(y) else float("nan"),
        "brier": float(brier_score_loss(y, p)) if two else float("nan"),
    }


def best_threshold(y_true, y_prob, fallback: float = 0.5) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if not len(thresholds):
        return float(fallback)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(
        precision[:-1] + recall[:-1], 1e-12, None
    )
    return float(thresholds[int(np.argmax(f1))])


def bootstrap_auc(y_true, y_prob, seed: int, repetitions: int = 2000) -> list[float]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true)
    p = np.asarray(y_prob)
    vals = []
    for _ in range(repetitions):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            vals.append(roc_auc_score(y[idx], p[idx]))
    if not vals:
        return [float("nan"), float("nan")]
    return [float(v) for v in np.quantile(vals, [0.025, 0.975])]


def initialize_class_prototypes(model, loader, device: torch.device, cfg: Config) -> torch.Tensor:
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for item in loader:
            forward_batch(model, item, device)
            feats.append(model.latest_fh.detach().cpu())
            labels.append(int(item["label"].item()))
    with torch.no_grad():
        features = torch.cat(feats, dim=0)
        labels_t = torch.tensor(labels, dtype=torch.long)
        if len(torch.unique(labels_t)) != 2:
            raise ValueError("training split requires both classes")
        centers = []
        for c in (0, 1):
            center = model.geo.mean(features[labels_t == c].unsqueeze(0)).squeeze(0)
            centers.append(center[1:].to(device))

        axis = centers[1] - centers[0]
        if not torch.isfinite(axis).all() or float(axis.norm()) < 1e-6:
            g = torch.Generator().manual_seed(int(cfg.seed))
            axis = torch.randn(axis.shape, generator=g).to(device)
        u = axis / axis.norm()

        radius = float(getattr(cfg, "proto_init_radius", 1.0))
        sep = float(getattr(cfg, "proto_init_sep", 1.2))
        sep = min(sep, 2.0 * radius * 0.999)
        angle = math.asin(max(1e-4, sep / (2.0 * radius)))

        g = torch.Generator().manual_seed(int(cfg.seed) + 1)
        v = torch.randn(u.shape, generator=g).to(device)
        v = v - (v @ u) * u
        if float(v.norm()) < 1e-6:
            e = torch.zeros_like(u)
            e[0] = 1.0
            v = e - (e @ u) * u
        v = v / v.norm()

        scale = float(getattr(model.geo, "spatial_scale", 1.0))
        proto = scale * torch.stack((
            radius * (math.cos(angle) * v - math.sin(angle) * u),
            radius * (math.cos(angle) * v + math.sin(angle) * u),
        )).to(device)
        model.class_prototypes.copy_(proto)
        return proto


def run_epoch(model, loader, cfg: Config, device: torch.device,
              optimizer=None, class_weights=None, threshold: float = 0.5,
              lambda_scale: float = 1.0) -> dict:
    training = optimizer is not None
    model.train(training)
    criterion_weights = class_weights
    totals = dict(loss=0.0, cls=0.0, hyp=0.0, ent=0.0, cone=0.0, proto_sep=0.0, bal=0.0)
    labels, probs, weights_all = [], [], []
    window_features = []
    if training:
        optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(training):
        for step, item in enumerate(loader):
            label = item["label"].to(device).view(1)
            if training:
                item = mask_modality(item, cfg)
            elif str(getattr(cfg, "modality_mode", "both")) != "both":
                item = mask_modality(item, cfg)
            logits, weights = forward_batch(model, item, device)
            losses = compute_losses(
                model, logits, label, cfg, criterion_weights, lambda_scale=lambda_scale
            )
            total = losses["total"]
            if not torch.isfinite(total):
                raise FloatingPointError(f"nonfinite loss at step {step}")
            window_start = step // cfg.grad_accum * cfg.grad_accum
            window_size = min(cfg.grad_accum, len(loader) - window_start)
            if training:
                (total / window_size).backward()
                if cfg.lambda_bal > 0 and model.num_experts > 1:
                    if str(getattr(cfg, "expert_split", "mixed")) == "modality":
                        window_features = []
                    else:
                        window_features.append(model.latest_combined_h.detach())
                if step + 1 == window_start + window_size:
                    if window_features:
                        balance = router_balance(model, torch.cat(window_features, dim=0))
                        if balance.requires_grad:
                            (cfg.lambda_bal * balance).backward()
                        totals["bal"] += float(balance.item()) * window_size
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.grad_clip, error_if_nonfinite=True
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    window_features = []
            for key in ("loss", "cls", "hyp", "ent", "cone", "proto_sep"):
                value = losses["total"] if key == "loss" else losses[key]
                totals[key] += float(value.item())
            labels.append(int(label.item()))
            probs.append(float(torch.softmax(logits.detach(), -1)[0, 1].item()))
            weights_all.append(weights.detach().cpu())
    weights = torch.cat(weights_all, dim=0).numpy()
    result = {k: v / len(loader) for k, v in totals.items()}
    entropy = -(weights * np.log(np.clip(weights, 1e-12, None))).sum(1)
    result.update(
        metrics=metric_dict(labels, probs, threshold),
        labels=labels,
        probs=probs,
        weights=weights,
        w1_std=float(weights[:, 0].std(ddof=1)) if len(weights) > 1 else 0.0,
        entropy_norm=float(entropy.mean() / math.log(model.num_experts))
        if model.num_experts > 1 else 0.0,
    )
    return result


def load_checkpoint(path: Path, model, device: torch.device):
    kwargs = {"map_location": device}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = True
    ckpt = torch.load(path, **kwargs)
    if ckpt.get("model_version") != MODEL_VERSION:
        raise ValueError("checkpoint model version mismatch")
    model.load_state_dict(ckpt["model_state"], strict=True)
    return ckpt


def run_fold(cfg: Config, fold: int, device: torch.device, run_dir: Path) -> dict:
    set_seed(cfg.seed + fold)
    train_loader = make_loader(cfg, fold, "train", shuffle=True)
    val_loader = make_loader(cfg, fold, "val", shuffle=False)
    holdout_loader = make_loader(cfg, fold, "holdout", shuffle=False)

    rows = load_split_rows(cfg, fold, "train")
    labels = np.asarray([int(r["label"]) for r in rows], dtype=float)
    pos_rate = float(labels.mean())
    raw = np.asarray(cfg.class_weights, dtype=float)
    norm = raw[0] * (1.0 - pos_rate) + raw[1] * pos_rate
    class_weights = torch.tensor(raw / norm, dtype=torch.float32, device=device)

    model = build_model(cfg).to(device)
    init_proto = initialize_class_prototypes(model, train_loader, device, cfg)
    print(
        f"fold {fold}: prototype init radius={float(init_proto.norm(dim=-1).mean()):.3f} "
        f"sep={float((init_proto[1] - init_proto[0]).norm()):.3f}",
        flush=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg.lr_factor, patience=cfg.lr_patience,
        threshold=0.0, threshold_mode="abs",
    )

    lr_warmup = int(getattr(cfg, "lr_warmup_epochs", 0))
    smooth = deque(maxlen=max(1, int(getattr(cfg, "smooth_window", 1))))
    best_auc, best_epoch, counter = -math.inf, 0, 0
    history = []
    for epoch in range(1, cfg.epochs + 1):
        start = time.perf_counter()
        ramp = (
            min(1.0, (epoch - 1) / cfg.warmup_epochs)
            if cfg.warmup_epochs > 0 else 1.0
        )
        if lr_warmup > 0 and epoch <= lr_warmup:
            for group in optimizer.param_groups:
                group["lr"] = cfg.lr * epoch / lr_warmup
        train_res = run_epoch(model, train_loader, cfg, device, optimizer,
                              class_weights, threshold=0.5, lambda_scale=ramp)
        val_res = run_epoch(model, val_loader, cfg, device, None,
                            class_weights, threshold=0.5)
        auc = val_res["metrics"]["auc"]
        if not (lr_warmup > 0 and epoch <= lr_warmup):
            scheduler.step(auc)
        smooth.append(auc)
        min_select = max(int(getattr(cfg, "min_select_epoch", 0)), smooth.maxlen)
        if epoch < min_select or len(smooth) < smooth.maxlen:
            score = -math.inf
        else:
            score = float(np.mean(smooth))
        if score > best_auc:
            best_auc, best_epoch, counter = score, epoch, 0
            torch.save({
                "model_version": MODEL_VERSION,
                "model_state": model.state_dict(),
                "fold": fold,
                "epoch": epoch,
                "val_auc": auc,
                "val_auc_smooth": score,
                "config": cfg.to_dict(),
            }, run_dir / f"fold{fold}_best.pth")
        else:
            counter += 1
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
               "patience": counter, "best_epoch": best_epoch}
        for prefix, res in (("train", train_res), ("val", val_res)):
            for key in ("loss", "cls", "hyp", "ent", "cone", "proto_sep", "bal",
                        "w1_std", "entropy_norm"):
                row[f"{prefix}_{key}"] = res[key]
            for key in METRICS:
                row[f"{prefix}_{key}"] = res["metrics"][key]
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / f"fold{fold}_history.csv", index=False)
        print(
            f"fold {fold} epoch {epoch:03d} "
            f"train_loss={train_res['loss']:.4f} cls={train_res['cls']:.4f} "
            f"hyp={train_res['hyp']:.3f} ent={train_res['ent']:.3f} "
            f"cone={train_res['cone']:.3f} | "
            f"val_auc={auc:.4f} val_f1@0.5={val_res['metrics']['f1']:.4f} "
            f"lr={row['lr']:.2e} patience={counter}/{cfg.patience} "
            f"({time.perf_counter() - start:.1f}s)"
        )
        if counter >= cfg.patience:
            print(f"fold {fold}: early stop at epoch {epoch}, best={best_epoch}")
            break

    if best_epoch == 0:
        best_epoch = epoch
        best_auc = float(np.mean(smooth)) if len(smooth) else -math.inf
        print(f"fold {fold}: no epoch passed the smoothing floor; falling back to epoch {epoch}")
        torch.save({
            "model_version": MODEL_VERSION,
            "model_state": model.state_dict(),
            "fold": fold,
            "epoch": epoch,
            "val_auc": auc,
            "val_auc_smooth": best_auc,
            "config": cfg.to_dict(),
        }, run_dir / f"fold{fold}_best.pth")

    ckpt = load_checkpoint(run_dir / f"fold{fold}_best.pth", model, device)
    model.eval()
    val_final = run_epoch(model, val_loader, cfg, device, None, class_weights, threshold=0.5)
    threshold = best_threshold(val_final["labels"], val_final["probs"])
    val_metrics = metric_dict(val_final["labels"], val_final["probs"], threshold)
    hold_final = run_epoch(
        model, holdout_loader, cfg, device, None, class_weights, threshold=threshold
    )
    hold_metrics = metric_dict(hold_final["labels"], hold_final["probs"], threshold)

    val_pred = pd.DataFrame({
        "case_id": [r["case_id"] for r in make_loader(cfg, fold, "val").dataset.rows],
        "fold": fold,
        "label": val_final["labels"],
        "prob": val_final["probs"],
        "threshold": threshold,
    })
    val_pred["pred"] = (val_pred["prob"] >= threshold).astype(int)
    val_pred.to_csv(run_dir / f"fold{fold}_val_predictions.csv", index=False)

    hold_pred = pd.DataFrame({
        "case_id": [r["case_id"] for r in make_loader(cfg, fold, "holdout").dataset.rows],
        "fold": fold,
        "label": hold_final["labels"],
        "prob": hold_final["probs"],
        "threshold": threshold,
    })
    hold_pred["pred"] = (hold_pred["prob"] >= threshold).astype(int)
    hold_pred.to_csv(run_dir / f"fold{fold}_holdout_predictions.csv", index=False)

    summary = {
        "fold": fold,
        "best_epoch": ckpt["epoch"],
        "best_val_auc": ckpt["val_auc"],
        "threshold": threshold,
        "n_train": len(train_loader.dataset),
        "n_val": len(val_loader.dataset),
        "n_holdout": len(holdout_loader.dataset),
    }
    summary.update({f"val_{k}": val_metrics[k] for k in METRICS})
    summary.update({f"holdout_{k}": hold_metrics[k] for k in METRICS})
    return {"summary": summary, "val_pred": val_pred, "hold_pred": hold_pred}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wsi-dir", default=None)
    p.add_argument("--mri-dir", default=None)
    p.add_argument("--split-dir", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num-experts", type=int, default=None)
    p.add_argument("--moe-hidden", type=int, default=None)
    p.add_argument("--moe-dropout", type=float, default=None)
    p.add_argument("--proto-init-sep", type=float, default=None)
    p.add_argument("--lr-warmup-epochs", type=int, default=None)
    p.add_argument("--smooth-window", type=int, default=None)
    p.add_argument("--min-select-epoch", type=int, default=None)
    p.add_argument("--modality-dropout", type=float, default=None)
    p.add_argument("--modality-mode", choices=["both", "mri", "wsi"], default=None)
    p.add_argument("--lambda-hyp", type=float, default=None)
    p.add_argument("--lambda-ent", type=float, default=None)
    p.add_argument("--lambda-cone", type=float, default=None)
    p.add_argument("--lambda-proto-sep", type=float, default=None)
    p.add_argument("--lambda-bal", type=float, default=None)
    p.add_argument("--warmup-epochs", type=int, default=None)
    p.add_argument("--temporal-dropout", type=float, default=None)
    p.add_argument("--temporal-dim", type=int, default=None)
    p.add_argument("--temporal-delta", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--temporal", choices=["t0", "t1", "t2"], default=None)
    p.add_argument("--geometry", choices=["hyper", "euclidean"], default=None)
    p.add_argument("--router-mode", choices=["proto", "tangent"], default=None)
    p.add_argument("--router-tau-init", type=float, default=None)
    p.add_argument("--expert-indep-init", action="store_true")
    p.add_argument("--expert-split", choices=["mixed", "modality"], default=None)
    p.add_argument(
        "--hierarchy-mode", choices=["full", "no_topo", "no_levels"], default=None
    )
    p.add_argument("--fusion-weight-mode", choices=["gate", "scalar", "fixed"], default=None)
    p.add_argument("--fusion-w-init", type=float, default=None)
    p.add_argument("--expert-aux", type=float, default=None)
    p.add_argument("--patch-aux", type=float, default=None)
    p.add_argument("--cone-aperture-deg", type=float, default=None)
    p.add_argument("--cone-normalize", action="store_true")
    p.add_argument("--cone-tie-class", action="store_true")
    p.add_argument("--level-adaptor", action="store_true")
    p.add_argument("--level-adaptor-hidden", type=int, default=None)
    p.add_argument(
        "--level-radii",
        type=float,
        nargs="+",
        default=None,
        help="explicit radii for (patch, region, slide, roi, fs); empty = learned from the pooled means",
    )
    p.add_argument("--level-radius-band", type=float, default=None)
    p.add_argument("--cone-level-edges", action="store_true")
    p.add_argument("--head-radius", type=float, default=None)
    p.add_argument("--run-name", default="run")
    p.add_argument("--test-name", default="hypermoe")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = Config()
    if args.wsi_dir:
        cfg.wsi_dir = Path(args.wsi_dir)
    if args.mri_dir:
        cfg.mri_dir = Path(args.mri_dir)
    if args.split_dir:
        cfg.split_dir = Path(args.split_dir)
    if args.out_dir:
        cfg.out_dir = Path(args.out_dir)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.lr is not None:
        cfg.lr = args.lr
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    if args.seed is not None:
        cfg.seed = args.seed
    if args.num_experts is not None:
        cfg.num_experts = args.num_experts
    if args.moe_hidden is not None:
        cfg.moe_hidden = args.moe_hidden
    if args.moe_dropout is not None:
        cfg.moe_dropout = args.moe_dropout
    if args.proto_init_sep is not None:
        cfg.proto_init_sep = args.proto_init_sep
    if args.lr_warmup_epochs is not None:
        cfg.lr_warmup_epochs = args.lr_warmup_epochs
    if args.smooth_window is not None:
        cfg.smooth_window = args.smooth_window
    if args.min_select_epoch is not None:
        cfg.min_select_epoch = args.min_select_epoch
    if args.modality_dropout is not None:
        cfg.modality_dropout = args.modality_dropout
    if args.modality_mode is not None:
        cfg.modality_mode = args.modality_mode
    if args.lambda_hyp is not None:
        cfg.lambda_hyp = args.lambda_hyp
    if args.lambda_ent is not None:
        cfg.lambda_ent = args.lambda_ent
    if args.lambda_cone is not None:
        cfg.lambda_cone = args.lambda_cone
    if args.lambda_proto_sep is not None:
        cfg.lambda_proto_sep = args.lambda_proto_sep
    if args.lambda_bal is not None:
        cfg.lambda_bal = args.lambda_bal
    if args.cone_aperture_deg is not None:
        cfg.cone_aperture_deg = args.cone_aperture_deg
    if args.cone_normalize:
        cfg.cone_normalize = True
    if args.cone_tie_class:
        cfg.cone_tie_class = True
    if args.level_adaptor:
        cfg.level_adaptor = True
    if args.level_adaptor_hidden is not None:
        cfg.level_adaptor_hidden = args.level_adaptor_hidden
    if args.level_radii:
        cfg.level_radii = [float(r) for r in args.level_radii]
    if args.level_radius_band is not None:
        cfg.level_radius_band = args.level_radius_band
    if args.cone_level_edges:
        cfg.cone_level_edges = True
    if args.head_radius is not None:
        cfg.head_radius = args.head_radius
    if args.warmup_epochs is not None:
        cfg.warmup_epochs = args.warmup_epochs
    if args.temporal_dropout is not None:
        cfg.temporal_dropout = args.temporal_dropout
    if args.temporal_dim is not None:
        cfg.temporal_dim = args.temporal_dim
    if args.temporal_delta:
        cfg.temporal_delta = True
    if args.temporal:
        cfg.temporal = args.temporal
    if args.geometry:
        cfg.geometry = args.geometry
    if args.router_mode:
        cfg.router_mode = args.router_mode
    if args.router_tau_init is not None:
        cfg.router_tau_init = args.router_tau_init
    if args.expert_indep_init:
        cfg.expert_indep_init = True
    if args.expert_split:
        cfg.expert_split = args.expert_split
    if args.hierarchy_mode:
        cfg.hierarchy_mode = args.hierarchy_mode
    if args.fusion_weight_mode:
        cfg.fusion_weight_mode = args.fusion_weight_mode
    if args.fusion_w_init is not None:
        cfg.fusion_w_init = args.fusion_w_init
    if args.expert_aux is not None:
        cfg.expert_aux = args.expert_aux
    if args.patch_aux is not None:
        cfg.patch_aux = args.patch_aux
    cfg.validate()

    folds = [int(v) for v in args.folds.split(",")]
    if not folds or len(set(folds)) != len(folds) or any(v not in range(5) for v in folds):
        raise ValueError("--folds must be distinct indices in 0..4")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    run_dir = cfg.out_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps({
            **cfg.to_dict(),
            "args": vars(args),
            "model_version": MODEL_VERSION,
            "torch_version": str(torch.__version__),
            "device": str(device),
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"device={device} out={run_dir}")
    print(f"temporal={cfg.temporal} geometry={cfg.geometry} lr={cfg.lr} wd={cfg.weight_decay}")

    summaries, val_preds, hold_preds = [], [], []
    for fold in folds:
        out = run_fold(cfg, fold, device, run_dir)
        summaries.append(out["summary"])
        val_preds.append(out["val_pred"])
        hold_preds.append(out["hold_pred"])
        pd.DataFrame(summaries).to_csv(run_dir / "fold_summary.csv", index=False)

    summary = pd.DataFrame(summaries)
    pd.DataFrame([
        {"metric": k, "mean": summary[f"val_{k}"].mean(),
         "std": summary[f"val_{k}"].std(ddof=1), "n_folds": len(summary)}
        for k in METRICS
    ]).to_csv(run_dir / "val_summary_mean_std.csv", index=False)
    pd.DataFrame([
        {"metric": k, "mean": summary[f"holdout_{k}"].mean(),
         "std": summary[f"holdout_{k}"].std(ddof=1), "n_folds": len(summary)}
        for k in METRICS
    ]).to_csv(run_dir / "holdout_summary_mean_std.csv", index=False)

    oof = pd.concat(val_preds, ignore_index=True)
    if oof["case_id"].duplicated().any():
        raise ValueError("duplicate OOF patients")
    oof.to_csv(run_dir / "oof_predictions.csv", index=False)
    oof_metrics = metric_dict(oof["label"], oof["prob"], 0.5)
    oof_metrics["auc_ci95"] = bootstrap_auc(oof["label"], oof["prob"], cfg.seed, cfg.bootstrap_n)
    oof_metrics["n_patients"] = int(len(oof))
    oof_metrics["n_folds"] = len(folds)
    (run_dir / "oof_metrics.json").write_text(
        json.dumps(oof_metrics, indent=2), encoding="utf-8"
    )

    hold = pd.concat(hold_preds, ignore_index=True)
    hold.to_csv(run_dir / "holdout_predictions.csv", index=False)
    hold_metrics = {
        f"fold_mean_{k}": float(summary[f"holdout_{k}"].mean()) for k in METRICS
    }
    hold_metrics["n_folds"] = len(folds)
    hold_metrics["n_per_fold"] = int(summary["n_holdout"].iloc[0])
    (run_dir / "holdout_metrics.json").write_text(
        json.dumps(hold_metrics, indent=2), encoding="utf-8"
    )

    print("\nOOF:")
    print(json.dumps(oof_metrics, indent=2))
    print("\nHoldout fold-mean:")
    print(json.dumps(hold_metrics, indent=2))
    print(f"\nwrote {run_dir}")

if __name__ == "__main__":
    main()
