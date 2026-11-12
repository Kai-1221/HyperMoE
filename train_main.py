"""Patient-level CV training for HyperMoE 1110.

Validation AUC selects checkpoints and controls both patience counters. At each
epoch, the validation threshold is selected to maximize F1 and is saved with
the AUC-selected checkpoint. The reported scores are internal model-selection
validation, not an independent test evaluation.
"""
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix, precision_recall_curve)
from math import pi

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts_1110 import config
    from scripts_1110.dataset_loader import get_dataloader, load_split
    from scripts_1110.feature.moe_wsi_mri import UnifiedHierarchyMoE, MODEL_VERSION
    from scripts_1110.utils.lorentz_utils import (lorentz_distance, origin_like,
                                                   entail_cone_loss, angular_separation, lorentz_mean)
else:
    from . import config
    from .dataset_loader import get_dataloader, load_split
    from .feature.moe_wsi_mri import UnifiedHierarchyMoE, MODEL_VERSION
    from .utils.lorentz_utils import (lorentz_distance, origin_like,
                                      entail_cone_loss, angular_separation, lorentz_mean)


METRICS = ("auc", "specificity", "precision", "recall", "f1", "acc")


def calculate_metrics(y_true, y_prob, threshold=0.5, y_pred=None):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    pred = ((p >= threshold).astype(int) if y_pred is None
            else np.asarray(y_pred, dtype=int))
    if pred.shape != y.shape or not np.isin(pred, [0, 1]).all():
        raise ValueError("y_pred must be binary and have the same shape as y_true")
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(auc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
                acc=float(accuracy_score(y, pred)),
                f1=float(f1_score(y, pred, zero_division=0)),
                precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred, zero_division=0)),
                specificity=float(tn / (tn + fp)) if tn + fp else float("nan"))


def get_best_threshold(y_true, y_prob, fallback=0.5):
    """Return the validation probability threshold that maximizes F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if not len(thresholds):
        return float(fallback)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.argmax(f1))])


def classification_counts(y_true, y_prob, threshold):
    y = np.asarray(y_true, dtype=int)
    pred = (np.asarray(y_prob, dtype=float) >= threshold).astype(int)
    result = {}
    for label in (0, 1):
        mask = y == label
        result[f"class_{label}_total"] = int(mask.sum())
        result[f"class_{label}_correct"] = int((pred[mask] == label).sum())
    result["total"] = int(len(y))
    result["correct"] = int((pred == y).sum())
    return result


def forward_batch(model, wsi, mri, device):
    w = {k: v[0].to(device) for k, v in wsi.items()} if wsi is not None else {}
    m = {k: v[0].to(device) for k, v in mri.items()} if mri is not None else {}
    return model(w.get("patches"), w.get("regions"), w.get("slide"),
                 mri_roi=m.get("roi_feat"), mri_fs=m.get("fs_feat"), p2r=w.get("p2r"))


def geometry_losses(model, label):
    """Return supervised hyperbolic, radial, cone, and prototype losses."""
    fh = model.latest_fh
    cph = model._lift(model.class_prototypes)
    target = label.reshape(-1)
    pos = lorentz_distance(fh, cph[target]).mean()
    neg = lorentz_distance(fh, cph[1 - target]).mean()
    hyp = (pos + F.relu(pos - neg + 0.5)).clamp(max=2.0)
    raw = model.latest_raw_h
    ent = raw.new_zeros(())
    cone = raw.new_zeros(())
    for parent_index, child_index in model.hierarchy_edges:
        parent, child = raw[parent_index], raw[child_index]
        parent_r = lorentz_distance(parent, origin_like(parent))
        child_r = lorentz_distance(child, origin_like(child))
        ent = ent + F.relu(parent_r - child_r + 0.1).mean()
        cone = cone + entail_cone_loss(parent, child)
    families = max(1, len(model.hierarchy_edges))
    cone = cone / families

    # The class-specific cones organize the five pooled scale embeddings.
    class_index = int(target[0].item())
    apex = model.latest_cone_apex[class_index:class_index + 1]
    axis = model.latest_cone_axis[class_index:class_index + 1]
    samples = model.latest_level_h
    angle = angular_separation(apex.expand_as(samples), samples,
                               axis.expand_as(samples))
    aperture = torch.sigmoid(model.cone_apertures[class_index]) * (pi / 2.0)
    cone = cone + F.relu(angle - aperture).mean()

    class_distance = lorentz_distance(cph[0:1], cph[1:2])
    cone_distance = lorentz_distance(model.latest_cone_apex[0:1],
                                     model.latest_cone_apex[1:2])
    proto_sep = F.relu(2.0 - class_distance).mean() + F.relu(2.0 - cone_distance).mean()
    return hyp, ent / families, cone, proto_sep


def router_balance(model, detached_features):
    """Window usage penalty; detached inputs deliberately train only router prototypes."""
    ph = model._lift(model.prototypes)
    distances = lorentz_distance(detached_features.unsqueeze(1), ph.unsqueeze(0)).squeeze(-1)
    usage = torch.softmax(-distances, dim=-1).mean(0)
    return (usage - 1.0 / model.num_experts).square().sum()


def run_epoch(model, loader, optimizer=None, *, class_weights=None, accum_steps=16,
              threshold=0.5, lambda_hyp=0.1, lambda_ent=0.01, lambda_cone=0.01,
              lambda_proto_sep=0.01, bal_weight=0.05):
    training = optimizer is not None
    model.train(training)
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
    if not len(loader):
        raise ValueError("Cannot evaluate an empty loader")
    totals = dict(loss=0.0, cls=0.0, bal=0.0, hyp=0.0, ent=0.0, cone=0.0, proto_sep=0.0)
    labels, probs, all_weights, output_gaps = [], [], [], []
    window_features = []
    if training:
        optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(training):
        for step, (wsi, mri, label) in enumerate(loader):
            label = label.to(device)
            logits, weights = forward_batch(model, wsi, mri, device)
            cls = criterion(logits, label).mean()
            hyp = ent = cone = proto_sep = cls.new_zeros(())
            if training and (lambda_hyp or lambda_ent or lambda_cone or lambda_proto_sep):
                hyp, ent, cone, proto_sep = geometry_losses(model, label)
            total = (cls + lambda_hyp * hyp + lambda_ent * ent +
                     lambda_cone * cone + lambda_proto_sep * proto_sep)
            if not torch.isfinite(total):
                raise FloatingPointError(f"Nonfinite loss at patient step {step}; training stopped")
            window_start = step // accum_steps * accum_steps
            window_size = min(accum_steps, len(loader) - window_start)
            if training:
                (total / window_size).backward()
                if bal_weight > 0 and model.num_experts > 1:
                    window_features.append(model.latest_combined_h.detach())
                if step + 1 == window_start + window_size:
                    if window_features:
                        balance = router_balance(model, torch.cat(window_features))
                        (bal_weight * balance).backward()
                        totals["bal"] += balance.item() * window_size
                        totals["loss"] += bal_weight * balance.item() * window_size
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    window_features = []
            for key, value in (("loss", total), ("cls", cls), ("hyp", hyp), ("ent", ent),
                               ("cone", cone), ("proto_sep", proto_sep)):
                totals[key] += value.item()
            labels.extend(label.detach().cpu().tolist())
            probs.extend(torch.softmax(logits.detach(), -1)[:, 1].cpu().tolist())
            all_weights.append(weights.detach().cpu())
            if model.num_experts > 1:
                h = model.latest_expert_h.detach()[0]
                gaps = [lorentz_distance(h[i:i+1], h[j:j+1]).item()
                        for i in range(model.num_experts) for j in range(i+1, model.num_experts)]
                output_gaps.append(float(np.mean(gaps)))
    weights = torch.cat(all_weights).numpy()
    entropy = -(weights * np.log(np.clip(weights, 1e-12, None))).sum(1)
    result = {k: v / len(loader) for k, v in totals.items()}
    if not training:
        result["bal"] = float(np.square(weights.mean(0) - 1.0 / model.num_experts).sum())
    result.update(metrics=calculate_metrics(labels, probs, threshold), labels=labels, probs=probs,
                  weights=weights, w1_std=float(weights[:, 0].std(ddof=1)) if len(weights) > 1 else 0.0,
                  entropy_norm=float(entropy.mean() / math.log(model.num_experts)) if model.num_experts > 1 else 0.0,
                  expert_distance=float(np.mean(output_gaps)) if output_gaps else 0.0)
    return result


def initialize_class_prototypes(model, train_loader):
    """Use both training classes; never read validation labels for initialization."""
    model.eval()
    features, labels = [], []
    device = next(model.parameters()).device
    with torch.no_grad():
        for wsi, mri, label in train_loader:
            forward_batch(model, wsi, mri, device)
            features.append(model.latest_fh.detach())
            labels.extend(label.tolist())
        features = torch.cat(features)
        labels = torch.tensor(labels, device=device)
        for c in (0, 1):
            if not (labels == c).any():
                raise ValueError("Training split requires both classes")
            center = lorentz_mean(features[labels == c].unsqueeze(0)).squeeze(0)
            model.class_prototypes[c].copy_(center[1:])


def load_checkpoint(path, model):
    kwargs = dict(map_location=next(model.parameters()).device)
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = True
    checkpoint = torch.load(path, **kwargs)
    if checkpoint.get("model_version") != MODEL_VERSION:
        raise ValueError("Checkpoint is not a HyperMoE 1110 model; retraining is required")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return checkpoint


def auc_interval(labels, probs, seed, repetitions=2000):
    """Patient bootstrap conditional on the fitted/selected models."""
    rng = np.random.default_rng(seed)
    labels, probs = np.asarray(labels), np.asarray(probs)
    aucs = []
    for _ in range(repetitions):
        index = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[index])) == 2:
            aucs.append(roc_auc_score(labels[index], probs[index]))
    return [float(v) for v in np.quantile(aucs, [0.025, 0.975])]


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name, default in (("results_dir", config.RESULTS_DIR), ("split_dir", config.SPLIT_DIR),
                          ("wsi_dir", config.WSI_DIR), ("mri_dir", config.MRI_DIR)):
        p.add_argument(f"--{name}", default=default)
    p.add_argument("--test_name", default="hyper_1110")
    p.add_argument("--task", default="hyper_final", choices=["hyper", "hyper_new", "hyper_final", "wsi", "mri"])
    for name, default in (("max_epochs", config.DEFAULT_EPOCHS), ("accum_steps", config.DEFAULT_ACCUM_STEPS),
                          ("num_experts", config.NUM_EXPERTS), ("expert_hidden", 0), ("warmup_epochs", 10),
                          ("seed", config.SEED), ("patience", config.PATIENCE), ("lr_patience", config.LR_PATIENCE),
                          ("num_workers", 4)):
        p.add_argument(f"--{name}", type=int, default=default)
    for name, default in (("lr", config.DEFAULT_LR), ("reg", config.DEFAULT_WEIGHT_DECAY),
                          ("lambda_hyp", 0.1), ("lambda_ent", 0.01), ("lambda_cone", 0.01),
                          ("lambda_proto_sep", 0.01),
                          ("bal_weight", config.MOE_BALANCE_LAMBDA), ("threshold", 0.5)):
        p.add_argument(f"--{name}", type=float, default=default)
    p.add_argument("--class_weights", default=",".join(map(str, config.CLASS_WEIGHTS)))
    p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    return p


def main():
    script_start = time.time()
    args = build_parser().parse_args()
    folds = [int(f) for f in args.folds.split(",")]
    cw = [float(v) for v in args.class_weights.split(",")]
    if not folds or len(set(folds)) != len(folds) or any(f not in range(5) for f in folds):
        raise ValueError("--folds must contain distinct fold indices in 0..4")
    if len(cw) != 2 or any(not math.isfinite(v) or v <= 0 for v in cw):
        raise ValueError("--class_weights requires two positive finite values")
    if min(args.max_epochs, args.accum_steps, args.num_experts) < 1 or args.expert_hidden < 0:
        raise ValueError("Epochs, accumulation and expert count must be positive")
    if args.num_workers < 0 or args.warmup_epochs < 0 or args.lr_patience < 0 or args.patience <= args.lr_patience + 1:
        raise ValueError("Invalid workers/warmup or patience must allow a step after LR reduction")
    if (not 0 <= args.threshold <= 1 or not math.isfinite(args.lr) or not math.isfinite(args.reg)
            or args.lr <= 0 or args.reg < 0):
        raise ValueError("Invalid threshold, lr or regularization")
    if any(not math.isfinite(v) or v < 0 for v in
           (args.lambda_hyp, args.lambda_ent, args.lambda_cone,
            args.lambda_proto_sep, args.bal_weight)):
        raise ValueError("Loss weights must be finite and nonnegative")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    # Reject overlapping validation patients before calling the result OOF.
    seen = set()
    for fold in folds:
        df = load_split(args.split_dir, fold)
        val_ids = set(df.loc[df.val, "case_id"])
        if seen.intersection(val_ids):
            raise ValueError("Validation patients overlap across folds; pooled OOF is invalid")
        seen.update(val_ids)
    save_dir = Path(args.results_dir) / args.task / args.test_name
    save_dir.mkdir(parents=True, exist_ok=True)
    record = dict(vars(args), model_version=MODEL_VERSION, resolved_device=str(device),
                  evaluation_role="internal_model_selection_validation",
                  threshold_policy="validation_f1_at_auc_selected_epoch_per_fold",
                  torch_version=str(torch.__version__))
    for field in ("results_dir", "split_dir", "wsi_dir", "mri_dir"):
        record[field] = "<configured at runtime>"
    record["paths_redacted"] = True
    source = Path(__file__).resolve().parent
    record["source_sha256"] = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in source.rglob("*.py") if "tests" not in p.parts}
    (save_dir / "config.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("################ Settings ##################")
    print(f"task:  {args.task}")
    print(f"max_epochs:  {args.max_epochs}")
    print("results_dir:  <configured at runtime>")
    print(f"lr:  {args.lr}")
    print(f"experiment:  {args.test_name}")
    print(f"reg:  {args.reg}")
    print(f"accum_steps: {args.accum_steps}")
    print("Loss: CrossEntropy (weighted, reduction='none' -> mean)")
    print(f"lambda_hyp (Hyperbolic Prototype Classification): {args.lambda_hyp}")
    print(f"lambda_ent (Entailment): {args.lambda_ent}")
    print(f"lambda_cone (Cone): {args.lambda_cone}")
    print(f"lambda_proto_sep (Prototype Separation): {args.lambda_proto_sep}")
    print(f"bal_weight (window-level): {args.bal_weight}")
    print(f"warmup: {args.warmup_epochs} epochs (lambdas linearly from 0 to set value)")
    print(f"num_experts: {args.num_experts} | expert_hidden: {args.expert_hidden or 512}")
    print(f"seed:  {args.seed}")
    print(f"device:  {device}")
    print("threshold: validation F1 adaptive; AUC selects checkpoints")
    print(f"Model: UnifiedHierarchyMoE (dim={config.FEATURE_DIM}, version={MODEL_VERSION})")
    print(f"start script at: {time.ctime(script_start)}")
    summaries, pooled = [], []
    for fold in folds:
        seed = args.seed + fold
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        train, val, _ = get_dataloader(fold, args.split_dir, args.wsi_dir, args.mri_dir, args.num_workers, args.task)
        if train.dataset.df.label.nunique() != 2 or val.dataset.df.label.nunique() != 2:
            raise ValueError("AUC-based CV requires both classes in train and validation")
        pos_rate = train.dataset.df.label.mean()
        norm = cw[0] * (1 - pos_rate) + cw[1] * pos_rate
        class_weights = torch.tensor(cw, device=device, dtype=torch.float32) / norm
        model = UnifiedHierarchyMoE(config.FEATURE_DIM, args.num_experts, args.task, args.expert_hidden or None).to(device)
        print(f"Fold {fold} params: {sum(p.numel() for p in model.parameters()):,} "
              f"(num_experts={args.num_experts}, expert_hidden={args.expert_hidden or 512})")
        print(f"------------- Training Fold {fold}! -------------")
        print(f"class_weights raw={cw} -> normalized="
              f"{[round(float(x), 4) for x in class_weights.tolist()]} (train pos_rate={pos_rate:.4f})")
        print(f"Train: {len(train.dataset)} samples | Val: {len(val.dataset)} samples")
        print("Initializing class prototypes from training features...")
        initialize_class_prototypes(model, train)
        with torch.no_grad():
            print("Prototypes: class 0 norm = {:.3f}, class 1 norm = {:.3f}".format(
                model.class_prototypes[0].norm().item(), model.class_prototypes[1].norm().item()))
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.reg)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=args.lr_patience, threshold=0.0, threshold_mode="abs")
        best_auc, counter, best_epoch = -math.inf, 0, 0
        history = []
        for epoch in range(1, args.max_epochs + 1):
            start = time.perf_counter()
            ramp = min(1.0, (epoch - 1) / args.warmup_epochs) if args.warmup_epochs else 1.0
            train_result = run_epoch(model, train, optimizer, class_weights=class_weights,
                                     accum_steps=args.accum_steps, threshold=args.threshold,
                                     lambda_hyp=args.lambda_hyp * ramp, lambda_ent=args.lambda_ent * ramp,
                                     lambda_cone=args.lambda_cone * ramp,
                                     lambda_proto_sep=args.lambda_proto_sep,
                                     bal_weight=args.bal_weight)
            val_result = run_epoch(model, val, class_weights=class_weights, threshold=args.threshold)
            adaptive_threshold = get_best_threshold(
                val_result["labels"], val_result["probs"], fallback=args.threshold)
            val_result["metrics"] = calculate_metrics(
                val_result["labels"], val_result["probs"], adaptive_threshold)
            auc = val_result["metrics"]["auc"]
            scheduler.step(auc)
            if auc > best_auc:
                best_auc, counter, best_epoch = auc, 0, epoch
                torch.save(dict(model_version=MODEL_VERSION, model_state=model.state_dict(), fold=fold,
                                epoch=epoch, val_auc=auc, threshold=adaptive_threshold),
                           save_dir / f"fold{fold}_best.pth")
            else:
                counter += 1
            row = dict(epoch=epoch, ramp=ramp, threshold=adaptive_threshold,
                       lr=optimizer.param_groups[0]["lr"], patience=counter)
            for prefix, result in (("train", train_result), ("val", val_result)):
                for key in ("loss", "cls", "bal", "hyp", "ent", "cone", "proto_sep",
                            "w1_std", "entropy_norm", "expert_distance"):
                    row[f"{prefix}_{key}"] = result[key]
                row.update({f"{prefix}_{k}": v for k, v in result["metrics"].items()})
            history.append(row)
            pd.DataFrame(history).to_csv(save_dir / f"fold{fold}_history.csv", index=False)
            train_counts = classification_counts(
                train_result["labels"], train_result["probs"], args.threshold)
            val_counts = classification_counts(
                val_result["labels"], val_result["probs"], adaptive_threshold)
            print(f"Epoch: {epoch}, train_loss: {train_result['loss']:.4f} | "
                  f"cls={train_result['cls']:.4f} bal={train_result['bal']:.4f} "
                  f"hyp={train_result['hyp']:.4f} ent={train_result['ent']:.4f} "
                      f"cone={train_result['cone']:.4f} proto_sep={train_result['proto_sep']:.4f}")
            for label in (0, 1):
                correct = train_counts[f"class_{label}_correct"]
                total = train_counts[f"class_{label}_total"]
                print(f"class {label}: acc {correct / total if total else 0.0:.4f}, "
                      f"correct {correct}/{total}")
            print(f"Overall: acc {train_result['metrics']['acc']:.4f}, "
                  f"correct {train_counts['correct']}/{train_counts['total']}")
            print(f"Val Set (Threshold {adaptive_threshold:.3f}): val_loss={val_result['loss']:.4f} "
                  f"val_error={1.0 - val_result['metrics']['acc']:.4f} auc={auc:.4f} | "
                  f"cls={val_result['cls']:.4f} bal={val_result['bal']:.4f}")
            for label in (0, 1):
                correct = val_counts[f"class_{label}_correct"]
                total = val_counts[f"class_{label}_total"]
                print(f"class {label}: acc {correct / total if total else 0.0:.4f}, "
                      f"correct {correct}/{total}")
            print(f"Overall: acc {val_result['metrics']['acc']:.4f}, "
                  f"correct {val_counts['correct']}/{val_counts['total']}")
            print(f"Epoch time: {time.perf_counter()-start:.4f} | "
                  f"LR: {row['lr']:.6f} | Patience: {counter}/{args.patience}\n")
            if counter >= args.patience:
                print(f"  Early stop at Epoch {epoch}. Best Epoch: {best_epoch}\n")
                break
        checkpoint = load_checkpoint(save_dir / f"fold{fold}_best.pth", model)
        final = run_epoch(model, val, class_weights=class_weights, threshold=checkpoint["threshold"])
        predictions = pd.DataFrame(dict(case_id=val.dataset.df.case_id.tolist(), fold=fold,
                                        label=final["labels"], prob=final["probs"]))
        predictions["threshold"] = checkpoint["threshold"]
        predictions["pred"] = (predictions.prob >= checkpoint["threshold"]).astype(int)
        for i in range(args.num_experts):
            predictions[f"w{i+1}"] = final["weights"][:, i]
        predictions.drop(columns="case_id").to_csv(
            save_dir / f"s_{fold}_val_predictions.csv", index=False)
        pooled.append(predictions)
        summary = dict(fold=fold, best_epoch=checkpoint["epoch"], threshold=checkpoint["threshold"],
                       n_train=len(train.dataset), n_val=len(val.dataset),
                       num_params=sum(p.numel() for p in model.parameters()))
        summary.update({f"val_{k}": final["metrics"][k] for k in METRICS})
        summaries.append(summary)
        pd.DataFrame(summaries).to_csv(save_dir / "fold_summary.csv", index=False)
        plot_history(history, save_dir / f"fold_{fold}_loss_curve.png", fold)
    summary = pd.DataFrame(summaries)
    pd.DataFrame([dict(metric=k, mean=summary[f"val_{k}"].mean(), std=summary[f"val_{k}"].std(ddof=1),
                       n_folds=len(summary)) for k in METRICS]).to_csv(save_dir / "summary_mean_std.csv", index=False)
    oof = pd.concat(pooled, ignore_index=True)
    if oof.case_id.duplicated().any():
        raise ValueError("Duplicate patient predictions in OOF")
    oof.drop(columns="case_id").to_csv(save_dir / "oof_predictions.csv", index=False)
    pooled_result = calculate_metrics(oof.label, oof.prob, y_pred=oof.pred)
    pooled_result.update(auc_ci95=auc_interval(oof.label, oof.prob, args.seed), n_patients=len(oof),
                         n_folds=len(folds), evaluation_role=record["evaluation_role"],
                         threshold_policy=record["threshold_policy"],
                         ci_scope="patient bootstrap conditional on selected models; excludes selection/training uncertainty")
    (save_dir / "pooled_metrics.json").write_text(json.dumps(pooled_result, indent=2), encoding="utf-8")
    total_seconds = time.time() - script_start
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"end script at: {time.ctime()}\n")
    print("\nTraining complete; results saved to the configured output directory")
    print(f"Total script runtime: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds")


def plot_history(history, path, fold):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(data.epoch, data.train_cls, label="Train CE")
    ax.plot(data.epoch, data.val_cls, label="Validation CE")
    ax.plot(data.epoch, data.train_loss, "--", label="Train total (includes geometry/balance)")
    ax.set(xlabel="Epoch (1-based)", ylabel="Loss", title=f"Fold {fold}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
