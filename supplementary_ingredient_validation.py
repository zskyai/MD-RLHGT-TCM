from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


@dataclass
class ValidationContext:
    project_root: str
    checkpoint_path: str
    data_path: str
    save_root: str
    threshold: float
    temperature: float
    device: torch.device
    train_mod: object
    model: torch.nn.Module
    msg_graph: object
    bundle: object
    batch_size: int


def _abs_path(project_root: str, value: str) -> str:
    if os.path.isabs(value):
        return value
    return os.path.normpath(os.path.join(project_root, value))


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _candidate_checkpoint_paths(project_root: str, run_dir: Optional[str]) -> List[str]:
    candidates: List[str] = []
    if run_dir:
        candidates.append(os.path.join(run_dir, "best_model.pt"))

    candidates.extend([
        os.path.join(project_root, "checkpoints", "best_model.pt"),
        os.path.join(project_root, "checkpoints", "ablation", "runs", "proposed_model", "best_model.pt"),
    ])

    ablation_run_root = os.path.join(project_root, "checkpoints", "ablation", "runs", "proposed_model")
    if os.path.isdir(ablation_run_root):
        seed_dirs = []
        for entry in os.listdir(ablation_run_root):
            full_path = os.path.join(ablation_run_root, entry)
            if entry.startswith("seed_") and os.path.isdir(full_path):
                seed_dirs.append(full_path)
        seed_dirs.sort()
        for seed_dir in seed_dirs:
            candidates.append(os.path.join(seed_dir, "best_model.pt"))

    seen = set()
    ordered = []
    for path in candidates:
        norm_path = os.path.normpath(path)
        if norm_path in seen:
            continue
        seen.add(norm_path)
        ordered.append(norm_path)
    return ordered


def _resolve_checkpoint_path(project_root: str, checkpoint_path: Optional[str], run_dir: Optional[str]) -> str:
    if checkpoint_path:
        resolved = os.path.normpath(checkpoint_path)
        if not os.path.exists(resolved):
            raise FileNotFoundError(
                "Checkpoint path does not exist.\n"
                f"Tried: {resolved}"
            )
        return resolved

    candidates = _candidate_checkpoint_paths(project_root, run_dir)
    existing = [path for path in candidates if os.path.exists(path)]
    if existing:
        existing.sort(key=lambda path: os.path.getmtime(path), reverse=True)
        return existing[0]

    raise FileNotFoundError(
        "Checkpoint not found. Expected one of the current project output locations.\n"
        "Server-side defaults inferred from experiment_suite.py are:\n"
        "  1. <project_root>/checkpoints/best_model.pt\n"
        "  2. <project_root>/checkpoints/ablation/runs/proposed_model/seed_<n>/best_model.pt\n"
        "You can also pass --checkpoint-path or --run-dir explicitly."
    )


def _apply_figure_style() -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for supplementary plotting")
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 600,
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save_figure(fig, out_dir: str, name: str) -> str:
    _ensure_dir(out_dir)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    return os.path.join(out_dir, f"{name}.pdf")


def _load_train_module(project_root: str):
    project_root = os.path.normpath(project_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    import train as train_mod  # type: ignore

    return train_mod


def _resolve_split_cfg(train_mod, epochs: int) -> dict:
    _stage_name, split_cfg = train_mod.resolve_task_stage(epoch=max(1, int(epochs)))
    return split_cfg


def _build_padded_ingredients(ingredient_lists: Sequence[Sequence[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(ingredient_lists)
    max_k = max((len(items) for items in ingredient_lists), default=1)
    padded = torch.zeros(batch_size, max_k, dtype=torch.long)
    mask = torch.zeros(batch_size, max_k, dtype=torch.bool)
    for row_idx, items in enumerate(ingredient_lists):
        if not items:
            continue
        current = list(items)
        padded[row_idx, : len(current)] = torch.tensor(current, dtype=torch.long)
        mask[row_idx, : len(current)] = True
    return padded, mask


def _compute_metrics(labels: np.ndarray, logits: np.ndarray, threshold: float, temperature: float) -> dict:
    safe_temp = float(np.clip(temperature, 1e-3, 10.0))
    probs = 1.0 / (1.0 + np.exp(-(logits / safe_temp)))
    preds = (probs >= float(threshold)).astype(int)

    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = float("nan")
    try:
        auprc = average_precision_score(labels, probs)
    except Exception:
        auprc = float("nan")

    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    return {
        "AUC": auc,
        "AUPRC": auprc,
        "F1": f1_score(labels, preds, zero_division=0),
        "Precision": precision,
        "Recall": recall,
        "ACC": float((preds == labels).mean()) if labels.size else 0.0,
        "PR_gap": abs(float(precision) - float(recall)),
        "threshold": float(threshold),
        "temperature": safe_temp,
        "pos_mean_logit": float(logits[labels > 0.5].mean()) if np.any(labels > 0.5) else float("nan"),
        "neg_mean_logit": float(logits[labels <= 0.5].mean()) if np.any(labels <= 0.5) else float("nan"),
    }


def _build_context(
    project_root: str,
    checkpoint_path: Optional[str],
    data_path: Optional[str],
    run_dir: Optional[str],
) -> ValidationContext:
    train_mod = _load_train_module(project_root)

    default_data = os.path.join(project_root, "processed", "hetero_graph.pt")
    checkpoint_path = _resolve_checkpoint_path(project_root, checkpoint_path, run_dir)
    data_path = os.path.normpath(data_path or default_data)
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            "Graph data not found. Expected the current project default from train.py:\n"
            f"  {default_data}\n"
            "If your server stores processed data elsewhere, pass --data-path explicitly.\n"
            f"Tried: {data_path}"
        )

    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    required_keys = ["cfg", "model_state_dict"]
    missing_keys = [key for key in required_keys if key not in checkpoint]
    if missing_keys:
        raise RuntimeError(
            "Incompatible checkpoint format for the current project code. "
            "Expected keys from the current train.py output: cfg, model_state_dict. "
            f"Missing: {', '.join(missing_keys)}. "
            "Please use a checkpoint produced by HIT(第十四版本-正在修改)."
        )
    cfg = dict(checkpoint["cfg"])
    cfg["data_path"] = data_path
    cfg["save_dir"] = os.path.dirname(checkpoint_path)
    cfg["rl_save_path"] = _abs_path(project_root, cfg.get("rl_save_path", "checkpoints/rl_agent.pt"))

    train_mod.reset_cfg()
    train_mod.merge_cfg(cfg)
    train_mod.set_global_seed(train_mod.CFG["seed"], deterministic=train_mod.CFG["deterministic"])

    device = torch.device(train_mod.CFG["device"] if torch.cuda.is_available() else "cpu")
    data = train_mod.load_data(train_mod.CFG["data_path"])
    data = train_mod._apply_ingredient_feature_mode(  # pylint: disable=protected-access
        data,
        train_mod.CFG.get("ingredient_feature_mode", "all"),
    )

    split_cfg = _resolve_split_cfg(train_mod, epochs=int(train_mod.CFG.get("epochs", 300)))
    bundle = train_mod.cold_start_split(
        data,
        val_ratio=train_mod.CFG["val_ratio"],
        test_ratio=train_mod.CFG["test_ratio"],
        neg_ratio=train_mod.CFG["neg_ratio"],
        hard_neg_ratio=split_cfg["val_hard_neg_ratio"],
        pop_neg_ratio=split_cfg["val_pop_neg_ratio"],
        it_mask_ratio=split_cfg["it_mask_ratio"],
        seed=train_mod.CFG["seed"],
        strict_eval_neg_filter=split_cfg["strict_eval_neg_filter"],
        val_hard_neg_ratio=split_cfg["val_hard_neg_ratio"],
        val_pop_neg_ratio=split_cfg["val_pop_neg_ratio"],
        test_hard_neg_ratio=split_cfg["test_hard_neg_ratio"],
        test_pop_neg_ratio=split_cfg["test_pop_neg_ratio"],
        min_pos_per_herb_val=split_cfg["min_pos_per_herb_val"],
        min_pos_per_herb_test=split_cfg["min_pos_per_herb_test"],
        max_fallback_per_herb=split_cfg["max_fallback_per_herb"],
        use_it_mask=bool(train_mod.CFG.get("use_it_mask", True)),
        filter_hi_to_train_herbs=bool(train_mod.CFG.get("filter_hi_to_train_herbs", True)),
    )

    base_msg_graph = bundle.msg_graph.to(device)
    if hasattr(data["ingredient"], "pos") and data["ingredient"].pos is not None:
        base_msg_graph["ingredient"].pos = data["ingredient"].pos.to(device)

    it_etype = next(et for et in base_msg_graph.edge_types if et[0] == "ingredient" and et[2] == "target")
    train_it_edge = base_msg_graph[it_etype].edge_index
    num_ing = int(data["ingredient"].num_nodes)
    num_tgt = int(data["target"].num_nodes)
    ing_degree = torch.zeros(num_ing, dtype=torch.float32)
    tgt_degree = torch.zeros(num_tgt, dtype=torch.float32)
    for edge_idx in range(train_it_edge.size(1)):
        ing = int(train_it_edge[0, edge_idx].item())
        tgt = int(train_it_edge[1, edge_idx].item())
        if 0 <= ing < num_ing:
            ing_degree[ing] += 1.0
        if 0 <= tgt < num_tgt:
            tgt_degree[tgt] += 1.0
    ingredient_log_freq = torch.log1p(ing_degree)
    target_log_freq = torch.log1p(tgt_degree)

    ve_manager = train_mod.VirtualEdgeManager(base_graph=base_msg_graph, data=data)
    if len(ve_manager.edge_types) > 0:
        ve_manager.load_edge_params(
            saved_params=checkpoint.get("ve_params"),
            legacy_threshold=checkpoint.get("ve_threshold", train_mod.CFG["virtual_edge_threshold"]),
            legacy_topk=checkpoint.get("ve_topk", train_mod.CFG["virtual_edge_topk"]),
        )
        msg_graph = ve_manager.build_virtual_graph(device)
    else:
        msg_graph = base_msg_graph

    model_variant = str(train_mod.CFG.get("model_variant", "proposed")).lower()
    encoder_backbone = str(train_mod.CFG.get("encoder_backbone", "hgt")).lower()
    if model_variant == "vanilla_hgt":
        model = train_mod.VanillaHGTBaselineModel(
            node_types=msg_graph.node_types,
            edge_types=msg_graph.edge_types,
            in_dim_dict={nt: msg_graph[nt].x.size(1) for nt in msg_graph.node_types},
            hidden_dim=train_mod.CFG["hidden_dim"],
            num_layers=train_mod.CFG["num_layers"],
            num_heads=train_mod.CFG["num_heads"],
            dropout=train_mod.CFG["dropout"],
        ).to(device)
    else:
        model = train_mod.IngredientAwareHTModel(
            node_types=msg_graph.node_types,
            edge_types=msg_graph.edge_types,
            in_dim_dict={nt: msg_graph[nt].x.size(1) for nt in msg_graph.node_types},
            hidden_dim=train_mod.CFG["hidden_dim"],
            num_layers=train_mod.CFG["num_layers"],
            num_heads=train_mod.CFG["num_heads"],
            dropout=train_mod.CFG["dropout"],
            attention_activation=train_mod.CFG.get("attention_activation", "softmax"),
            attention_normalization=train_mod.CFG.get("attention_normalization", "length_mean"),
            freq_bias_beta=float(train_mod.CFG.get("freq_bias_beta", 0.0)),
            use_spatial_encoder=bool(train_mod.CFG.get("use_spatial_encoder", True)),
            spatial_dim=int(train_mod.CFG.get("spatial_dim", 64)),
            spatial_dropout=float(train_mod.CFG.get("spatial_dropout", 0.1)),
            spatial_max_atoms=int(train_mod.CFG.get("spatial_max_atoms", 64)),
            spatial_dist_hidden_dim=int(train_mod.CFG.get("spatial_dist_hidden_dim", 64)),
            spatial_centroid_hidden_dim=int(train_mod.CFG.get("spatial_centroid_hidden_dim", 16)),
            spatial_semantic_bias=float(train_mod.CFG.get("spatial_semantic_bias", 1.5)),
            encoder_backbone=encoder_backbone,
            decoder_use_ingredient_path=bool(train_mod.CFG.get("decoder_use_ingredient_path", True)),
            decoder_use_tri_attention=bool(train_mod.CFG.get("decoder_use_tri_attention", True)),
            decoder_use_global_path=bool(train_mod.CFG.get("decoder_use_global_path", True)),
        ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    if hasattr(model, "decoder") and hasattr(model.decoder, "set_frequency_bias"):
        model.decoder.set_frequency_bias(ingredient_log_freq, target_log_freq)
    model.eval()

    threshold = float(checkpoint.get("frozen_val_threshold", checkpoint.get("best_threshold", 0.5)))
    temperature = float(checkpoint.get("frozen_temperature", checkpoint.get("calibrated_temperature", 1.0)))
    save_root = os.path.join(os.path.dirname(checkpoint_path), "supplementary_ingredient_validation")

    return ValidationContext(
        project_root=os.path.normpath(project_root),
        checkpoint_path=checkpoint_path,
        data_path=data_path,
        save_root=save_root,
        threshold=threshold,
        temperature=temperature,
        device=device,
        train_mod=train_mod,
        model=model,
        msg_graph=msg_graph,
        bundle=bundle,
        batch_size=int(train_mod.CFG.get("batch_size", 512)),
    )


@torch.no_grad()
def _score_pairs(
    context: ValidationContext,
    herb_ids: Sequence[int],
    target_ids: Sequence[int],
    ingredient_lists: Sequence[Sequence[int]],
    return_attention: bool = False,
):
    logits_all: List[torch.Tensor] = []
    attn_all: List[np.ndarray] = []

    for start in range(0, len(herb_ids), context.batch_size):
        end = min(start + context.batch_size, len(herb_ids))
        batch_h = torch.tensor(herb_ids[start:end], dtype=torch.long, device=context.device)
        batch_t = torch.tensor(target_ids[start:end], dtype=torch.long, device=context.device)
        padded, mask = _build_padded_ingredients(ingredient_lists[start:end])
        padded = padded.to(context.device)
        mask = mask.to(context.device)

        if return_attention:
            logits, attn_tensor = context.model(
                context.msg_graph,
                batch_h,
                batch_t,
                padded,
                mask,
                return_attention=True,
                return_aux=False,
            )
            logits_all.append(logits.detach().cpu())
            for row_idx in range(attn_tensor.size(0)):
                valid_len = int(mask[row_idx].sum().item())
                attn_all.append(attn_tensor[row_idx, :valid_len].detach().cpu().numpy())
        else:
            logits = context.model(context.msg_graph, batch_h, batch_t, padded, mask)
            logits_all.append(logits.detach().cpu())

    logits_np = torch.cat(logits_all).numpy() if logits_all else np.array([], dtype=np.float32)
    if return_attention:
        return logits_np, attn_all
    return logits_np


def _collect_test_arrays(bundle, max_pairs: Optional[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos_h = bundle.test_pos_h.cpu().numpy().astype(np.int64)
    pos_t = bundle.test_pos_t.cpu().numpy().astype(np.int64)
    neg_h = bundle.test_neg_h.cpu().numpy().astype(np.int64)
    neg_t = bundle.test_neg_t.cpu().numpy().astype(np.int64)

    if max_pairs is not None and max_pairs > 0:
        pos_take = min(len(pos_h), int(np.ceil(max_pairs / 2)))
        neg_take = min(len(neg_h), int(np.floor(max_pairs / 2)))
        remaining = max_pairs - pos_take - neg_take
        if remaining > 0:
            extra_pos = min(len(pos_h) - pos_take, remaining)
            pos_take += extra_pos
            remaining -= extra_pos
        if remaining > 0:
            extra_neg = min(len(neg_h) - neg_take, remaining)
            neg_take += extra_neg
        pos_h = pos_h[:pos_take]
        pos_t = pos_t[:pos_take]
        neg_h = neg_h[:neg_take]
        neg_t = neg_t[:neg_take]

    herb_ids = np.concatenate([pos_h, neg_h]).astype(np.int64)
    target_ids = np.concatenate([pos_t, neg_t]).astype(np.int64)
    labels = np.concatenate([
        np.ones(len(pos_h), dtype=np.int64),
        np.zeros(len(neg_h), dtype=np.int64),
    ])
    return herb_ids, target_ids, labels


def run_baseline_validation(context: ValidationContext, max_pairs: Optional[int]) -> tuple[np.ndarray, np.ndarray, dict]:
    herb_ids, target_ids, labels = _collect_test_arrays(context.bundle, max_pairs=max_pairs)
    ingredient_lists = [context.bundle.herb_to_ings.get(int(herb_id), []) for herb_id in herb_ids]
    logits = _score_pairs(context, herb_ids, target_ids, ingredient_lists, return_attention=False)
    metrics = _compute_metrics(labels, logits, context.threshold, context.temperature)
    return labels, logits, metrics


def run_ingredient_replacement_validation(
    context: ValidationContext,
    max_pairs: Optional[int],
    trials: int,
    seed: int,
) -> dict:
    herb_ids, target_ids, labels = _collect_test_arrays(context.bundle, max_pairs=max_pairs)
    ingredient_pool = sorted({int(item) for values in context.bundle.herb_to_ings.values() for item in values})
    rng = np.random.default_rng(seed)
    trial_rows = []

    for trial_idx in range(int(trials)):
        replacement_map: Dict[int, List[int]] = {}
        for herb_id in sorted(set(int(item) for item in herb_ids)):
            original = list(context.bundle.herb_to_ings.get(herb_id, []))
            if not original:
                replacement_map[herb_id] = []
                continue
            sample_size = min(len(original), len(ingredient_pool))
            sampled = rng.choice(ingredient_pool, size=sample_size, replace=False)
            replacement_map[herb_id] = [int(value) for value in sampled.tolist()]

        ingredient_lists = [replacement_map.get(int(herb_id), []) for herb_id in herb_ids]
        logits = _score_pairs(context, herb_ids, target_ids, ingredient_lists, return_attention=False)
        metrics = _compute_metrics(labels, logits, context.threshold, context.temperature)
        metrics["trial"] = int(trial_idx)
        trial_rows.append(metrics)

    frame = pd.DataFrame(trial_rows)
    summary = {}
    for column in frame.columns:
        if column == "trial":
            continue
        summary[f"{column}_mean"] = _safe_float(frame[column].mean())
        summary[f"{column}_std"] = _safe_float(frame[column].std(ddof=0))
    return {"trials": trial_rows, "summary": summary}


def _select_drop_indices(attention: np.ndarray, drop_ratio: float, mode: str, rng: np.random.Generator) -> np.ndarray:
    item_count = int(attention.size)
    if item_count <= 1:
        return np.array([], dtype=np.int64)
    drop_count = int(np.floor(item_count * float(drop_ratio)))
    drop_count = max(1, drop_count)
    drop_count = min(drop_count, item_count - 1)

    if mode == "top":
        return np.argsort(-attention)[:drop_count]
    if mode == "bottom":
        return np.argsort(attention)[:drop_count]
    if mode == "random":
        order = np.arange(item_count)
        rng.shuffle(order)
        return order[:drop_count]
    raise ValueError(f"Unsupported drop mode: {mode}")


def run_attention_pruning_validation(
    context: ValidationContext,
    max_pairs: Optional[int],
    drop_ratio: float,
    seed: int,
) -> dict:
    herb_ids, target_ids, labels = _collect_test_arrays(context.bundle, max_pairs=max_pairs)
    base_ingredient_lists = [list(context.bundle.herb_to_ings.get(int(herb_id), [])) for herb_id in herb_ids]
    baseline_logits, baseline_attention = _score_pairs(
        context,
        herb_ids,
        target_ids,
        base_ingredient_lists,
        return_attention=True,
    )

    rng = np.random.default_rng(seed)
    results = {
        "baseline": _compute_metrics(labels, baseline_logits, context.threshold, context.temperature)
    }

    for mode in ("top", "bottom", "random"):
        pruned_lists: List[List[int]] = []
        for ingredient_list, attention in zip(base_ingredient_lists, baseline_attention):
            if not ingredient_list:
                pruned_lists.append([])
                continue
            drop_indices = set(int(item) for item in _select_drop_indices(np.asarray(attention), drop_ratio, mode, rng))
            kept = [ingredient for idx, ingredient in enumerate(ingredient_list) if idx not in drop_indices]
            if not kept:
                fallback_idx = int(np.argmin(attention)) if mode == "top" else int(np.argmax(attention))
                kept = [ingredient_list[fallback_idx]]
            pruned_lists.append(kept)

        logits = _score_pairs(context, herb_ids, target_ids, pruned_lists, return_attention=False)
        metrics = _compute_metrics(labels, logits, context.threshold, context.temperature)
        metrics["avg_logit_drop"] = float((baseline_logits - logits).mean())
        metrics["avg_pos_logit_drop"] = float((baseline_logits[labels > 0.5] - logits[labels > 0.5]).mean()) if np.any(labels > 0.5) else float("nan")
        metrics["avg_neg_logit_drop"] = float((baseline_logits[labels <= 0.5] - logits[labels <= 0.5]).mean()) if np.any(labels <= 0.5) else float("nan")
        results[mode] = metrics

    return results


def save_validation_outputs(output_dir: str, baseline_metrics: dict, replacement_results: dict, pruning_results: dict) -> None:
    _ensure_dir(output_dir)
    payload = {
        "baseline": baseline_metrics,
        "ingredient_replacement": replacement_results,
        "attention_pruning": pruning_results,
    }
    json_path = os.path.join(output_dir, "supplementary_ingredient_validation.json")
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    rows = []
    rows.append({"group": "baseline", **baseline_metrics})
    for mode_name, metrics in pruning_results.items():
        if mode_name == "baseline":
            continue
        rows.append({"group": f"attention_pruning:{mode_name}", **metrics})
    if "summary" in replacement_results:
        rows.append({"group": "ingredient_replacement", **replacement_results["summary"]})
    pd.DataFrame(rows).to_csv(
        os.path.join(output_dir, "supplementary_ingredient_validation.csv"),
        index=False,
        encoding="utf-8-sig",
    )


def plot_replacement_validation(
    output_dir: str,
    baseline_metrics: dict,
    replacement_results: dict,
) -> Optional[str]:
    if plt is None:
        return None
    trials = replacement_results.get("trials", [])
    if not trials:
        return None

    _apply_figure_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))

    metrics_to_plot = ["AUC", "AUPRC", "F1", "Precision"]
    baseline_values = [float(baseline_metrics.get(metric, np.nan)) for metric in metrics_to_plot]
    replacement_values = [float(replacement_results["summary"].get(f"{metric}_mean", np.nan)) for metric in metrics_to_plot]

    x = np.arange(len(metrics_to_plot))
    width = 0.36
    axes[0].bar(x - width / 2, baseline_values, width=width, color="#1F77B4", label="Baseline")
    axes[0].bar(x + width / 2, replacement_values, width=width, color="#D62728", label="Ingredient replacement")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics_to_plot)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Metric value")
    axes[0].set_title("Metric shift after ingredient replacement")
    axes[0].legend(frameon=False, loc="lower left")

    trial_ids = [int(row.get("trial", idx)) for idx, row in enumerate(trials)]
    trial_auc = [float(row.get("AUC", np.nan)) for row in trials]
    trial_f1 = [float(row.get("F1", np.nan)) for row in trials]
    axes[1].plot(trial_ids, trial_auc, marker="o", lw=1.4, color="#1F77B4", label="AUC")
    axes[1].plot(trial_ids, trial_f1, marker="s", lw=1.4, color="#D62728", label="F1")
    axes[1].axhline(float(baseline_metrics.get("AUC", np.nan)), color="#1F77B4", lw=0.9, ls="--", alpha=0.7)
    axes[1].axhline(float(baseline_metrics.get("F1", np.nan)), color="#D62728", lw=0.9, ls="--", alpha=0.7)
    axes[1].set_xlabel("Trial")
    axes[1].set_ylabel("Metric value")
    axes[1].set_title("Trial-level variability")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(frameon=False, loc="lower left")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout(w_pad=1.6)
    return _save_figure(fig, output_dir, "Figure_supplementary_ingredient_replacement")


def plot_attention_pruning_validation(
    output_dir: str,
    baseline_metrics: dict,
    pruning_results: dict,
) -> Optional[str]:
    if plt is None:
        return None

    _apply_figure_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))

    modes = ["top", "bottom", "random"]
    mode_labels = ["Top-attention drop", "Bottom-attention drop", "Random drop"]
    colors = ["#D62728", "#2CA02C", "#9467BD"]

    avg_pos_drops = [float(pruning_results.get(mode, {}).get("avg_pos_logit_drop", np.nan)) for mode in modes]
    avg_neg_drops = [float(pruning_results.get(mode, {}).get("avg_neg_logit_drop", np.nan)) for mode in modes]
    x = np.arange(len(modes))
    width = 0.36
    axes[0].bar(x - width / 2, avg_pos_drops, width=width, color="#D62728", label="Positive-pair logit drop")
    axes[0].bar(x + width / 2, avg_neg_drops, width=width, color="#1F77B4", label="Negative-pair logit drop")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(mode_labels, rotation=10)
    axes[0].set_ylabel("Average logit drop")
    axes[0].set_title("Logit sensitivity to ingredient pruning")
    axes[0].legend(frameon=False, loc="upper right")

    f1_values = [float(pruning_results.get(mode, {}).get("F1", np.nan)) for mode in modes]
    precision_values = [float(pruning_results.get(mode, {}).get("Precision", np.nan)) for mode in modes]
    baseline_f1 = float(baseline_metrics.get("F1", np.nan))
    baseline_precision = float(baseline_metrics.get("Precision", np.nan))
    axes[1].plot(mode_labels, f1_values, marker="o", lw=1.6, color="#1F77B4", label="F1")
    axes[1].plot(mode_labels, precision_values, marker="s", lw=1.6, color="#FF7F0E", label="Precision")
    axes[1].axhline(baseline_f1, color="#1F77B4", lw=0.9, ls="--", alpha=0.7)
    axes[1].axhline(baseline_precision, color="#FF7F0E", lw=0.9, ls="--", alpha=0.7)
    axes[1].set_ylabel("Metric value")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Decision quality after ingredient pruning")
    axes[1].tick_params(axis="x", rotation=10)
    axes[1].legend(frameon=False, loc="lower left")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout(w_pad=1.6)
    return _save_figure(fig, output_dir, "Figure_supplementary_attention_pruning")


def generate_supplementary_figures(
    output_dir: str,
    baseline_metrics: dict,
    replacement_results: dict,
    pruning_results: dict,
) -> Dict[str, Optional[str]]:
    figure_dir = os.path.join(output_dir, "figures")
    outputs = {
        "replacement_figure": plot_replacement_validation(figure_dir, baseline_metrics, replacement_results),
        "attention_pruning_figure": plot_attention_pruning_validation(figure_dir, baseline_metrics, pruning_results),
    }
    manifest_path = os.path.join(figure_dir, "supplementary_figure_manifest.json")
    _ensure_dir(figure_dir)
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(outputs, file, ensure_ascii=False, indent=2)
    outputs["manifest"] = manifest_path
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Supplementary ingredient validation without changing existing main experiments."
    )
    parser.add_argument("--project-root", default=os.path.dirname(os.path.abspath(__file__)), help="Project root directory.")
    parser.add_argument("--run-dir", default=None, help="Optional run directory such as checkpoints/ablation/runs/proposed_model/seed_42.")
    parser.add_argument("--checkpoint-path", default=None, help="Path to best_model.pt.")
    parser.add_argument("--data-path", default=None, help="Path to hetero_graph.pt.")
    parser.add_argument("--output-dir", default=None, help="Output directory for supplementary validation results.")
    parser.add_argument("--replace-trials", type=int, default=3, help="Number of ingredient replacement trials.")
    parser.add_argument("--drop-ratio", type=float, default=0.3, help="Ingredient drop ratio for attention pruning.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for perturbation experiments.")
    parser.add_argument("--max-pairs", type=int, default=None, help="Optional cap for quick smoke tests.")
    parser.add_argument("--skip-figures", action="store_true", help="Skip supplementary figure generation.")
    args = parser.parse_args()

    context = _build_context(
        project_root=args.project_root,
        checkpoint_path=args.checkpoint_path,
        data_path=args.data_path,
        run_dir=args.run_dir,
    )
    output_dir = args.output_dir or context.save_root

    _labels, _baseline_logits, baseline_metrics = run_baseline_validation(context, max_pairs=args.max_pairs)
    replacement_results = run_ingredient_replacement_validation(
        context,
        max_pairs=args.max_pairs,
        trials=args.replace_trials,
        seed=args.seed,
    )
    pruning_results = run_attention_pruning_validation(
        context,
        max_pairs=args.max_pairs,
        drop_ratio=args.drop_ratio,
        seed=args.seed,
    )
    save_validation_outputs(output_dir, baseline_metrics, replacement_results, pruning_results)
    figure_outputs = {}
    if not args.skip_figures:
        figure_outputs = generate_supplementary_figures(
            output_dir,
            baseline_metrics,
            replacement_results,
            pruning_results,
        )

    print("\n=== Supplementary Ingredient Validation ===")
    print(f"Checkpoint path: {context.checkpoint_path}")
    print(f"Data path: {context.data_path}")
    print(f"Output directory: {output_dir}")
    print(f"Baseline F1: {baseline_metrics['F1']:.4f}")
    print(f"Replacement AUC mean: {replacement_results['summary'].get('AUC_mean', float('nan')):.4f}")
    print(f"Top-drop avg pos logit drop: {pruning_results['top']['avg_pos_logit_drop']:.4f}")
    print(f"Bottom-drop avg pos logit drop: {pruning_results['bottom']['avg_pos_logit_drop']:.4f}")
    if figure_outputs:
        print(f"Figure manifest: {figure_outputs.get('manifest')}")


if __name__ == "__main__":
    main()

