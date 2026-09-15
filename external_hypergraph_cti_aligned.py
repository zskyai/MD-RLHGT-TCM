from __future__ import annotations

import argparse
import copy
import os
import traceback
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

import cold_start_split as css
import experiment_suite as es
import train
import external_htinet2_strict as base


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})


SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "checkpoints", "external_hypergraph_cti_aligned")
DEFAULT_VARIANT = "hypergraph_cti_aligned"
DEFAULT_CFG = {
    "data_path": os.path.join(SCRIPT_DIR, "processed", "hetero_graph.pt"),
    "epochs": 160,
    "batch_size": 512,
    "lr": 2e-4,
    "weight_decay": 1e-5,
    "hidden_dim": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "patience": 28,
    "min_epochs_before_es": 35,
    "device": "cuda",
    "deterministic": True,
}


class IngredientFusion(nn.Module):
    def __init__(self, input_dims: list[int], hidden_dim: int, dropout: float):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for dim in input_dims
        ])
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * len(input_dims), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(input_dims)),
        )

    def forward(self, views: list[torch.Tensor]) -> torch.Tensor:
        projected = [proj(view) for proj, view in zip(self.projections, views)]
        concat = torch.cat(projected, dim=-1)
        weights = torch.softmax(self.gate(concat), dim=-1)
        fused = 0.0
        for idx, proj_view in enumerate(projected):
            fused = fused + proj_view * weights[:, idx:idx + 1]
        return fused


class HypergraphLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.ing_self = nn.Linear(hidden_dim, hidden_dim)
        self.ing_msg = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.herb_self = nn.Linear(hidden_dim, hidden_dim)
        self.herb_msg = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.ing_norm = nn.LayerNorm(hidden_dim)
        self.herb_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ingredient_x: torch.Tensor, herb_x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if edge_index.numel() == 0:
            return ingredient_x, herb_x
        herb_idx = edge_index[0]
        ing_idx = edge_index[1]

        herb_deg = torch.bincount(herb_idx, minlength=herb_x.size(0)).float().clamp_min_(1.0)
        ing_deg = torch.bincount(ing_idx, minlength=ingredient_x.size(0)).float().clamp_min_(1.0)
        norm = torch.rsqrt(herb_deg[herb_idx] * ing_deg[ing_idx]).unsqueeze(-1)

        herb_agg = torch.zeros_like(herb_x)
        herb_agg.index_add_(0, herb_idx, ingredient_x[ing_idx] * norm)

        ing_agg = torch.zeros_like(ingredient_x)
        ing_agg.index_add_(0, ing_idx, herb_x[herb_idx] * norm)

        new_herb = self.herb_norm(herb_x + self.dropout(F.gelu(self.herb_self(herb_x) + self.herb_msg(herb_agg))))
        new_ing = self.ing_norm(ingredient_x + self.dropout(F.gelu(self.ing_self(ingredient_x) + self.ing_msg(ing_agg))))
        return new_ing, new_herb


class HypergraphCTIAlignedModel(nn.Module):
    def __init__(self, data, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        ingredient_store = data["ingredient"]
        target_store = data["target"]

        ingredient_dims = [
            int(getattr(ingredient_store, "bert_x", ingredient_store.x).size(1)),
            int(getattr(ingredient_store, "gpt_x", ingredient_store.x).size(1)),
            int(getattr(ingredient_store, "fp_x", ingredient_store.x).size(1)),
        ]
        if getattr(ingredient_store, "shape_x", None) is not None:
            ingredient_dims.append(int(ingredient_store.shape_x.size(1)))
        target_dims = [
            int(getattr(target_store, "text_x", target_store.x).size(1)),
            int(getattr(target_store, "seq_stats_x", target_store.x).size(1)) if getattr(target_store, "seq_stats_x", None) is not None else int(target_store.x.size(1)),
        ]

        self.ingredient_fusion = IngredientFusion(ingredient_dims, hidden_dim=hidden_dim, dropout=dropout)
        self.target_fusion = IngredientFusion(target_dims, hidden_dim=hidden_dim, dropout=dropout)
        self.herb_embedding = nn.Embedding(int(data["herb"].num_nodes), int(hidden_dim))
        self.layers = nn.ModuleList([
            HypergraphLayer(hidden_dim=hidden_dim, dropout=dropout)
            for _ in range(int(num_layers))
        ])
        self.target_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.edge_index = torch.empty((2, 0), dtype=torch.long)

    def set_incidence(self, hi_edge_index: torch.Tensor) -> None:
        self.edge_index = hi_edge_index.cpu() if hi_edge_index is not None else torch.empty((2, 0), dtype=torch.long)

    def encode(self, data) -> dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        ingredient_store = data["ingredient"]
        target_store = data["target"]

        ingredient_views = [
            getattr(ingredient_store, "bert_x", ingredient_store.x).to(device),
            getattr(ingredient_store, "gpt_x", ingredient_store.x).to(device),
            getattr(ingredient_store, "fp_x", ingredient_store.x).to(device),
        ]
        if getattr(ingredient_store, "shape_x", None) is not None:
            ingredient_views.append(ingredient_store.shape_x.to(device))
        target_views = [
            getattr(target_store, "text_x", target_store.x).to(device),
            getattr(target_store, "seq_stats_x", target_store.x).to(device) if getattr(target_store, "seq_stats_x", None) is not None else target_store.x.to(device),
        ]

        ingredient_x = self.ingredient_fusion(ingredient_views)
        target_x = self.target_head(self.target_fusion(target_views))
        herb_ids = torch.arange(int(data["herb"].num_nodes), device=device)
        herb_x = self.herb_embedding(herb_ids)
        edge_index = self.edge_index.to(device)
        for layer in self.layers:
            ingredient_x, herb_x = layer(ingredient_x, herb_x, edge_index)
        return {"ingredient": ingredient_x, "herb": herb_x, "target": target_x}

    def score_ingredient_target(self, ingredient_ids: torch.Tensor, target_ids: torch.Tensor, encoded: Optional[dict[str, torch.Tensor]] = None, data=None) -> torch.Tensor:
        if encoded is None:
            encoded = self.encode(data)
        ingredient_emb = encoded["ingredient"][ingredient_ids]
        target_emb = encoded["target"][target_ids]
        return (ingredient_emb * target_emb).sum(dim=-1)

    def forward(
        self,
        data,
        herb_ids: torch.Tensor,
        target_ids: torch.Tensor,
        herb_ing_padded: Optional[torch.Tensor] = None,
        herb_ing_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        return_aux: bool = False,
    ):
        encoded = self.encode(data)
        if herb_ing_padded is None or herb_ing_mask is None:
            raise ValueError("HypergraphCTIAlignedModel requires herb_ing_padded and herb_ing_mask for herb-target scoring.")

        padded = herb_ing_padded.to(next(self.parameters()).device)
        mask = herb_ing_mask.to(next(self.parameters()).device)
        target_emb = encoded["target"][target_ids]
        ing_emb = encoded["ingredient"][padded]
        ingredient_scores = (ing_emb * target_emb.unsqueeze(1)).sum(dim=-1)
        ingredient_scores = ingredient_scores.masked_fill(~mask, -1e9)
        attn = torch.softmax(ingredient_scores, dim=-1)
        attn = attn * mask.float()
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        scores = (attn * ingredient_scores.masked_fill(~mask, 0.0)).sum(dim=-1)
        if return_attention and return_aux:
            return scores, attn, {"ingredient_scores": ingredient_scores}
        if return_attention:
            return scores, attn
        if return_aux:
            return scores, {"ingredient_scores": ingredient_scores}
        return scores


def _sample_it_negatives(pos_ing: torch.Tensor, num_targets: int, generator: torch.Generator) -> torch.Tensor:
    return torch.randint(low=0, high=int(num_targets), size=(pos_ing.size(0),), generator=generator, dtype=torch.long)


def train_one_epoch(
    model: HypergraphCTIAlignedModel,
    msg_graph,
    pos_ing: torch.Tensor,
    pos_tgt: torch.Tensor,
    optimizer,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = max(1, int(pos_ing.size(0)))
    perm = torch.randperm(pos_ing.size(0), generator=generator)
    pos_ing = pos_ing[perm]
    pos_tgt = pos_tgt[perm]
    num_targets = int(msg_graph["target"].num_nodes)

    for start in range(0, pos_ing.size(0), batch_size):
        end = min(start + batch_size, pos_ing.size(0))
        batch_ing = pos_ing[start:end]
        batch_tgt = pos_tgt[start:end]
        neg_tgt = _sample_it_negatives(batch_ing, num_targets=num_targets, generator=generator)

        encoded = model.encode(msg_graph)
        pos_scores = model.score_ingredient_target(batch_ing.to(device), batch_tgt.to(device), encoded=encoded)
        neg_scores = model.score_ingredient_target(batch_ing.to(device), neg_tgt.to(device), encoded=encoded)
        loss = F.binary_cross_entropy_with_logits(
            torch.cat([pos_scores, neg_scores], dim=0),
            torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=0),
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.item()) * int(batch_ing.size(0))
    return float(total_loss / total_count)


def _summary(cfg: dict, val_metrics: dict, frozen_val_metrics: dict, test_metrics: dict, bundle_stats: dict, best_epoch: int, best_threshold: float, save_dir: str) -> dict:
    payload = {
        "variant": DEFAULT_VARIANT,
        "experiment_name": DEFAULT_VARIANT,
        "group": "EXTERNAL_SUPPLEMENTARY",
        "title": "Hypergraph-CTI Aligned Supplementary Baseline",
        "display_name": "Hypergraph-CTI-aligned",
        "source": "external_supplementary",
        "external_model": "Hypergraph-CTI",
        "alignment_mode": "project_only_compound_to_herb_aggregation",
        "save_dir": save_dir,
        "best_epoch": int(best_epoch),
        "best_threshold": float(best_threshold),
        "seed": int(cfg["seed"]),
        "threshold_policy": es.STRICT_THRESHOLD_POLICY,
        "split_id": es.STRICT_SPLIT_ID,
    }
    for prefix, metrics in (("val", val_metrics), ("frozen_val", frozen_val_metrics), ("test", test_metrics)):
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                payload[f"{prefix}_{key}"] = float(value)
    for key, value in (bundle_stats or {}).items():
        if isinstance(value, (int, float, bool)):
            payload[f"split_{key}"] = float(value) if isinstance(value, bool) else value
    return payload


def run_seed(cfg: dict, paths: dict[str, str], seed: int) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg["seed"] = int(seed)
    run_dir = os.path.join(paths["runs"], DEFAULT_VARIANT, f"seed_{int(seed)}")
    os.makedirs(run_dir, exist_ok=True)
    try:
        train.set_global_seed(int(seed), deterministic=bool(cfg.get("deterministic", True)))
        device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
        data = train.load_data(cfg["data_path"])
        data = train._apply_ingredient_feature_mode(data, "all")
        bundle = css.cold_start_split(
            data,
            val_ratio=es.STRICT_SPLIT_CFG["val_ratio"],
            test_ratio=es.STRICT_SPLIT_CFG["test_ratio"],
            neg_ratio=es.STRICT_SPLIT_CFG["neg_ratio"],
            hard_neg_ratio=0.0,
            pop_neg_ratio=0.0,
            it_mask_ratio=es.STRICT_SPLIT_CFG["it_mask_ratio"],
            seed=int(seed),
            strict_eval_neg_filter=es.STRICT_SPLIT_CFG["strict_eval_neg_filter"],
            val_hard_neg_ratio=0.0,
            val_pop_neg_ratio=0.0,
            test_hard_neg_ratio=0.0,
            test_pop_neg_ratio=0.0,
            min_pos_per_herb_val=es.STRICT_SPLIT_CFG["min_pos_per_herb_val"],
            min_pos_per_herb_test=es.STRICT_SPLIT_CFG["min_pos_per_herb_test"],
            max_fallback_per_herb=es.STRICT_SPLIT_CFG["max_fallback_per_herb"],
            use_it_mask=bool(es.STRICT_SPLIT_CFG["use_it_mask"]),
            filter_hi_to_train_herbs=bool(es.STRICT_SPLIT_CFG["filter_hi_to_train_herbs"]),
        )

        model = HypergraphCTIAlignedModel(
            data=bundle.msg_graph,
            hidden_dim=int(cfg["hidden_dim"]),
            num_layers=int(cfg["num_layers"]),
            dropout=float(cfg["dropout"]),
        ).to(device)
        hi_etype = next(et for et in bundle.msg_graph.edge_types if et[0] == "herb" and et[2] == "ingredient")
        it_etype = next(et for et in bundle.msg_graph.edge_types if et[0] == "ingredient" and et[2] == "target")
        model.set_incidence(bundle.msg_graph[hi_etype].edge_index)
        optimizer = AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
        it_edge = bundle.msg_graph[it_etype].edge_index
        pos_ing = it_edge[0].cpu()
        pos_tgt = it_edge[1].cpu()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))

        best_rank = -float("inf")
        best_state = None
        best_epoch = 0
        best_threshold = 0.5
        best_temperature = 1.0
        best_val_metrics = None
        patience = 0

        for epoch in range(1, int(cfg["epochs"]) + 1):
            loss = train_one_epoch(
                model=model,
                msg_graph=bundle.msg_graph,
                pos_ing=pos_ing,
                pos_tgt=pos_tgt,
                optimizer=optimizer,
                batch_size=int(cfg["batch_size"]),
                device=device,
                generator=generator,
            )
            raw_val = train.evaluate(
                model=model,
                msg_graph=bundle.msg_graph,
                pos_h=bundle.val_pos_h,
                pos_t=bundle.val_pos_t,
                neg_h=bundle.val_neg_h,
                neg_t=bundle.val_neg_t,
                herb_to_ings=bundle.herb_to_ings,
                batch_size=int(cfg["batch_size"]),
                device=device,
                threshold=None,
                return_threshold=True,
                return_logits=True,
                return_probs=True,
                temperature=1.0,
            )
            best_temperature = train.fit_temperature(
                logits=np.asarray(raw_val.get("logits", []), dtype=np.float32),
                labels=np.asarray(raw_val.get("labels", []), dtype=np.float32),
                steps=40,
                lr=0.05,
            )
            val_metrics = train.evaluate(
                model=model,
                msg_graph=bundle.msg_graph,
                pos_h=bundle.val_pos_h,
                pos_t=bundle.val_pos_t,
                neg_h=bundle.val_neg_h,
                neg_t=bundle.val_neg_t,
                herb_to_ings=bundle.herb_to_ings,
                batch_size=int(cfg["batch_size"]),
                device=device,
                threshold=None,
                return_threshold=True,
                return_probs=True,
                temperature=float(best_temperature),
            )
            rank_score = 0.45 * float(val_metrics["AUC"]) + 0.55 * float(val_metrics["AUPRC"])
            if rank_score > best_rank:
                best_rank = rank_score
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                best_epoch = int(epoch)
                best_threshold = float(val_metrics.get("best_threshold", 0.5))
                best_val_metrics = dict(val_metrics)
                patience = 0
                torch.save({"model_state_dict": best_state, "cfg": cfg}, os.path.join(run_dir, "best_model.pt"))
            else:
                patience += 1

            print(f"[Hypergraph-CTI-aligned] seed={seed} epoch={epoch:03d} loss={loss:.4f} val_auc={float(val_metrics['AUC']):.4f} val_f1={float(val_metrics['F1']):.4f}")
            if epoch >= int(cfg["min_epochs_before_es"]) and patience >= int(cfg["patience"]):
                break

        if best_state is None:
            raise RuntimeError("Hypergraph-CTI-aligned training did not produce a valid checkpoint.")
        model.load_state_dict(best_state)
        frozen_val = train.evaluate(
            model=model,
            msg_graph=bundle.msg_graph,
            pos_h=bundle.val_pos_h,
            pos_t=bundle.val_pos_t,
            neg_h=bundle.val_neg_h,
            neg_t=bundle.val_neg_t,
            herb_to_ings=bundle.herb_to_ings,
            batch_size=int(cfg["batch_size"]),
            device=device,
            threshold=float(best_threshold),
            return_threshold=True,
            return_probs=True,
            temperature=float(best_temperature),
        )
        test_metrics = train.evaluate(
            model=model,
            msg_graph=bundle.msg_graph,
            pos_h=bundle.test_pos_h,
            pos_t=bundle.test_pos_t,
            neg_h=bundle.test_neg_h,
            neg_t=bundle.test_neg_t,
            herb_to_ings=bundle.herb_to_ings,
            batch_size=int(cfg["batch_size"]),
            device=device,
            threshold=float(best_threshold),
            return_threshold=True,
            return_probs=True,
            temperature=float(best_temperature),
        )
        summary = _summary(cfg, best_val_metrics or {}, frozen_val, test_metrics, bundle.stats, best_epoch, best_threshold, run_dir)
        summary.update({"status": "ok", "strict_protocol_version": "hypergraph_cti_aligned_v1"})
        base._write_json(os.path.join(run_dir, "summary.json"), summary)
        return summary
    except Exception as exc:
        payload = {
            "variant": DEFAULT_VARIANT,
            "display_name": "Hypergraph-CTI-aligned",
            "source": "external_supplementary",
            "status": "failed",
            "seed": int(seed),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        base._write_json(os.path.join(run_dir, "error.json"), payload)
        return payload


def _collect_run_rows(paths: dict[str, str]) -> pd.DataFrame:
    variant_root = os.path.join(paths["runs"], DEFAULT_VARIANT)
    rows = []
    if not os.path.exists(variant_root):
        return pd.DataFrame()
    for entry in sorted(os.listdir(variant_root)):
        seed_dir = os.path.join(variant_root, entry)
        if not entry.startswith("seed_") or not os.path.isdir(seed_dir):
            continue
        payload = base._load_json(os.path.join(seed_dir, "summary.json")) or base._load_json(os.path.join(seed_dir, "error.json"))
        if payload is not None:
            rows.append(dict(payload))
    return pd.DataFrame(rows)


def _aggregate(run_df: pd.DataFrame, expected_seeds: list[int]) -> pd.DataFrame:
    if run_df.empty:
        return pd.DataFrame()
    frame = es._normalize_metric_columns(run_df.copy())
    ok_group = frame[frame["status"] == "ok"].copy()
    completed = sorted(pd.to_numeric(ok_group.get("seed", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).tolist())
    row = {
        "variant": DEFAULT_VARIANT,
        "display_name": "Hypergraph-CTI-aligned",
        "source": "external_supplementary",
        "status": "ok" if not ok_group.empty else "failed",
        "split_id": es.STRICT_SPLIT_ID,
        "threshold_policy": es.STRICT_THRESHOLD_POLICY,
        "expected_seeds": ",".join(str(x) for x in expected_seeds),
        "completed_seeds": ",".join(str(x) for x in completed),
        "expected_seed_count": len(expected_seeds),
        "completed_seed_count": len(completed),
        "strict_ready": completed == sorted(expected_seeds),
    }
    for metric in es.STRICT_METRIC_COLUMNS:
        values = pd.to_numeric(ok_group.get(metric, pd.Series(dtype=float)), errors="coerce").dropna()
        if values.empty:
            continue
        mean_value = float(values.mean())
        std_value = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        row[metric] = mean_value
        row[f"{metric}_std"] = std_value
        row[f"{metric}_ci95"] = float(1.96 * std_value / np.sqrt(len(values))) if len(values) > 1 else 0.0
        row[f"{metric}_mean_std"] = es._format_mean_std(mean_value, std_value)
    return pd.DataFrame([row])


def generate_report(paths: dict[str, str], seeds: list[int], min_seed_count: int) -> tuple[dict[str, str], pd.DataFrame]:
    run_df = _collect_run_rows(paths)
    aggregated_df = _aggregate(run_df, expected_seeds=seeds)
    outputs = {}
    run_csv = os.path.join(paths["root"], "hypergraph_cti_runs.csv")
    agg_csv = os.path.join(paths["root"], "hypergraph_cti_results.csv")
    run_df.to_csv(run_csv, index=False, encoding="utf-8-sig")
    aggregated_df.to_csv(agg_csv, index=False, encoding="utf-8-sig")
    outputs["hypergraph_cti_runs.csv"] = run_csv
    outputs["hypergraph_cti_results.csv"] = agg_csv

    env_path = os.path.join(paths["root"], "hypergraph_cti_environment.json")
    base._write_json(env_path, base._environment_snapshot())
    outputs["hypergraph_cti_environment.json"] = env_path

    tracked_files = ["train.py", "model.py", "cold_start_split.py", "external_hypergraph_cti_aligned.py"]
    rows = []
    for name in tracked_files:
        full = os.path.join(SCRIPT_DIR, name)
        if os.path.exists(full):
            rows.append({
                "file": name,
                "path": full,
                "sha256": base._file_sha256(full),
                "size_bytes": int(os.path.getsize(full)),
            })
    prov_path = os.path.join(paths["root"], "hypergraph_cti_code_provenance.json")
    base._write_json(prov_path, rows)
    outputs["hypergraph_cti_code_provenance.json"] = prov_path

    fairness_path = os.path.join(paths["root"], "hypergraph_cti_fairness_contract.json")
    base._write_json(fairness_path, {
        "external_model": "Hypergraph-CTI",
        "mode": "aligned supplementary baseline",
        "alignment_mode": "ingredient_target_training_with_herb_target_aggregation",
        "shared_split_id": es.STRICT_SPLIT_ID,
        "shared_threshold_policy": es.STRICT_THRESHOLD_POLICY,
        "notes": [
            "This is not claimed as a full original-paper reproduction.",
            "The original Hypergraph-CTI focuses on compound-target interaction; here it is aligned to the local herb-target task by ingredient aggregation.",
            "It is intended for supplementary comparison only, not for the strict main table.",
        ],
    })
    outputs["hypergraph_cti_fairness_contract.json"] = fairness_path

    audit_path = os.path.join(paths["root"], "hypergraph_cti_integrity_audit.csv")
    ok_group = run_df[run_df["status"] == "ok"].copy() if (not run_df.empty and "status" in run_df.columns) else pd.DataFrame()
    completed = sorted(pd.to_numeric(ok_group.get("seed", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).tolist())
    pd.DataFrame([{
        "variant": DEFAULT_VARIANT,
        "display_name": "Hypergraph-CTI-aligned",
        "expected_seeds": ",".join(str(x) for x in seeds),
        "completed_seeds": ",".join(str(x) for x in completed),
        "complete": completed == sorted(seeds),
        "top_journal_ready": False,
        "supplementary_ready": completed == sorted(seeds) and len(completed) >= int(min_seed_count),
    }]).to_csv(audit_path, index=False, encoding="utf-8-sig")
    outputs["hypergraph_cti_integrity_audit.csv"] = audit_path
    return outputs, aggregated_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aligned supplementary Hypergraph-CTI baseline under the local HIT project protocol.")
    parser.add_argument("--mode", choices=["run", "report", "all"], default="all")
    parser.add_argument("--data-path", type=str, default=DEFAULT_CFG["data_path"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CFG["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CFG["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CFG["lr"])
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_CFG["weight_decay"])
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_CFG["hidden_dim"])
    parser.add_argument("--num-layers", type=int, default=DEFAULT_CFG["num_layers"])
    parser.add_argument("--dropout", type=float, default=DEFAULT_CFG["dropout"])
    parser.add_argument("--patience", type=int, default=DEFAULT_CFG["patience"])
    parser.add_argument("--min-epochs-before-es", type=int, default=DEFAULT_CFG["min_epochs_before_es"])
    parser.add_argument("--device", type=str, default=DEFAULT_CFG["device"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--min-seed-count", type=int, default=1)
    return parser.parse_args()


def build_cfg(args: argparse.Namespace) -> dict:
    cfg = copy.deepcopy(DEFAULT_CFG)
    cfg.update({
        "data_path": args.data_path,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "hidden_dim": int(args.hidden_dim),
        "num_layers": int(args.num_layers),
        "dropout": float(args.dropout),
        "patience": int(args.patience),
        "min_epochs_before_es": int(args.min_epochs_before_es),
        "device": str(args.device),
    })
    return cfg


def main() -> None:
    args = parse_args()
    cfg = build_cfg(args)
    seeds = base.normalize_seeds(seeds=args.seeds, seed=args.seed)
    paths = base.ensure_dirs(args.output_root)
    if args.mode in ("run", "all"):
        base._write_json(os.path.join(paths["root"], "suite_config.json"), {
            "variant": DEFAULT_VARIANT,
            "expected_seeds": seeds,
            "split_id": es.STRICT_SPLIT_ID,
            "threshold_policy": es.STRICT_THRESHOLD_POLICY,
            "data_path": cfg["data_path"],
            "min_top_journal_seed_count": int(args.min_seed_count),
        })
        for seed in seeds:
            print(f"\n{'=' * 72}\nRunning Hypergraph-CTI-aligned supplementary baseline | seed={seed}\n{'=' * 72}")
            result = run_seed(cfg, paths, int(seed))
            if result.get("status") == "ok":
                print(f"[Done] seed={seed} AUC={float(result.get('test_AUC', np.nan)):.4f} F1={float(result.get('test_F1', np.nan)):.4f}")
            else:
                print(f"[Failed] seed={seed}: {result.get('error', 'Unknown error')}")
    if args.mode in ("report", "all"):
        outputs, _ = generate_report(paths, seeds=seeds, min_seed_count=int(args.min_seed_count))
        print("\nSaved outputs:")
        for _, path in sorted(outputs.items()):
            print(f"  {path}")


if __name__ == "__main__":
    main()
