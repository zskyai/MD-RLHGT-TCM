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
from model import IngredientAwareHTModel


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
DEFAULT_OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "checkpoints", "strong_herb_only_strict")
DEFAULT_VARIANT = "strong_herb_only_strict"
DEFAULT_CFG = {
    "data_path": os.path.join(SCRIPT_DIR, "processed", "hetero_graph.pt"),
    "epochs": 220,
    "batch_size": 512,
    "lr": 2e-4,
    "weight_decay": 5e-5,
    "hidden_dim": 128,
    "num_layers": 3,
    "num_heads": 4,
    "dropout": 0.2,
    "patience": 30,
    "min_epochs_before_es": 40,
    "device": "cuda",
    "deterministic": True,
    "use_spatial_encoder": True,
    "encoder_backbone": "hgt",
}


class StrongHerbOnlyModel(nn.Module):
    """
    Strong herb-only baseline:
    1. Use the same heterogeneous encoder as the main model.
    2. Aggregate herb ingredient embeddings into a target-independent herb super representation.
    3. Score herb-target pairs without target-conditioned ingredient attention.
    """

    def __init__(
        self,
        node_types,
        edge_types,
        in_dim_dict,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        use_spatial_encoder: bool = True,
        encoder_backbone: str = "hgt",
    ):
        super().__init__()
        self.base_model = IngredientAwareHTModel(
            node_types=node_types,
            edge_types=edge_types,
            in_dim_dict=in_dim_dict,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_spatial_encoder=use_spatial_encoder,
            encoder_backbone=encoder_backbone,
            decoder_use_ingredient_path=False,
            decoder_use_tri_attention=False,
            decoder_use_global_path=False,
        )
        self.herb_mix = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.target_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.herb_context_projector = self.herb_mix
        self.target_projector = self.target_proj
        self.scorer = nn.Bilinear(hidden_dim, hidden_dim, 1)

    def encode(self, data):
        return self.base_model.encode(data)

    def forward(
        self,
        data,
        herb_ids: torch.Tensor,
        target_ids: torch.Tensor,
        herb_ing_padded: torch.Tensor,
        herb_ing_mask: torch.Tensor,
        return_attention: bool = False,
        return_aux: bool = False,
    ):
        x_dict = self.encode(data)
        herb_emb = x_dict["herb"][herb_ids]
        target_emb = x_dict["target"][target_ids]
        ingredient_emb = x_dict["ingredient"]

        safe_idx = herb_ing_padded.clamp(min=0)
        ing_embs = ingredient_emb[safe_idx]
        mask = herb_ing_mask.unsqueeze(-1).float()
        ing_context = (ing_embs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        herb_super = self.herb_mix(torch.cat([herb_emb, ing_context], dim=-1))
        target_vec = self.target_proj(target_emb)
        scores = self.scorer(herb_super, target_vec).squeeze(-1)

        aux = {
            "herb_emb": herb_emb,
            "ingredient_context": ing_context,
            "herb_super": herb_super,
            "target_emb": target_vec,
        }
        if return_attention and return_aux:
            return scores, None, aux
        if return_attention:
            return scores, None
        if return_aux:
            return scores, aux
        return scores


def _build_summary(cfg: dict, val_metrics: dict, frozen_val_metrics: dict, test_metrics: dict, bundle_stats: dict, best_epoch: int, best_threshold: float, save_dir: str) -> dict:
    payload = {
        "variant": DEFAULT_VARIANT,
        "experiment_name": DEFAULT_VARIANT,
        "group": "CORE_CONTROL",
        "title": "Strong Herb-only Strict Baseline",
        "display_name": "Strong herb-only",
        "source": "internal_control",
        "save_dir": save_dir,
        "best_epoch": int(best_epoch),
        "best_threshold": float(best_threshold),
        "seed": int(cfg["seed"]),
        "threshold_policy": es.STRICT_THRESHOLD_POLICY,
        "split_id": es.STRICT_SPLIT_ID,
        "decoder_statement": "Target-independent herb super-representation without ingredient-mediated decoding",
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

        msg_graph = bundle.msg_graph.to(device)
        if hasattr(data["ingredient"], "pos") and data["ingredient"].pos is not None:
            msg_graph["ingredient"].pos = data["ingredient"].pos.to(device)

        model = StrongHerbOnlyModel(
            node_types=msg_graph.node_types,
            edge_types=msg_graph.edge_types,
            in_dim_dict={nt: msg_graph[nt].x.size(1) for nt in msg_graph.node_types},
            hidden_dim=int(cfg["hidden_dim"]),
            num_layers=int(cfg["num_layers"]),
            num_heads=int(cfg["num_heads"]),
            dropout=float(cfg["dropout"]),
            use_spatial_encoder=bool(cfg["use_spatial_encoder"]),
            encoder_backbone=str(cfg["encoder_backbone"]),
        ).to(device)

        optimizer = AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
        neg_sampler = base.build_neg_sampler(bundle, data, seed=int(seed))
        perm_generator = torch.Generator(device="cpu")
        perm_generator.manual_seed(int(seed))

        best_rank = -float("inf")
        best_state = None
        best_epoch = 0
        best_threshold = 0.5
        best_temperature = 1.0
        best_val_metrics = None
        patience = 0

        for epoch in range(1, int(cfg["epochs"]) + 1):
            train_loss = train.train_one_epoch(
                model=model,
                msg_graph=msg_graph,
                train_pos_h=bundle.train_pos_h,
                train_pos_t=bundle.train_pos_t,
                herb_to_ings=bundle.herb_to_ings,
                neg_sampler=neg_sampler,
                optimizer=optimizer,
                batch_size=int(cfg["batch_size"]),
                device=device,
                pos_weight=1.0,
                neg_weight=1.0,
                train_neg_ratio=1.0,
                hard_neg_ratio=0.0,
                perm_generator=perm_generator,
            )

            raw_val = train.evaluate(
                model=model,
                msg_graph=msg_graph,
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
                msg_graph=msg_graph,
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

            print(
                f"[StrongHerbOnly] seed={seed} epoch={epoch:03d} "
                f"loss={train_loss:.4f} val_auc={float(val_metrics['AUC']):.4f} "
                f"val_f1={float(val_metrics['F1']):.4f}"
            )
            if epoch >= int(cfg["min_epochs_before_es"]) and patience >= int(cfg["patience"]):
                break

        if best_state is None:
            raise RuntimeError("Strong herb-only training did not produce a valid checkpoint.")

        model.load_state_dict(best_state)
        frozen_val = train.evaluate(
            model=model,
            msg_graph=msg_graph,
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
            msg_graph=msg_graph,
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

        summary = _build_summary(
            cfg=cfg,
            val_metrics=best_val_metrics or {},
            frozen_val_metrics=frozen_val,
            test_metrics=test_metrics,
            bundle_stats=bundle.stats,
            best_epoch=best_epoch,
            best_threshold=best_threshold,
            save_dir=run_dir,
        )
        summary.update({"status": "ok", "strict_protocol_version": "strong_herb_only_v1"})
        base._write_json(os.path.join(run_dir, "summary.json"), summary)
        return summary
    except Exception as exc:
        payload = {
            "variant": DEFAULT_VARIANT,
            "display_name": "Strong herb-only",
            "source": "internal_control",
            "status": "failed",
            "seed": int(seed),
            "split_id": es.STRICT_SPLIT_ID,
            "threshold_policy": es.STRICT_THRESHOLD_POLICY,
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
        "display_name": "Strong herb-only",
        "source": "internal_control",
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
    run_csv = os.path.join(paths["root"], "strong_herb_only_runs.csv")
    agg_csv = os.path.join(paths["root"], "strong_herb_only_results.csv")
    run_df.to_csv(run_csv, index=False, encoding="utf-8-sig")
    aggregated_df.to_csv(agg_csv, index=False, encoding="utf-8-sig")
    outputs["strong_herb_only_runs.csv"] = run_csv
    outputs["strong_herb_only_results.csv"] = agg_csv

    env_path = os.path.join(paths["root"], "strong_herb_only_environment.json")
    base._write_json(env_path, base._environment_snapshot())
    outputs["strong_herb_only_environment.json"] = env_path

    fairness_path = os.path.join(paths["root"], "strong_herb_only_fairness_contract.json")
    base._write_json(fairness_path, {
        "baseline": "strong herb-only",
        "task_definition": "strict herb-level cold-start herb-target prediction",
        "shared_split_id": es.STRICT_SPLIT_ID,
        "shared_threshold_policy": es.STRICT_THRESHOLD_POLICY,
        "notes": [
            "Same strict split and evaluation policy as the main model.",
            "Same heterogeneous encoder family as the main model.",
            "Difference: ingredients are pooled into a target-independent herb super-representation, without explicit ingredient-mediated decoding.",
        ],
    })
    outputs["strong_herb_only_fairness_contract.json"] = fairness_path

    audit_path = os.path.join(paths["root"], "strong_herb_only_integrity_audit.csv")
    ok_group = run_df[run_df["status"] == "ok"].copy() if (not run_df.empty and "status" in run_df.columns) else pd.DataFrame()
    completed = sorted(pd.to_numeric(ok_group.get("seed", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).tolist())
    pd.DataFrame([{
        "variant": DEFAULT_VARIANT,
        "display_name": "Strong herb-only",
        "expected_seeds": ",".join(str(x) for x in seeds),
        "completed_seeds": ",".join(str(x) for x in completed),
        "complete": completed == sorted(seeds),
        "meets_min_seed_count": len(completed) >= int(min_seed_count),
    }]).to_csv(audit_path, index=False, encoding="utf-8-sig")
    outputs["strong_herb_only_integrity_audit.csv"] = audit_path
    return outputs, aggregated_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict strong herb-only baseline under the local HIT project protocol.")
    parser.add_argument("--mode", choices=["run", "report", "all"], default="all")
    parser.add_argument("--data-path", type=str, default=DEFAULT_CFG["data_path"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CFG["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CFG["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CFG["lr"])
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_CFG["weight_decay"])
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_CFG["hidden_dim"])
    parser.add_argument("--num-layers", type=int, default=DEFAULT_CFG["num_layers"])
    parser.add_argument("--num-heads", type=int, default=DEFAULT_CFG["num_heads"])
    parser.add_argument("--dropout", type=float, default=DEFAULT_CFG["dropout"])
    parser.add_argument("--patience", type=int, default=DEFAULT_CFG["patience"])
    parser.add_argument("--min-epochs-before-es", type=int, default=DEFAULT_CFG["min_epochs_before_es"])
    parser.add_argument("--device", type=str, default=DEFAULT_CFG["device"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--min-seed-count", type=int, default=1)
    parser.add_argument("--require-complete", action="store_true")
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
        "num_heads": int(args.num_heads),
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
            print(f"\n{'=' * 72}\nRunning strong herb-only strict baseline | seed={seed}\n{'=' * 72}")
            result = run_seed(cfg, paths, int(seed))
            if result.get("status") == "ok":
                print(f"[Done] seed={seed} AUC={float(result.get('test_AUC', np.nan)):.4f} F1={float(result.get('test_F1', np.nan)):.4f}")
            else:
                print(f"[Failed] seed={seed}: {result.get('error', 'Unknown error')}")
    if args.mode in ("report", "all"):
        outputs, aggregated_df = generate_report(paths, seeds=seeds, min_seed_count=int(args.min_seed_count))
        print("\nSaved outputs:")
        for _, path in sorted(outputs.items()):
            print(f"  {path}")
        if args.require_complete:
            if aggregated_df.empty or "strict_ready" not in aggregated_df.columns or "completed_seed_count" not in aggregated_df.columns:
                raise SystemExit("Strong herb-only report is incomplete: no strict-ready rows available.")
            ready_mask = aggregated_df["strict_ready"] & (aggregated_df["completed_seed_count"] >= int(args.min_seed_count))
            if int(ready_mask.sum()) != int(len(aggregated_df)):
                raise SystemExit("Strong herb-only report failed the completeness gate.")


if __name__ == "__main__":
    main()

