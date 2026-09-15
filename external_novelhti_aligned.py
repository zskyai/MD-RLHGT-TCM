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
DEFAULT_OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "checkpoints", "external_novelhti_aligned")
DEFAULT_VARIANT = "novelhti_aligned"
DEFAULT_SEEDS = [42]
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


class MultiViewFusion(nn.Module):
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

    def forward(self, views: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        projected = [proj(view) for proj, view in zip(self.projections, views)]
        concat = torch.cat(projected, dim=-1)
        weights = torch.softmax(self.gate(concat), dim=-1)
        fused = 0.0
        for idx, proj_view in enumerate(projected):
            fused = fused + proj_view * weights[:, idx:idx + 1]
        return fused, weights


class NovelHTIAlignedModel(nn.Module):
    def __init__(self, data, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        herb_text_dim = int(getattr(data["herb"], "text_x", data["herb"].x).size(1))
        herb_struct = getattr(data["herb"], "struct_x", None)
        herb_struct_dim = int(herb_struct.size(1)) if herb_struct is not None else int(data["herb"].x.size(1))
        target_text_dim = int(getattr(data["target"], "text_x", data["target"].x).size(1))
        target_seq = getattr(data["target"], "seq_stats_x", None)
        target_seq_dim = int(target_seq.size(1)) if target_seq is not None else int(data["target"].x.size(1))

        self.herb_fusion = MultiViewFusion([herb_text_dim, herb_struct_dim], hidden_dim=hidden_dim, dropout=dropout)
        self.target_fusion = MultiViewFusion([target_text_dim, target_seq_dim], hidden_dim=hidden_dim, dropout=dropout)
        self.layers = nn.ModuleList([
            base.ResidualBipartiteLayer(hidden_dim=hidden_dim, dropout=dropout)
            for _ in range(int(num_layers))
        ])
        self.herb_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.target_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.ht_edge_index = torch.empty((2, 0), dtype=torch.long)

    def set_ht_graph(self, train_pos_h: torch.Tensor, train_pos_t: torch.Tensor) -> None:
        if train_pos_h.numel() == 0:
            self.ht_edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            self.ht_edge_index = torch.stack([train_pos_h.cpu(), train_pos_t.cpu()], dim=0)

    def encode(self, data) -> dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        herb_text = getattr(data["herb"], "text_x", data["herb"].x).to(device)
        herb_struct = getattr(data["herb"], "struct_x", data["herb"].x).to(device)
        target_text = getattr(data["target"], "text_x", data["target"].x).to(device)
        target_seq = getattr(data["target"], "seq_stats_x", data["target"].x).to(device)

        herb_x, herb_view_weights = self.herb_fusion([herb_text, herb_struct])
        target_x, target_view_weights = self.target_fusion([target_text, target_seq])
        edge_index = self.ht_edge_index.to(device)
        for layer in self.layers:
            herb_x, target_x = layer(herb_x, target_x, edge_index)

        herb_x = self.herb_head(herb_x)
        target_x = self.target_head(target_x)
        return {
            "herb": herb_x,
            "target": target_x,
            "herb_view_weights": herb_view_weights,
            "target_view_weights": target_view_weights,
        }

    def forward(
        self,
        data,
        herb_ids: torch.Tensor,
        target_ids: torch.Tensor,
        herb_ing_padded=None,
        herb_ing_mask=None,
        return_attention: bool = False,
        return_aux: bool = False,
    ):
        encoded = self.encode(data)
        herb_emb = encoded["herb"][herb_ids]
        target_emb = encoded["target"][target_ids]
        scores = (herb_emb * target_emb).sum(dim=-1)
        aux = {
            "herb_view_weights": encoded["herb_view_weights"][herb_ids],
            "target_view_weights": encoded["target_view_weights"][target_ids],
        }
        if return_attention and return_aux:
            return scores, None, aux
        if return_attention:
            return scores, None
        if return_aux:
            return scores, aux
        return scores


def build_neg_sampler(bundle, data, seed: int):
    it_etype = next(et for et in bundle.msg_graph.edge_types if et[0] == "ingredient" and et[2] == "target")
    train_it_edge = bundle.msg_graph[it_etype].edge_index
    train_ing_to_tgts = {}
    for idx in range(train_it_edge.size(1)):
        ing = int(train_it_edge[0, idx].item())
        tgt = int(train_it_edge[1, idx].item())
        train_ing_to_tgts.setdefault(ing, set()).add(tgt)
    train_ht_set = set(zip(bundle.train_pos_h.tolist(), bundle.train_pos_t.tolist()))
    return css.OnlineHTNegSampler(
        all_pos_global=train_ht_set,
        herb_to_ings=bundle.herb_to_ings,
        ing_to_tgts=train_ing_to_tgts,
        num_tgts=int(data["target"].num_nodes),
        hard_ratio=0.0,
        pop_ratio=0.0,
        seed=int(seed),
    )


def train_one_epoch(
    model: NovelHTIAlignedModel,
    msg_graph,
    train_pos_h: torch.Tensor,
    train_pos_t: torch.Tensor,
    neg_sampler,
    optimizer,
    batch_size: int,
    device: torch.device,
    perm_generator: Optional[torch.Generator] = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = max(1, int(train_pos_h.size(0)))
    if perm_generator is not None:
        perm = torch.randperm(train_pos_h.size(0), generator=perm_generator)
    else:
        perm = torch.randperm(train_pos_h.size(0))
    train_h = train_pos_h[perm]
    train_t = train_pos_t[perm]

    for start in range(0, train_h.size(0), batch_size):
        end = min(start + batch_size, train_h.size(0))
        batch_h = train_h[start:end]
        batch_t = train_t[start:end]
        neg_h, neg_t = neg_sampler.sample(batch_h, batch_t, neg_ratio=1.0)

        all_h = torch.cat([batch_h, neg_h])
        all_t = torch.cat([batch_t, neg_t])
        labels = torch.cat([
            torch.ones(batch_h.size(0)),
            torch.zeros(neg_h.size(0)),
        ]).to(device)

        scores = model(msg_graph, all_h.to(device), all_t.to(device), None, None)
        loss = F.binary_cross_entropy_with_logits(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.item()) * int(batch_h.size(0))

    return float(total_loss / total_count)


def _summary(cfg: dict, val_metrics: dict, frozen_val_metrics: dict, test_metrics: dict, bundle_stats: dict, best_epoch: int, best_threshold: float, save_dir: str) -> dict:
    payload = {
        "variant": DEFAULT_VARIANT,
        "experiment_name": DEFAULT_VARIANT,
        "group": "EXTERNAL_SUPPLEMENTARY",
        "title": "NovelHTI Aligned Supplementary Baseline",
        "display_name": "NovelHTI-aligned",
        "source": "external_supplementary",
        "external_model": "NovelHTI",
        "alignment_mode": "project_only_without_extra_symptom_pathway_inputs",
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

        model = NovelHTIAlignedModel(
            data=bundle.msg_graph,
            hidden_dim=int(cfg["hidden_dim"]),
            num_layers=int(cfg["num_layers"]),
            dropout=float(cfg["dropout"]),
        ).to(device)
        model.set_ht_graph(bundle.train_pos_h, bundle.train_pos_t)
        optimizer = AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
        neg_sampler = build_neg_sampler(bundle, data, seed=int(seed))
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
            loss = train_one_epoch(
                model=model,
                msg_graph=bundle.msg_graph,
                train_pos_h=bundle.train_pos_h,
                train_pos_t=bundle.train_pos_t,
                neg_sampler=neg_sampler,
                optimizer=optimizer,
                batch_size=int(cfg["batch_size"]),
                device=device,
                perm_generator=perm_generator,
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

            print(f"[NovelHTI-aligned] seed={seed} epoch={epoch:03d} loss={loss:.4f} val_auc={float(val_metrics['AUC']):.4f} val_f1={float(val_metrics['F1']):.4f}")
            if epoch >= int(cfg["min_epochs_before_es"]) and patience >= int(cfg["patience"]):
                break

        if best_state is None:
            raise RuntimeError("NovelHTI-aligned training did not produce a valid checkpoint.")
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
        summary.update({"status": "ok", "strict_protocol_version": "novelhti_aligned_v1"})
        base._write_json(os.path.join(run_dir, "summary.json"), summary)
        return summary
    except Exception as exc:
        payload = {
            "variant": DEFAULT_VARIANT,
            "display_name": "NovelHTI-aligned",
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
        "display_name": "NovelHTI-aligned",
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
    run_csv = os.path.join(paths["root"], "novelhti_runs.csv")
    agg_csv = os.path.join(paths["root"], "novelhti_results.csv")
    run_df.to_csv(run_csv, index=False, encoding="utf-8-sig")
    aggregated_df.to_csv(agg_csv, index=False, encoding="utf-8-sig")
    outputs["novelhti_runs.csv"] = run_csv
    outputs["novelhti_results.csv"] = agg_csv

    env_path = os.path.join(paths["root"], "novelhti_environment.json")
    base._write_json(env_path, base._environment_snapshot())
    outputs["novelhti_environment.json"] = env_path

    tracked_files = ["train.py", "model.py", "cold_start_split.py", "external_novelhti_aligned.py"]
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
    prov_path = os.path.join(paths["root"], "novelhti_code_provenance.json")
    base._write_json(prov_path, rows)
    outputs["novelhti_code_provenance.json"] = prov_path

    fairness_path = os.path.join(paths["root"], "novelhti_fairness_contract.json")
    base._write_json(fairness_path, {
        "external_model": "NovelHTI",
        "mode": "aligned supplementary baseline",
        "alignment_mode": "project_only_without_symptom_or_pathway_side_inputs",
        "shared_split_id": es.STRICT_SPLIT_ID,
        "shared_threshold_policy": es.STRICT_THRESHOLD_POLICY,
        "notes": [
            "This is not claimed as a full original-paper reproduction.",
            "The local project does not provide the full symptom/pathway side information used by NovelHTI.",
            "This aligned baseline keeps the same strict evaluation protocol and uses only locally available herb/target views.",
        ],
    })
    outputs["novelhti_fairness_contract.json"] = fairness_path

    audit_path = os.path.join(paths["root"], "novelhti_integrity_audit.csv")
    ok_group = run_df[run_df["status"] == "ok"].copy() if (not run_df.empty and "status" in run_df.columns) else pd.DataFrame()
    completed = sorted(pd.to_numeric(ok_group.get("seed", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).tolist())
    pd.DataFrame([{
        "variant": DEFAULT_VARIANT,
        "display_name": "NovelHTI-aligned",
        "expected_seeds": ",".join(str(x) for x in seeds),
        "completed_seeds": ",".join(str(x) for x in completed),
        "complete": completed == sorted(seeds),
        "top_journal_ready": False,
        "supplementary_ready": completed == sorted(seeds) and len(completed) >= int(min_seed_count),
    }]).to_csv(audit_path, index=False, encoding="utf-8-sig")
    outputs["novelhti_integrity_audit.csv"] = audit_path
    return outputs, aggregated_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aligned supplementary NovelHTI baseline under the local HIT project protocol.")
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
            print(f"\n{'=' * 72}\nRunning NovelHTI-aligned supplementary baseline | seed={seed}\n{'=' * 72}")
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
