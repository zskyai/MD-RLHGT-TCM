from __future__ import annotations

import os
import json
import shutil
import argparse
import traceback
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None


SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "checkpoints", "ablation")

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


def build_variant_catalog() -> list[dict]:
    return [
        {
            "variant": "hgt_baseline",
            "group": "BASELINE",
            "display_name": "Vanilla HGT",
            "title": "Vanilla HGT",
            "order": 50,
            "compare": True,
            "backbone_compare": False,
            "overrides": {
                "model_variant": "vanilla_hgt",
                "encoder_backbone": "hgt",
                "use_rl_virtual_edge": False,
                "virtual_edge_types": [],
                "use_spatial_encoder": False,
                "decoder_use_ingredient_path": False,
                "decoder_use_tri_attention": False,
                "decoder_use_global_path": False,
                "use_ranking_loss": False,
                "use_contrastive_loss": False,
                "warmup_epochs": 0,
                "pos_weight": 1.0,
                "hard_neg_start": 0.0,
                "hard_neg_end": 0.0,
                "pop_neg_start": 0.0,
                "pop_neg_end": 0.0,
                "train_neg_ratio_start": 1.0,
                "train_neg_ratio_end": 1.0,
            },
        },
        {
            "variant": "static_similarity_hgt",
            "group": "BASELINE",
            "display_name": "Static Similarity HGT",
            "title": "Static Similarity HGT",
            "order": 60,
            "compare": True,
            "backbone_compare": False,
            "overrides": {
                "model_variant": "proposed",
                "encoder_backbone": "hgt",
                "use_rl_virtual_edge": False,
            },
        },
        {
            "variant": "proposed_model",
            "group": "CORE",
            "display_name": "Proposed",
            "title": "Proposed Model",
            "order": 70,
            "compare": True,
            "backbone_compare": True,
            "overrides": {
                "model_variant": "proposed",
                "encoder_backbone": "hgt",
            },
        },
        {
            "variant": "backbone_graphsage",
            "group": "BACKBONE",
            "display_name": "GraphSAGE",
            "title": "GraphSAGE Backbone",
            "order": 71,
            "compare": False,
            "backbone_compare": True,
            "overrides": {
                "model_variant": "proposed",
                "encoder_backbone": "sage",
            },
        },
        {
            "variant": "backbone_gat",
            "group": "BACKBONE",
            "display_name": "GAT",
            "title": "GAT Backbone",
            "order": 72,
            "compare": False,
            "backbone_compare": True,
            "overrides": {
                "model_variant": "proposed",
                "encoder_backbone": "gat",
            },
        },
        {
            "variant": "backbone_graphconv",
            "group": "BACKBONE",
            "display_name": "GraphConv",
            "title": "GraphConv Backbone",
            "order": 73,
            "compare": False,
            "backbone_compare": True,
            "overrides": {
                "model_variant": "proposed",
                "encoder_backbone": "graphconv",
            },
        },
        {
            "variant": "wo_spatial_encoder",
            "group": "ABLATION",
            "display_name": "w/o Spatial Encoder",
            "title": "w/o Spatial Encoder",
            "order": 100,
            "compare": False,
            "backbone_compare": False,
            "overrides": {
                "use_spatial_encoder": False,
            },
        },
        {
            "variant": "wo_ingredient_path",
            "group": "ABLATION",
            "display_name": "w/o Ingredient Path",
            "title": "w/o Ingredient Path",
            "order": 110,
            "compare": False,
            "backbone_compare": False,
            "overrides": {
                "decoder_use_ingredient_path": False,
            },
        },
        {
            "variant": "wo_tri_attention",
            "group": "ABLATION",
            "display_name": "w/o Tri-Attention",
            "title": "w/o Tri-Attention",
            "order": 120,
            "compare": False,
            "backbone_compare": False,
            "overrides": {
                "decoder_use_tri_attention": False,
            },
        },
        {
            "variant": "wo_global_path",
            "group": "ABLATION",
            "display_name": "w/o Global Path",
            "title": "w/o Global Path",
            "order": 130,
            "compare": False,
            "backbone_compare": False,
            "overrides": {
                "decoder_use_global_path": False,
            },
        },
        {
            "variant": "wo_virtual_edges",
            "group": "ABLATION",
            "display_name": "w/o Virtual Edges",
            "title": "w/o Virtual Edges",
            "order": 140,
            "compare": False,
            "backbone_compare": False,
            "overrides": {
                "use_rl_virtual_edge": False,
                "virtual_edge_types": [],
            },
        },
        {
            "variant": "wo_ranking_loss",
            "group": "ABLATION",
            "display_name": "w/o Ranking Loss",
            "title": "w/o Ranking Loss",
            "order": 150,
            "compare": False,
            "backbone_compare": False,
            "overrides": {
                "use_ranking_loss": False,
                "ranking_lambda": 0.0,
            },
        },
        {
            "variant": "wo_contrastive_loss",
            "group": "ABLATION",
            "display_name": "w/o Contrastive Loss",
            "title": "w/o Contrastive Loss",
            "order": 160,
            "compare": False,
            "backbone_compare": False,
            "overrides": {
                "use_contrastive_loss": False,
                "contrastive_lambda": 0.0,
            },
        },
        {
            "variant": "wo_chemberta",
            "group": "ABLATION",
            "display_name": "w/o ChemBERTa",
            "title": "w/o ChemBERTa",
            "order": 170,
            "compare": False,
            "backbone_compare": False,
            "overrides": {
                "ingredient_feature_mode": "no_bert",
            },
        },
        {
            "variant": "wo_chemgpt",
            "group": "ABLATION",
            "display_name": "w/o ChemGPT",
            "title": "w/o ChemGPT",
            "order": 180,
            "compare": False,
            "backbone_compare": False,
            "overrides": {
                "ingredient_feature_mode": "no_gpt",
            },
        },
        {
            "variant": "wo_fingerprint",
            "group": "ABLATION",
            "display_name": "w/o Fingerprint",
            "title": "w/o Fingerprint",
            "order": 190,
            "compare": False,
            "backbone_compare": False,
            "overrides": {
                "ingredient_feature_mode": "no_fp",
            },
        },
    ]


VARIANTS = build_variant_catalog()
VARIANT_INDEX = {item["variant"]: item for item in VARIANTS}
COMPARISON_KEYS = [item["variant"] for item in VARIANTS if item.get("compare", False)]
BACKBONE_COMPARISON_KEYS = [item["variant"] for item in VARIANTS if item.get("backbone_compare", False)]
BACKBONE_LABELS = {
    "proposed_model": "Proposed (HGT)",
    "backbone_graphsage": "GraphSAGE",
    "backbone_gat": "GAT",
    "backbone_graphconv": "GraphConv",
}
MODALITY_KEYS = ["wo_chemberta", "wo_chemgpt", "wo_fingerprint", "proposed_model"]
MODALITY_LABELS = {
    "wo_chemberta": "w/o ChemBERTa",
    "wo_chemgpt": "w/o ChemGPT",
    "wo_fingerprint": "w/o Fingerprint",
    "proposed_model": "Proposed",
}
MAIN_CURVE_VARIANTS = [
    ("hgt_baseline", "Vanilla HGT", "#2F7ED8", 1.8),
    ("static_similarity_hgt", "Static Similarity HGT", "#F39C12", 1.8),
    ("proposed_model", "Proposed", "#1D9E75", 2.2),
]
STRICT_SPLIT_CFG = {
    "val_ratio": 0.16,
    "test_ratio": 0.20,
    "neg_ratio": 1.0,
    "it_mask_ratio": 0.20,
    "strict_eval_neg_filter": True,
    "filter_hi_to_train_herbs": True,
    "use_it_mask": True,
    "min_pos_per_herb_val": 1,
    "min_pos_per_herb_test": 1,
    "max_fallback_per_herb": 2,
}
CORE_ABLATION_GROUPS = {"CORE", "ABLATION"}
STRICT_METRIC_COLUMNS = ["AUC", "AUPRC", "F1", "Precision", "Recall", "ACC"]
DEFAULT_STRICT_SEEDS = [42]
MIN_STRICT_SEEDS = 1
STRICT_PROTOCOL_VERSION = "strict_v2"
STRICT_SPLIT_ID = "cold_start_herb_strict"
STRICT_THRESHOLD_POLICY = "freeze_once_on_validation_then_apply_on_test"


def ensure_dirs(output_root: str) -> dict[str, str]:
    root = os.path.abspath(output_root)
    paths = {
        "root": root,
        "runs": os.path.join(root, "runs"),
        "figures": os.path.join(root, "figures"),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths


def _suite_config_path(paths: dict[str, str]) -> str:
    return os.path.join(paths["root"], "suite_config.json")


def normalize_seeds(
    seeds: Optional[list[int]] = None,
    seed: Optional[int] = None,
) -> list[int]:
    if seeds:
        seed_list = [int(item) for item in seeds]
    elif seed is not None:
        seed_list = [int(seed)]
    else:
        seed_list = list(DEFAULT_STRICT_SEEDS)

    normalized = []
    seen = set()
    for item in seed_list:
        if item in seen:
            continue
        seen.add(item)
        normalized.append(int(item))
    return normalized


def write_suite_config(paths: dict[str, str], seeds: list[int], groups: Optional[list[str]] = None) -> dict:
    payload = {
        "protocol_version": STRICT_PROTOCOL_VERSION,
        "split_id": STRICT_SPLIT_ID,
        "expected_seeds": [int(item) for item in seeds],
        "expected_seed_count": len(seeds),
        "min_strict_seeds": MIN_STRICT_SEEDS,
        "threshold_policy": STRICT_THRESHOLD_POLICY,
        "groups": list(groups or []),
        "notes": [
            "Strict tables require all planned seeds to finish.",
            "All comparison rows come from models reproduced inside this project under the same split.",
            "Split-sensitive options are locked across all variants and cannot be ablated.",
        ],
    }
    _write_json(_suite_config_path(paths), payload)
    return payload


def load_suite_config(paths: dict[str, str]) -> dict:
    payload = _load_json(_suite_config_path(paths))
    if payload is None:
        return {
            "protocol_version": STRICT_PROTOCOL_VERSION,
            "split_id": STRICT_SPLIT_ID,
            "expected_seeds": list(DEFAULT_STRICT_SEEDS),
            "expected_seed_count": len(DEFAULT_STRICT_SEEDS),
            "min_strict_seeds": MIN_STRICT_SEEDS,
            "threshold_policy": STRICT_THRESHOLD_POLICY,
            "groups": [],
        }
    payload.setdefault("expected_seeds", list(DEFAULT_STRICT_SEEDS))
    payload.setdefault("expected_seed_count", len(payload["expected_seeds"]))
    payload.setdefault("min_strict_seeds", MIN_STRICT_SEEDS)
    payload.setdefault("split_id", STRICT_SPLIT_ID)
    payload.setdefault("threshold_policy", STRICT_THRESHOLD_POLICY)
    payload.setdefault("groups", [])
    return payload


def parse_groups(raw_groups: Optional[list[str]]) -> set[str]:
    default_groups = {"CORE", "BASELINE", "BACKBONE", "ABLATION"}
    if not raw_groups:
        return default_groups

    alias_map = {
        "CORE": "CORE",
        "MAIN": "CORE",
        "PROPOSED": "CORE",
        "BASELINE": "BASELINE",
        "COMPARE": "BASELINE",
        "COMPARISON": "BASELINE",
        "BACKBONE": "BACKBONE",
        "BB": "BACKBONE",
        "ENCODER": "BACKBONE",
        "ABLATION": "ABLATION",
        "ABL": "ABLATION",
        "A": "ABLATION",
        "B": "ABLATION",
        "C": "ABLATION",
        "D": "BASELINE",
        "E": "ABLATION",
        "F": "ABLATION",
        "G": "BASELINE",
        "H": "BASELINE",
    }

    groups: set[str] = set()
    for raw in raw_groups:
        for token in str(raw).replace(",", " ").split():
            key = token.strip().upper()
            if key:
                groups.add(alias_map.get(key, key))
    return groups or default_groups


def select_variants(groups: set[str]) -> list[dict]:
    selected = [item for item in VARIANTS if item["group"] in groups]
    if not any(item["variant"] == "proposed_model" for item in selected):
        selected = [VARIANT_INDEX["proposed_model"], *selected]
    selected.sort(key=lambda item: item["order"])
    return selected


def _variant_root(paths: dict[str, str], variant_key: str) -> str:
    return os.path.join(paths["runs"], variant_key)


def _seed_dir(paths: dict[str, str], variant_key: str, seed: int) -> str:
    return os.path.join(_variant_root(paths, variant_key), f"seed_{int(seed)}")


def _summary_path(paths: dict[str, str], variant_key: str, seed: Optional[int] = None) -> str:
    if seed is None:
        return os.path.join(_variant_root(paths, variant_key), "aggregate_summary.json")
    return os.path.join(_seed_dir(paths, variant_key, seed), "summary.json")


def _error_path(paths: dict[str, str], variant_key: str, seed: int) -> str:
    return os.path.join(_seed_dir(paths, variant_key, seed), "error.json")


def _curve_path(paths: dict[str, str], variant_key: str, seed: Optional[int] = None) -> str:
    if seed is None:
        return os.path.join(_variant_root(paths, variant_key), f"{variant_key}_curves.json")
    return os.path.join(_seed_dir(paths, variant_key, seed), f"{variant_key}_curves.json")


def _load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def _write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _normalize_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    frame = df.copy()
    for src, dst in (
        ("test_AUC", "AUC"),
        ("test_AUPRC", "AUPRC"),
        ("test_F1", "F1"),
        ("test_Precision", "Precision"),
        ("test_Recall", "Recall"),
        ("test_ACC", "ACC"),
    ):
        if src in frame.columns and dst in frame.columns:
            frame[dst] = frame[dst].where(frame[dst].notna(), frame[src])
            frame[src] = frame[src].where(frame[src].notna(), frame[dst])
        elif src in frame.columns:
            frame[dst] = frame[src]
        elif dst in frame.columns:
            frame[src] = frame[dst]
    return frame


def run_variant_seed(variant: dict, paths: dict[str, str], seed: int) -> dict:
    variant_key = variant["variant"]
    run_dir = _seed_dir(paths, variant_key, seed)
    os.makedirs(run_dir, exist_ok=True)

    import train

    train.reset_cfg()
    overrides = dict(variant.get("overrides", {}))
    overrides.update({
        "save_dir": run_dir,
        "seed": int(seed),
        "curve_alias": variant_key,
        "experiment_name": variant_key,
        "experiment_group": variant["group"],
        "experiment_title": variant["title"],
        "rl_save_path": os.path.join(run_dir, "rl_agent.pt"),
    })
    overrides.update(STRICT_SPLIT_CFG)
    train.merge_cfg(overrides)

    try:
        summary = train.main()
        summary.update({
            "display_name": variant["display_name"],
            "status": "ok",
            "source": "internal",
            "order": variant["order"],
            "seed": int(seed),
            "strict_protocol_version": STRICT_PROTOCOL_VERSION,
            "split_id": STRICT_SPLIT_ID,
            "threshold_policy": STRICT_THRESHOLD_POLICY,
        })
        _write_json(_summary_path(paths, variant_key, seed), summary)
        return summary
    except Exception as exc:
        error_payload = {
            "variant": variant_key,
            "group": variant["group"],
            "title": variant["title"],
            "display_name": variant["display_name"],
            "save_dir": run_dir,
            "status": "failed",
            "source": "internal",
            "order": variant["order"],
            "seed": int(seed),
            "strict_protocol_version": STRICT_PROTOCOL_VERSION,
            "split_id": STRICT_SPLIT_ID,
            "threshold_policy": STRICT_THRESHOLD_POLICY,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(_error_path(paths, variant_key, seed), error_payload)
        return error_payload


def collect_internal_run_rows(paths: dict[str, str]) -> list[dict]:
    rows: list[dict] = []
    for variant in VARIANTS:
        variant_root = _variant_root(paths, variant["variant"])
        if not os.path.exists(variant_root):
            continue

        seed_dirs = []
        for entry in sorted(os.listdir(variant_root)):
            full_path = os.path.join(variant_root, entry)
            if entry.startswith("seed_") and os.path.isdir(full_path):
                seed_dirs.append((entry, full_path))

        if not seed_dirs:
            legacy_summary = _load_json(os.path.join(variant_root, "summary.json"))
            legacy_error = _load_json(os.path.join(variant_root, "error.json"))
            payload = legacy_summary or legacy_error
            if payload is not None:
                payload = dict(payload)
                payload.setdefault("seed", payload.get("seed", np.nan))
                payload.setdefault("variant", variant["variant"])
                payload.setdefault("group", variant["group"])
                payload.setdefault("title", variant["title"])
                payload.setdefault("display_name", variant["display_name"])
                payload.setdefault("status", "ok" if legacy_summary is not None else "failed")
                payload.setdefault("source", "internal")
                payload.setdefault("order", variant["order"])
                rows.append(payload)
            continue

        for entry, seed_dir in seed_dirs:
            parsed_seed = entry.replace("seed_", "", 1)
            try:
                parsed_seed_value = int(parsed_seed)
            except Exception:
                parsed_seed_value = np.nan

            summary = _load_json(os.path.join(seed_dir, "summary.json"))
            error_payload = _load_json(os.path.join(seed_dir, "error.json"))
            payload = summary or error_payload
            if payload is None:
                continue
            payload = dict(payload)
            payload.setdefault("seed", parsed_seed_value)
            payload.setdefault("variant", variant["variant"])
            payload.setdefault("group", variant["group"])
            payload.setdefault("title", variant["title"])
            payload.setdefault("display_name", variant["display_name"])
            payload.setdefault("status", "ok" if summary is not None else "failed")
            payload.setdefault("source", "internal")
            payload.setdefault("order", variant["order"])
            rows.append(payload)
    return rows


def _format_mean_std(mean_value: float, std_value: float) -> str:
    return f"{float(mean_value):.4f} +/- {float(std_value):.4f}"


def _strict_reason(
    status: str,
    expected_seeds: list[int],
    completed_seeds: list[int],
) -> tuple[bool, str]:
    expected_list = sorted(int(item) for item in expected_seeds)
    completed_list = sorted(int(item) for item in completed_seeds)

    if status != "ok":
        return False, "no successful runs"
    if len(expected_list) < MIN_STRICT_SEEDS:
        return False, f"need at least {MIN_STRICT_SEEDS} seeds"
    if completed_list != expected_list:
        return False, f"completed seeds {completed_list} != expected {expected_list}"
    return True, "strict-ready"


def _sync_representative_curve(paths: dict[str, str], variant_key: str, ok_group: pd.DataFrame) -> Optional[int]:
    if ok_group.empty or "AUC" not in ok_group.columns:
        return None
    mean_auc = float(pd.to_numeric(ok_group["AUC"], errors="coerce").dropna().mean())
    candidates = []
    for _, row in ok_group.iterrows():
        seed = row.get("seed", np.nan)
        if pd.isna(seed):
            continue
        seed_value = int(seed)
        curve_source = _curve_path(paths, variant_key, seed_value)
        if not os.path.exists(curve_source):
            continue
        auc_value = float(row.get("AUC", np.nan))
        candidates.append((abs(auc_value - mean_auc), seed_value, curve_source))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    _, representative_seed, curve_source = candidates[0]
    shutil.copyfile(curve_source, _curve_path(paths, variant_key))
    return representative_seed


def aggregate_results_dataframe(
    run_df: pd.DataFrame,
    paths: dict[str, str],
    suite_config: dict,
) -> pd.DataFrame:
    if run_df.empty:
        return pd.DataFrame()

    frame = _normalize_metric_columns(run_df.copy())
    for col in ("order", "seed"):
        if col not in frame.columns:
            frame[col] = np.nan
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "source" not in frame.columns:
        frame["source"] = "internal"
    if "status" not in frame.columns:
        frame["status"] = ""
    if "protocol_version" not in frame.columns:
        frame["protocol_version"] = STRICT_PROTOCOL_VERSION
    if "split_id" not in frame.columns:
        frame["split_id"] = STRICT_SPLIT_ID
    if "threshold_policy" not in frame.columns:
        frame["threshold_policy"] = STRICT_THRESHOLD_POLICY

    expected_seeds = [int(item) for item in suite_config.get("expected_seeds", DEFAULT_STRICT_SEEDS)]
    aggregated_rows = []

    for (_, variant_key), group in frame.groupby(["source", "variant"], sort=False):
        group = group.sort_values(["seed", "order"], na_position="last").reset_index(drop=True)
        ok_group = group[group["status"] == "ok"].copy()
        base_row = ok_group.iloc[0] if not ok_group.empty else group.iloc[0]

        completed_seeds = sorted({int(item) for item in ok_group["seed"].dropna().astype(int).tolist()})
        strict_ready, strict_reason = _strict_reason(
            status="ok" if not ok_group.empty else "failed",
            expected_seeds=expected_seeds,
            completed_seeds=completed_seeds,
        )

        aggregate_row = {
            "variant": variant_key,
            "group": base_row.get("group", ""),
            "title": base_row.get("title", ""),
            "display_name": base_row.get("display_name", variant_key),
            "status": "ok" if not ok_group.empty else "failed",
            "source": base_row.get("source", "internal"),
            "order": float(base_row.get("order", 9999)),
            "split_id": str(base_row.get("split_id", "")),
            "threshold_policy": str(base_row.get("threshold_policy", "")),
            "strict_protocol_version": STRICT_PROTOCOL_VERSION,
            "expected_seeds": ",".join(str(item) for item in expected_seeds),
            "completed_seeds": ",".join(str(item) for item in completed_seeds),
            "expected_seed_count": len(expected_seeds),
            "completed_seed_count": len(completed_seeds),
            "strict_ready": bool(strict_ready),
            "strict_reason": strict_reason,
            "error_count": int((group["status"] != "ok").sum()),
        }

        failed_seeds = sorted({int(item) for item in group.loc[group["status"] != "ok", "seed"].dropna().astype(int).tolist()})
        aggregate_row["failed_seeds"] = ",".join(str(item) for item in failed_seeds)

        for metric in STRICT_METRIC_COLUMNS:
            values = pd.to_numeric(ok_group.get(metric, pd.Series(dtype=float)), errors="coerce").dropna()
            if values.empty:
                continue
            mean_value = float(values.mean())
            std_value = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            aggregate_row[metric] = mean_value
            aggregate_row[f"{metric}_std"] = std_value
            aggregate_row[f"{metric}_mean_std"] = _format_mean_std(mean_value, std_value)
            aggregate_row[f"test_{metric}"] = mean_value
            aggregate_row[f"test_{metric}_std"] = std_value

        if not ok_group.empty and "best_threshold" in ok_group.columns:
            threshold_values = pd.to_numeric(ok_group["best_threshold"], errors="coerce").dropna()
            if not threshold_values.empty:
                aggregate_row["best_threshold"] = float(threshold_values.mean())
                aggregate_row["best_threshold_std"] = float(threshold_values.std(ddof=1)) if len(threshold_values) > 1 else 0.0

        if str(base_row.get("source", "internal")) == "internal":
            representative_seed = _sync_representative_curve(paths, variant_key, ok_group)
            if representative_seed is not None:
                aggregate_row["representative_seed"] = int(representative_seed)
            _write_json(_summary_path(paths, variant_key), aggregate_row)

        aggregated_rows.append(aggregate_row)

    aggregated_df = pd.DataFrame(aggregated_rows)
    if aggregated_df.empty:
        return aggregated_df
    aggregated_df = aggregated_df.sort_values(["order", "variant"]).reset_index(drop=True)
    return aggregated_df


def collect_run_data(paths: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    suite_config = load_suite_config(paths)
    internal_rows = collect_internal_run_rows(paths)
    run_df = pd.DataFrame(internal_rows)
    if run_df.empty:
        return pd.DataFrame(), pd.DataFrame(), suite_config

    run_df = _normalize_metric_columns(run_df)
    aggregated_df = aggregate_results_dataframe(run_df, paths, suite_config)
    return run_df, aggregated_df, suite_config


def _strict_ready_mask(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)
    mask = df.get("strict_ready", pd.Series(False, index=df.index))
    if mask.dtype != bool:
        mask = mask.fillna("").astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})
    return mask


def _comparison_mask(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)
    variant_mask = df.get("variant", pd.Series("", index=df.index)).astype(str).isin(COMPARISON_KEYS)
    return variant_mask & _strict_ready_mask(df)


def _backbone_mask(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)
    variant_mask = df.get("variant", pd.Series("", index=df.index)).astype(str).isin(BACKBONE_COMPARISON_KEYS)
    return variant_mask & _strict_ready_mask(df)


def _sort_for_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    frame = df.copy()
    frame["order"] = pd.to_numeric(frame.get("order", 9999), errors="coerce").fillna(9999)
    return frame.sort_values(["order", "variant"]).reset_index(drop=True)


def _metrics_payload(frame: pd.DataFrame) -> list[dict]:
    ok_df = frame[frame["status"] == "ok"].copy()
    payload = []
    for _, row in ok_df.iterrows():
        item = {
            "variant": row.get("variant", ""),
            "group": row.get("group", ""),
            "title": row.get("title", ""),
            "display_name": row.get("display_name", ""),
            "status": row.get("status", ""),
            "source": row.get("source", ""),
            "strict_ready": bool(row.get("strict_ready", False)),
            "strict_reason": row.get("strict_reason", ""),
            "completed_seed_count": int(row.get("completed_seed_count", 0) or 0),
        }
        for key in STRICT_METRIC_COLUMNS:
            if key in row and pd.notna(row[key]):
                item[key] = float(row[key])
            std_key = f"{key}_std"
            if std_key in row and pd.notna(row.get(std_key, np.nan)):
                item[std_key] = float(row[std_key])
            text_key = f"{key}_mean_std"
            if text_key in row and pd.notna(row.get(text_key, np.nan)):
                item[text_key] = str(row[text_key])
        payload.append(item)
    return payload


def export_significance_tests(
    run_df: pd.DataFrame,
    aggregated_df: pd.DataFrame,
    paths: dict[str, str],
) -> dict[str, str]:
    if run_df.empty or aggregated_df.empty:
        return {}

    strict_variants = set(aggregated_df.loc[_comparison_mask(aggregated_df), "variant"].astype(str).tolist())
    if "proposed_model" not in strict_variants:
        return {}

    frame = _normalize_metric_columns(run_df.copy())
    frame = frame[frame.get("status", pd.Series("", index=frame.index)).astype(str).eq("ok")].copy()
    if frame.empty:
        return {}
    frame["seed"] = pd.to_numeric(frame.get("seed", np.nan), errors="coerce")
    frame = frame.dropna(subset=["seed"])
    if frame.empty:
        return {}

    proposed_runs = frame[frame["variant"].astype(str) == "proposed_model"].copy()
    if proposed_runs.empty:
        return {}

    rows = []
    for variant_key in sorted(strict_variants):
        if variant_key == "proposed_model":
            continue
        candidate_runs = frame[frame["variant"].astype(str) == variant_key].copy()
        if candidate_runs.empty:
            continue

        merged = proposed_runs.merge(
            candidate_runs,
            on="seed",
            how="inner",
            suffixes=("_proposed", "_baseline"),
        )
        if merged.empty:
            continue

        for metric in ("AUC", "AUPRC", "F1"):
            proposed_col = f"{metric}_proposed"
            baseline_col = f"{metric}_baseline"
            if proposed_col not in merged.columns or baseline_col not in merged.columns:
                continue

            pairs = merged[["seed", proposed_col, baseline_col]].dropna()
            if len(pairs) < 2:
                continue

            diffs = pairs[proposed_col].astype(float) - pairs[baseline_col].astype(float)
            ttest_p = np.nan
            wilcoxon_p = np.nan
            if scipy_stats is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    try:
                        ttest_p = float(
                            scipy_stats.ttest_rel(
                                pairs[proposed_col].astype(float).to_numpy(),
                                pairs[baseline_col].astype(float).to_numpy(),
                                nan_policy="omit",
                            ).pvalue
                        )
                    except Exception:
                        ttest_p = np.nan
                    try:
                        if np.allclose(diffs.to_numpy(), 0.0):
                            wilcoxon_p = 1.0
                        else:
                            wilcoxon_p = float(scipy_stats.wilcoxon(diffs.to_numpy(), zero_method="wilcox").pvalue)
                    except Exception:
                        wilcoxon_p = np.nan

            rows.append({
                "baseline_variant": variant_key,
                "metric": metric,
                "n_pairs": int(len(pairs)),
                "paired_seeds": ",".join(str(int(seed)) for seed in sorted(pairs["seed"].astype(int).tolist())),
                "proposed_mean": float(pairs[proposed_col].mean()),
                "baseline_mean": float(pairs[baseline_col].mean()),
                "mean_difference": float(diffs.mean()),
                "std_difference": float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0,
                "paired_t_pvalue": ttest_p,
                "wilcoxon_pvalue": wilcoxon_p,
                "scipy_available": bool(scipy_stats is not None),
            })

    if not rows:
        return {}

    stats_df = pd.DataFrame(rows).sort_values(["metric", "baseline_variant"]).reset_index(drop=True)
    csv_path = os.path.join(paths["root"], "significance_tests.csv")
    json_path = os.path.join(paths["root"], "significance_tests.json")
    stats_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    _write_json(json_path, stats_df.to_dict(orient="records"))
    return {
        "significance_tests.csv": csv_path,
        "significance_tests.json": json_path,
    }


def export_tables(
    run_df: pd.DataFrame,
    aggregated_df: pd.DataFrame,
    paths: dict[str, str],
    suite_config: dict,
) -> dict[str, str]:
    if aggregated_df.empty:
        return {}

    all_df = _sort_for_display(aggregated_df.copy())
    run_export_df = _sort_for_display(_normalize_metric_columns(run_df.copy())) if not run_df.empty else pd.DataFrame()
    strict_mask = _strict_ready_mask(all_df)

    ablation_df = _sort_for_display(all_df[strict_mask & all_df["group"].isin(CORE_ABLATION_GROUPS)].copy())
    comparison_df = _sort_for_display(all_df[_comparison_mask(all_df)].copy())
    backbone_df = _sort_for_display(all_df[_backbone_mask(all_df)].copy())
    modality_df = _paper_modality_df(all_df)
    non_strict_df = _sort_for_display(all_df[~strict_mask].copy())
    meta_df = pd.DataFrame([
        {
            "variant": item["variant"],
            "group": item["group"],
            "display_name": item["display_name"],
            "title": item["title"],
            "compare": item.get("compare", False),
            "backbone_compare": item.get("backbone_compare", False),
            "strict_protocol_version": STRICT_PROTOCOL_VERSION,
            "overrides": json.dumps(item.get("overrides", {}), ensure_ascii=False),
        }
        for item in VARIANTS
    ])

    out = {}
    for name, frame in (
        ("all_experiment_results.csv", all_df),
        ("all_experiment_runs.csv", run_export_df),
        ("ablation_results.csv", ablation_df),
        ("comparison_results.csv", comparison_df),
        ("backbone_results.csv", backbone_df),
        ("modality_results.csv", modality_df),
        ("non_strict_results.csv", non_strict_df),
        ("variant_catalog.csv", meta_df),
    ):
        if frame.empty and name == "all_experiment_runs.csv":
            continue
        path = os.path.join(paths["root"], name)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        out[name] = path

    xlsx_path = os.path.join(paths["root"], "experiment_results.xlsx")
    try:
        with pd.ExcelWriter(xlsx_path) as writer:
            all_df.to_excel(writer, sheet_name="all_results", index=False)
            if not run_export_df.empty:
                run_export_df.to_excel(writer, sheet_name="seed_runs", index=False)
            ablation_df.to_excel(writer, sheet_name="ablation", index=False)
            comparison_df.to_excel(writer, sheet_name="comparison", index=False)
            backbone_df.to_excel(writer, sheet_name="backbone", index=False)
            modality_df.to_excel(writer, sheet_name="modality", index=False)
            non_strict_df.to_excel(writer, sheet_name="non_strict", index=False)
            meta_df.to_excel(writer, sheet_name="catalog", index=False)
            pd.DataFrame([suite_config]).to_excel(writer, sheet_name="protocol", index=False)
        out["experiment_results.xlsx"] = xlsx_path
    except Exception:
        pass

    for name, frame in (
        ("ablation_results.json", ablation_df),
        ("comparison_results.json", comparison_df),
        ("backbone_results.json", backbone_df),
        ("modality_results.json", modality_df),
        ("non_strict_results.json", non_strict_df),
    ):
        path = os.path.join(paths["root"], name)
        _write_json(path, _metrics_payload(frame))
        out[name] = path

    protocol_path = os.path.join(paths["root"], "strict_protocol_summary.json")
    _write_json(protocol_path, {
        "protocol_version": suite_config.get("protocol_version", STRICT_PROTOCOL_VERSION),
        "split_id": suite_config.get("split_id", STRICT_SPLIT_ID),
        "expected_seeds": suite_config.get("expected_seeds", []),
        "expected_seed_count": suite_config.get("expected_seed_count", 0),
        "min_strict_seeds": suite_config.get("min_strict_seeds", MIN_STRICT_SEEDS),
        "threshold_policy": suite_config.get("threshold_policy", ""),
        "strict_comparison_variants": comparison_df["variant"].astype(str).tolist(),
        "strict_backbone_variants": backbone_df["variant"].astype(str).tolist(),
        "strict_ablation_variants": ablation_df["variant"].astype(str).tolist(),
        "strict_modality_variants": modality_df["variant"].astype(str).tolist(),
        "non_strict_variants": non_strict_df["variant"].astype(str).tolist(),
    })
    out["strict_protocol_summary.json"] = protocol_path
    out.update(export_significance_tests(run_df, all_df, paths))
    return out


def _save_figure(fig: plt.Figure, paths: dict[str, str], name: str) -> None:
    base = os.path.join(paths["figures"], name)
    fig.savefig(base + ".png")
    fig.savefig(base + ".pdf")
    plt.close(fig)


def _save_figure_many(fig: plt.Figure, paths: dict[str, str], names: list[str]) -> None:
    for name in names:
        base = os.path.join(paths["figures"], name)
        fig.savefig(base + ".png")
        fig.savefig(base + ".pdf")
    plt.close(fig)


def _figure_file_map(paths: dict[str, str], name: str) -> dict[str, str]:
    base = os.path.join(paths["figures"], name)
    return {
        f"{name}.png": base + ".png",
        f"{name}.pdf": base + ".pdf",
    }


def _ok_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["status"] == "ok"].copy()


def _metric_error(frame: pd.DataFrame, metric: str) -> np.ndarray:
    column = f"{metric}_std"
    if column not in frame.columns:
        return np.zeros(len(frame), dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)


def _axis_upper_bound(arrays: list[np.ndarray]) -> float:
    max_value = 0.0
    for array in arrays:
        if array.size:
            max_value = max(max_value, float(np.nanmax(array)))
    return min(1.0, max(0.6, max_value + 0.08))


def _load_curve_payload(paths: dict[str, str], variant_key: str) -> Optional[dict]:
    payload = _load_json(_curve_path(paths, variant_key))
    if not isinstance(payload, dict):
        return None
    return payload


def _parse_seed_text(value) -> list[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    items = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            items.append(int(token))
        except Exception:
            continue
    return items


def _aggregate_seed_curves(
    paths: dict[str, str],
    variant_key: str,
    seeds: list[int],
    x_key: str,
    y_key: str,
    grid: np.ndarray,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    curves = []
    for seed in seeds:
        payload = _load_json(_curve_path(paths, variant_key, seed))
        if not isinstance(payload, dict):
            continue
        if x_key not in payload or y_key not in payload:
            continue
        x = np.asarray(payload[x_key], dtype=float)
        y = np.asarray(payload[y_key], dtype=float)
        if x.size < 2 or y.size < 2:
            continue
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        uniq_x, uniq_idx = np.unique(x, return_index=True)
        uniq_y = y[uniq_idx]
        if uniq_x.size < 2:
            continue
        interp_y = np.interp(grid, uniq_x, uniq_y, left=uniq_y[0], right=uniq_y[-1])
        curves.append(interp_y)
    if not curves:
        return None, None, 0
    curve_arr = np.vstack(curves)
    mean_curve = curve_arr.mean(axis=0)
    std_curve = curve_arr.std(axis=0, ddof=1) if curve_arr.shape[0] > 1 else np.zeros_like(mean_curve)
    return mean_curve, std_curve, int(curve_arr.shape[0])


def _paper_modality_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    strict_mask = _strict_ready_mask(df)
    plot_df = _ok_rows(df[strict_mask & df["variant"].isin(MODALITY_KEYS)].copy())
    if plot_df.empty:
        return plot_df
    plot_df["plot_label"] = plot_df["variant"].map(MODALITY_LABELS).fillna(plot_df["display_name"])
    plot_df["plot_order"] = plot_df["variant"].map({key: idx for idx, key in enumerate(MODALITY_KEYS)})
    return plot_df.sort_values("plot_order").reset_index(drop=True)


def plot_ablation(df: pd.DataFrame, paths: dict[str, str]) -> None:
    strict_mask = _strict_ready_mask(df)
    plot_df = _ok_rows(df[strict_mask & df["group"].isin(CORE_ABLATION_GROUPS)].copy())
    if plot_df.empty or "AUC" not in plot_df.columns:
        return
    plot_df = _sort_for_display(plot_df)
    labels = plot_df["display_name"].astype(str).tolist()
    aucs = pd.to_numeric(plot_df["AUC"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    f1s = pd.to_numeric(plot_df.get("F1", pd.Series(0.0, index=plot_df.index)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(10, 0.72 * len(labels)), 4.8))
    ax.bar(x - width / 2, aucs, yerr=_metric_error(plot_df, "AUC"), width=width, color="#2F7ED8", label="AUC", capsize=3)
    ax.bar(x + width / 2, f1s, yerr=_metric_error(plot_df, "F1"), width=width, color="#1D9E75", label="F1", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=32, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study")
    ax.set_ylim(0.0, _axis_upper_bound([aucs, f1s]))
    ax.legend()
    _save_figure(fig, paths, "ablation_results")


def plot_comparison(df: pd.DataFrame, paths: dict[str, str]) -> None:
    plot_df = _ok_rows(df[_comparison_mask(df)].copy())
    if plot_df.empty or "AUC" not in plot_df.columns:
        return
    plot_df = _sort_for_display(plot_df)
    labels = plot_df["display_name"].astype(str).tolist()
    metrics = [
        ("AUC", "AUC", "#2F7ED8"),
        ("AUPRC", "AUPRC", "#F39C12"),
        ("F1", "F1", "#1D9E75"),
    ]
    x = np.arange(len(labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(max(8, 0.95 * len(labels)), 4.8))
    series = []
    for idx, (column, label, color) in enumerate(metrics):
        if column not in plot_df.columns:
            continue
        values = pd.to_numeric(plot_df[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        series.append(values)
        ax.bar(
            x + (idx - 1) * width,
            values,
            yerr=_metric_error(plot_df, column),
            width=width,
            label=label,
            color=color,
            capsize=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.set_ylim(0.0, _axis_upper_bound(series))
    ax.legend()
    _save_figure_many(fig, paths, ["main_comparison", "sota_comparison"])


def plot_backbone(df: pd.DataFrame, paths: dict[str, str]) -> None:
    plot_df = _ok_rows(df[_backbone_mask(df)].copy())
    if plot_df.empty or "AUC" not in plot_df.columns:
        return
    plot_df = _sort_for_display(plot_df)
    plot_df["plot_label"] = plot_df["variant"].map(BACKBONE_LABELS).fillna(plot_df["display_name"])

    labels = plot_df["plot_label"].astype(str).tolist()
    metrics = [
        ("AUC", "AUC", "#2F7ED8"),
        ("AUPRC", "AUPRC", "#F39C12"),
        ("F1", "F1", "#1D9E75"),
    ]
    x = np.arange(len(labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(max(7.2, 1.05 * len(labels)), 4.8))
    series = []
    for idx, (column, label, color) in enumerate(metrics):
        if column not in plot_df.columns:
            continue
        values = pd.to_numeric(plot_df[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        series.append(values)
        ax.bar(
            x + (idx - 1) * width,
            values,
            yerr=_metric_error(plot_df, column),
            width=width,
            label=label,
            color=color,
            capsize=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Backbone Comparison")
    ax.set_ylim(0.0, _axis_upper_bound(series))
    ax.legend()
    _save_figure(fig, paths, "backbone_comparison")


def plot_main_roc(df: pd.DataFrame, paths: dict[str, str]) -> None:
    plot_df = _ok_rows(df[_comparison_mask(df)].copy())
    if plot_df.empty:
        return
    curves = []
    grid = np.linspace(0.0, 1.0, 201)
    row_map = {str(row["variant"]): row for _, row in plot_df.iterrows()}
    for variant_key, label, color, lw in MAIN_CURVE_VARIANTS:
        row = row_map.get(variant_key)
        if row is None:
            continue
        seeds = _parse_seed_text(row.get("completed_seeds", ""))
        mean_curve, std_curve, n_curves = _aggregate_seed_curves(
            paths=paths,
            variant_key=variant_key,
            seeds=seeds,
            x_key="fpr",
            y_key="tpr",
            grid=grid,
        )
        if mean_curve is None:
            payload = _load_curve_payload(paths, variant_key)
            if not payload or "fpr" not in payload or "tpr" not in payload:
                continue
            curves.append((label, color, lw, np.asarray(payload["fpr"], dtype=float), np.asarray(payload["tpr"], dtype=float), None, float(row.get("AUC", payload.get("auc", 0.0))), 1))
            continue
        curves.append((label, color, lw, grid, mean_curve, std_curve, float(row.get("AUC", 0.0)), n_curves))
    if not curves:
        return

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for label, color, lw, x, y, y_std, auc_value, n_curves in curves:
        ax.plot(
            x,
            y,
            color=color,
            lw=lw,
            label=f"{label} (AUC={auc_value:.4f}, n={n_curves})",
        )
        if y_std is not None and n_curves > 1:
            ax.fill_between(x, np.clip(y - y_std, 0.0, 1.0), np.clip(y + y_std, 0.0, 1.0), color=color, alpha=0.12)
    ax.plot([0.0, 1.0], [0.0, 1.0], ls="--", lw=1.0, color="#7F8C8D")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Mean ROC Curves under Cold-Start Evaluation")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    _save_figure(fig, paths, "main_roc_curves")


def plot_main_pr(df: pd.DataFrame, paths: dict[str, str]) -> None:
    plot_df = _ok_rows(df[_comparison_mask(df)].copy())
    if plot_df.empty:
        return
    curves = []
    grid = np.linspace(0.0, 1.0, 201)
    row_map = {str(row["variant"]): row for _, row in plot_df.iterrows()}
    for variant_key, label, color, lw in MAIN_CURVE_VARIANTS:
        row = row_map.get(variant_key)
        if row is None:
            continue
        seeds = _parse_seed_text(row.get("completed_seeds", ""))
        mean_curve, std_curve, n_curves = _aggregate_seed_curves(
            paths=paths,
            variant_key=variant_key,
            seeds=seeds,
            x_key="recall",
            y_key="precision",
            grid=grid,
        )
        if mean_curve is None:
            payload = _load_curve_payload(paths, variant_key)
            if not payload or "recall" not in payload or "precision" not in payload:
                continue
            curves.append((label, color, lw, np.asarray(payload["recall"], dtype=float), np.asarray(payload["precision"], dtype=float), None, float(row.get("AUPRC", payload.get("auprc", 0.0))), 1))
            continue
        curves.append((label, color, lw, grid, mean_curve, std_curve, float(row.get("AUPRC", 0.0)), n_curves))
    if not curves:
        return

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for label, color, lw, x, y, y_std, auprc_value, n_curves in curves:
        ax.plot(
            x,
            y,
            color=color,
            lw=lw,
            label=f"{label} (AUPRC={auprc_value:.4f}, n={n_curves})",
        )
        if y_std is not None and n_curves > 1:
            ax.fill_between(x, np.clip(y - y_std, 0.0, 1.0), np.clip(y + y_std, 0.0, 1.0), color=color, alpha=0.12)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Mean Precision-Recall Curves under Cold-Start Evaluation")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    _save_figure(fig, paths, "main_pr_curves")


def plot_modality(df: pd.DataFrame, paths: dict[str, str]) -> None:
    plot_df = _paper_modality_df(df)
    if plot_df.empty or "AUC" not in plot_df.columns:
        return

    x = np.arange(len(plot_df))
    width = 0.38
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    aucs = pd.to_numeric(plot_df["AUC"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    f1s = pd.to_numeric(plot_df.get("F1", pd.Series(0.0, index=plot_df.index)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    ax.bar(x - width / 2, aucs, yerr=_metric_error(plot_df, "AUC"), width=width, color="#2F7ED8", label="AUC", capsize=3)
    ax.bar(x + width / 2, f1s, yerr=_metric_error(plot_df, "F1"), width=width, color="#1D9E75", label="F1", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["plot_label"].tolist())
    ax.set_ylabel("Score")
    ax.set_title("Modality Ablation")
    ax.set_ylim(0.0, _axis_upper_bound([aucs, f1s]))
    ax.legend()
    _save_figure(fig, paths, "modality_ablation")


def export_paper_assets(
    run_df: pd.DataFrame,
    aggregated_df: pd.DataFrame,
    paths: dict[str, str],
    suite_config: dict,
) -> dict[str, str]:
    if aggregated_df.empty:
        return {}

    all_df = _sort_for_display(aggregated_df.copy())
    strict_mask = _strict_ready_mask(all_df)
    comparison_df = _sort_for_display(all_df[_comparison_mask(all_df)].copy())
    backbone_df = _sort_for_display(all_df[_backbone_mask(all_df)].copy())
    ablation_df = _sort_for_display(all_df[strict_mask & all_df["group"].isin(CORE_ABLATION_GROUPS)].copy())
    modality_df = _paper_modality_df(all_df)

    significance_path = os.path.join(paths["root"], "significance_tests.csv")
    if os.path.exists(significance_path):
        try:
            significance_df = pd.read_csv(significance_path, encoding="utf-8-sig")
        except Exception:
            significance_df = pd.read_csv(significance_path)
    else:
        significance_df = pd.DataFrame()

    table_frames = [
        ("table1_main_comparison.csv", comparison_df),
        ("table2_backbone_comparison.csv", backbone_df),
        ("table3_core_ablation.csv", ablation_df),
        ("table4_modality_ablation.csv", modality_df),
    ]

    outputs = {}
    for name, frame in table_frames:
        path = os.path.join(paths["root"], name)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        outputs[name] = path

    if not significance_df.empty:
        sig_alias = os.path.join(paths["root"], "table_s1_significance.csv")
        significance_df.to_csv(sig_alias, index=False, encoding="utf-8-sig")
        outputs["table_s1_significance.csv"] = sig_alias

    paper_xlsx = os.path.join(paths["root"], "paper_tables.xlsx")
    try:
        with pd.ExcelWriter(paper_xlsx) as writer:
            comparison_df.to_excel(writer, sheet_name="table1_main", index=False)
            backbone_df.to_excel(writer, sheet_name="table2_backbone", index=False)
            ablation_df.to_excel(writer, sheet_name="table3_ablation", index=False)
            modality_df.to_excel(writer, sheet_name="table4_modality", index=False)
            if not significance_df.empty:
                significance_df.to_excel(writer, sheet_name="table_s1_significance", index=False)
            if not run_df.empty:
                _sort_for_display(_normalize_metric_columns(run_df.copy())).to_excel(writer, sheet_name="seed_runs", index=False)
            pd.DataFrame([suite_config]).to_excel(writer, sheet_name="protocol", index=False)
        outputs["paper_tables.xlsx"] = paper_xlsx
    except Exception:
        pass

    manifest_rows = [
        {
            "asset_id": "T1",
            "asset_type": "table",
            "title": "Main Comparison under Cold-Start Evaluation",
            "section": "Results",
            "required": True,
            "path": outputs.get("table1_main_comparison.csv", ""),
            "status": "ready" if not comparison_df.empty else "missing",
            "notes": "Use as the main performance table.",
        },
        {
            "asset_id": "T2",
            "asset_type": "table",
            "title": "Backbone Comparison under Cold-Start Evaluation",
            "section": "Results",
            "required": True,
            "path": outputs.get("table2_backbone_comparison.csv", ""),
            "status": "ready" if not backbone_df.empty else "missing",
            "notes": "Use for encoder backbone comparison.",
        },
        {
            "asset_id": "T3",
            "asset_type": "table",
            "title": "Core Ablation Study",
            "section": "Results",
            "required": True,
            "path": outputs.get("table3_core_ablation.csv", ""),
            "status": "ready" if not ablation_df.empty else "missing",
            "notes": "Use for core module ablations.",
        },
        {
            "asset_id": "T4",
            "asset_type": "table",
            "title": "Multimodal Feature Ablation",
            "section": "Results",
            "required": True,
            "path": outputs.get("table4_modality_ablation.csv", ""),
            "status": "ready" if not modality_df.empty else "missing",
            "notes": "Use for ChemBERTa, ChemGPT, and fingerprint ablations.",
        },
        {
            "asset_id": "TS1",
            "asset_type": "table",
            "title": "Statistical Significance Tests",
            "section": "Supplementary",
            "required": True,
            "path": outputs.get("table_s1_significance.csv", significance_path if os.path.exists(significance_path) else ""),
            "status": "ready" if not significance_df.empty else "missing",
            "notes": "Report paired t-test and Wilcoxon results.",
        },
    ]

    figure_specs = [
        ("F1", "Main Comparison", "Results", True, "main_comparison", "Primary comparison bar chart."),
        ("F2", "Backbone Comparison", "Results", True, "backbone_comparison", "Encoder backbone comparison."),
        ("F3", "Core Ablation Study", "Results", True, "ablation_results", "Core ablation bar chart."),
        ("F4", "Multimodal Feature Ablation", "Results", True, "modality_ablation", "Feature ablation bar chart."),
        ("F5", "Mean ROC Curves under Cold-Start Evaluation", "Results", True, "main_roc_curves", "Mean ROC curves with variability bands across seeds."),
        ("F6", "Mean Precision-Recall Curves under Cold-Start Evaluation", "Results", True, "main_pr_curves", "Mean PR curves with variability bands across seeds."),
        ("FR1", "Dataset Comparison", "Methods", False, os.path.abspath(os.path.join("figures", "fig6_dataset_comparison.pdf")), "Generate via generate_figures.py if needed."),
        ("FR2", "Cold-Start Split Protocol", "Methods", False, os.path.abspath(os.path.join("figures", "fig8_split_protocol.pdf")), "Generate via generate_figures.py if needed."),
        ("FR3", "Data Distribution", "Methods", False, os.path.abspath(os.path.join("figures", "fig7_distributions.pdf")), "Generate via generate_figures.py if needed."),
    ]

    for asset_id, title, section, required, name_or_path, notes in figure_specs:
        if name_or_path.endswith(".pdf"):
            figure_path = name_or_path
            status = "ready" if os.path.exists(figure_path) else "recommended"
        else:
            figure_map = _figure_file_map(paths, name_or_path)
            figure_path = figure_map[f"{name_or_path}.pdf"]
            status = "ready" if os.path.exists(figure_path) else "missing"
            outputs.update(figure_map)
        manifest_rows.append({
            "asset_id": asset_id,
            "asset_type": "figure",
            "title": title,
            "section": section,
            "required": required,
            "path": figure_path,
            "status": status,
            "notes": notes,
        })

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_csv = os.path.join(paths["root"], "paper_asset_manifest.csv")
    manifest_json = os.path.join(paths["root"], "paper_asset_manifest.json")
    manifest_df.to_csv(manifest_csv, index=False, encoding="utf-8-sig")
    _write_json(manifest_json, manifest_df.to_dict(orient="records"))
    outputs["paper_asset_manifest.csv"] = manifest_csv
    outputs["paper_asset_manifest.json"] = manifest_json
    return outputs


def generate_report(paths: dict[str, str]) -> dict[str, str]:
    run_df, aggregated_df, suite_config = collect_run_data(paths)
    outputs = export_tables(run_df, aggregated_df, paths, suite_config)
    if not aggregated_df.empty:
        plot_ablation(aggregated_df, paths)
        plot_comparison(aggregated_df, paths)
        plot_backbone(aggregated_df, paths)
        plot_main_roc(aggregated_df, paths)
        plot_main_pr(aggregated_df, paths)
        plot_modality(aggregated_df, paths)
        outputs.update(export_paper_assets(run_df, aggregated_df, paths, suite_config))
    return outputs


def run_ablation_suite(
    output_root: str = DEFAULT_OUTPUT_ROOT,
    groups: Optional[list[str]] = None,
    seed: Optional[int] = None,
    seeds: Optional[list[int]] = None,
) -> list[dict]:
    paths = ensure_dirs(output_root)
    seed_list = normalize_seeds(seeds=seeds, seed=seed)
    write_suite_config(paths, seed_list, groups=groups)
    selected = select_variants(parse_groups(groups))
    results = []
    for variant in selected:
        print(f"\n{'=' * 72}")
        print(f"Running {variant['variant']} | {variant['title']}")
        print(f"Seeds: {seed_list}")
        print(f"{'=' * 72}")
        for current_seed in seed_list:
            result = run_variant_seed(variant, paths, seed=current_seed)
            results.append(result)
            if result.get("status") == "failed":
                print(
                    f"[Failed] {variant['variant']} seed={current_seed}: "
                    f"{result.get('error', 'Unknown error')}"
                )
            else:
                print(
                    f"[Done] {variant['variant']} seed={current_seed} "
                    f"AUC={float(result.get('test_AUC', np.nan)):.4f} "
                    f"F1={float(result.get('test_F1', np.nan)):.4f}"
                )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run strict comparison and core ablation experiments."
    )
    parser.add_argument(
        "--mode",
        choices=["run", "report", "all"],
        default="all",
        help="run: train selected variants, report: aggregate existing results, all: run then report",
    )
    parser.add_argument(
        "--groups",
        nargs="*",
        default=None,
        help="Experiment groups to run, e.g. --groups CORE BASELINE BACKBONE ABLATION",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Single random seed. Use --seeds for strict multi-seed runs.",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Seed list for strict runs, e.g. --seeds 42 43 44 45 46",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory used to save runs, tables, and figures",
    )
    args = parser.parse_args()

    paths = ensure_dirs(args.output_root)
    if args.mode in ("run", "all"):
        run_ablation_suite(
            output_root=args.output_root,
            groups=args.groups,
            seed=args.seed,
            seeds=args.seeds,
        )
    if args.mode in ("report", "all"):
        outputs = generate_report(paths)
        print("\nSaved outputs:")
        for _, path in sorted(outputs.items()):
            print(f"  {path}")
        print(f"  {paths['figures']}")


if __name__ == "__main__":
    main()
