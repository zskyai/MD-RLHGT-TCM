from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import experiment_suite as es


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
DEFAULT_OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "checkpoints", "backbone_strict")
DEFAULT_MIN_TOP_JOURNAL_SEEDS = 1
DEFAULT_VARIANT_KEYS = [
    "proposed_model",
    "backbone_graphsage",
    "backbone_gat",
    "backbone_graphconv",
]


def _write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


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


def _environment_snapshot() -> dict:
    payload = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cwd": os.getcwd(),
        "script_dir": SCRIPT_DIR,
    }
    try:
        import torch

        payload["torch_version"] = getattr(torch, "__version__", "unknown")
        payload["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            payload["cuda_device_count"] = int(torch.cuda.device_count())
            payload["cuda_device_name"] = str(torch.cuda.get_device_name(0))
    except Exception as exc:
        payload["torch_version"] = f"unavailable: {exc}"
        payload["cuda_available"] = False
    return payload


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _export_code_provenance(paths: dict[str, str]) -> str:
    tracked_files = [
        "train.py",
        "model.py",
        "cold_start_split.py",
        "experiment_suite.py",
        "strict_backbone_comparison.py",
    ]
    rows = []
    for name in tracked_files:
        full_path = os.path.join(SCRIPT_DIR, name)
        if not os.path.exists(full_path):
            continue
        rows.append({
            "file": name,
            "path": full_path,
            "sha256": _file_sha256(full_path),
            "size_bytes": int(os.path.getsize(full_path)),
            "mtime": float(os.path.getmtime(full_path)),
        })
    path = os.path.join(paths["root"], "strict_backbone_code_provenance.json")
    _write_json(path, rows)
    return path


def _suite_config_path(paths: dict[str, str]) -> str:
    return os.path.join(paths["root"], "suite_config.json")


def write_suite_config(
    paths: dict[str, str],
    seeds: list[int],
    variants: list[str],
    min_seed_count: int = DEFAULT_MIN_TOP_JOURNAL_SEEDS,
) -> dict:
    payload = {
        "protocol_version": es.STRICT_PROTOCOL_VERSION,
        "split_id": es.STRICT_SPLIT_ID,
        "expected_seeds": [int(item) for item in seeds],
        "expected_seed_count": len(seeds),
        "min_top_journal_seed_count": int(min_seed_count),
        "threshold_policy": es.STRICT_THRESHOLD_POLICY,
        "variants": list(variants),
        "notes": [
            "This suite reuses the strict split and decoder settings from experiment_suite.py.",
            "Only the encoder backbone changes across rows.",
            "The current model logic remains unchanged.",
        ],
    }
    _write_json(_suite_config_path(paths), payload)
    return payload


def _load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _plot_metric(aggregated_df: pd.DataFrame, paths: dict[str, str], metric: str, ylabel: str) -> Optional[str]:
    if aggregated_df.empty or "strict_ready" not in aggregated_df.columns:
        return None
    subset = aggregated_df[aggregated_df["strict_ready"]].copy()
    if subset.empty or metric not in subset.columns:
        return None

    subset = subset[subset["variant"].isin(DEFAULT_VARIANT_KEYS)].copy()
    if subset.empty:
        return None

    subset = subset.sort_values("order").reset_index(drop=True)
    labels = [es.BACKBONE_LABELS.get(item, item) for item in subset["variant"].tolist()]
    values = [float(item) for item in subset[metric].tolist()]
    errors = [float(item) if pd.notna(item) else 0.0 for item in subset.get(f"{metric}_std", pd.Series([0.0] * len(subset))).tolist()]
    colors = ["#1D9E75", "#2F7ED8", "#F39C12", "#8E5EA2"][:len(subset)]

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    bars = ax.bar(labels, values, yerr=errors, capsize=3, color=colors, alpha=0.95)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Strict Backbone Comparison ({metric})")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)

    for rect, value in zip(bars, values):
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            rect.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    figure_path = os.path.join(paths["figures"], f"Figure_strict_backbone_{metric.lower()}.png")
    fig.savefig(figure_path)
    plt.close(fig)
    return figure_path


def _export_significance(aggregated_df: pd.DataFrame, run_df: pd.DataFrame, paths: dict[str, str]) -> Optional[str]:
    if run_df.empty or "status" not in run_df.columns:
        return None
    subset = run_df[run_df["status"] == "ok"].copy()
    if subset.empty:
        return None

    rows = []
    anchor = "proposed_model"
    for variant_key in DEFAULT_VARIANT_KEYS:
        if variant_key == anchor:
            continue
        lhs = subset[subset["variant"] == anchor]
        rhs = subset[subset["variant"] == variant_key]
        merged = lhs.merge(rhs, on="seed", suffixes=("_lhs", "_rhs"))
        if merged.empty:
            continue
        for metric in ("AUC", "AUPRC", "F1", "Precision", "Recall", "ACC"):
            lhs_values = pd.to_numeric(merged.get(f"{metric}_lhs"), errors="coerce")
            rhs_values = pd.to_numeric(merged.get(f"{metric}_rhs"), errors="coerce")
            mask = lhs_values.notna() & rhs_values.notna()
            if not mask.any():
                continue
            diffs = lhs_values[mask] - rhs_values[mask]
            diff_std = float(diffs.std(ddof=1)) if len(diffs) > 1 else np.nan
            effect_size = float(diffs.mean() / diff_std) if (pd.notna(diff_std) and diff_std > 0) else np.nan
            if len(diffs) <= 1:
                p_value = np.nan
            elif es.scipy_stats is not None:
                _, p_value = es.scipy_stats.ttest_rel(lhs_values[mask], rhs_values[mask], nan_policy="omit")
            else:
                p_value = np.nan
            rows.append({
                "lhs_variant": anchor,
                "rhs_variant": variant_key,
                "lhs_display_name": es.BACKBONE_LABELS.get(anchor, anchor),
                "rhs_display_name": es.BACKBONE_LABELS.get(variant_key, variant_key),
                "metric": metric,
                "mean_delta": float(diffs.mean()),
                "paired_cohens_d": effect_size,
                "p_value": float(p_value) if pd.notna(p_value) else np.nan,
                "seed_count": int(mask.sum()),
            })

    if not rows:
        return None
    path = os.path.join(paths["root"], "backbone_significance_tests.csv")
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _export_integrity_audit(run_df: pd.DataFrame, suite_config: dict, paths: dict[str, str]) -> Optional[str]:
    expected_variants = [str(item) for item in suite_config.get("variants", DEFAULT_VARIANT_KEYS)]
    expected_seeds = [int(item) for item in suite_config.get("expected_seeds", [42])]
    min_seed_count = int(suite_config.get("min_top_journal_seed_count", DEFAULT_MIN_TOP_JOURNAL_SEEDS))
    rows = []
    for variant_key in expected_variants:
        ok_group = run_df[
            (run_df.get("variant") == variant_key)
            & (run_df.get("status") == "ok")
        ].copy() if not run_df.empty else pd.DataFrame()
        completed_seeds = sorted({
            int(item)
            for item in pd.to_numeric(ok_group.get("seed", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).tolist()
        })
        rows.append({
            "variant": variant_key,
            "display_name": es.BACKBONE_LABELS.get(variant_key, es.VARIANT_INDEX[variant_key]["display_name"]),
            "expected_seeds": ",".join(str(item) for item in expected_seeds),
            "completed_seeds": ",".join(str(item) for item in completed_seeds),
            "expected_seed_count": len(expected_seeds),
            "completed_seed_count": len(completed_seeds),
            "complete": completed_seeds == sorted(expected_seeds),
            "meets_min_seed_count": len(completed_seeds) >= min_seed_count,
            "top_journal_ready": (completed_seeds == sorted(expected_seeds)) and (len(completed_seeds) >= min_seed_count),
            "threshold_policy": suite_config.get("threshold_policy", es.STRICT_THRESHOLD_POLICY),
            "split_id": suite_config.get("split_id", es.STRICT_SPLIT_ID),
        })

    path = os.path.join(paths["root"], "strict_backbone_integrity_audit.csv")
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _export_report_summary(aggregated_df: pd.DataFrame, suite_config: dict, paths: dict[str, str]) -> str:
    min_seed_count = int(suite_config.get("min_top_journal_seed_count", DEFAULT_MIN_TOP_JOURNAL_SEEDS))
    expected_variants = [str(item) for item in suite_config.get("variants", DEFAULT_VARIANT_KEYS)]
    payload = {
        "variant_count": len(expected_variants),
        "expected_rows": len(expected_variants),
        "observed_rows": int(len(aggregated_df)),
        "strict_ready_rows": int(aggregated_df["strict_ready"].sum()) if (not aggregated_df.empty and "strict_ready" in aggregated_df.columns) else 0,
        "min_top_journal_seed_count": min_seed_count,
        "top_journal_ready_rows": int(((aggregated_df["completed_seed_count"] >= min_seed_count) & (aggregated_df["strict_ready"])).sum())
        if (not aggregated_df.empty and {"completed_seed_count", "strict_ready"}.issubset(aggregated_df.columns))
        else 0,
    }
    path = os.path.join(paths["root"], "strict_backbone_report_summary.json")
    _write_json(path, payload)
    return path


def _export_fairness_contract(paths: dict[str, str], suite_config: dict) -> str:
    payload = {
        "task_definition": "strict_backbone_comparison_for_herb_target_prediction",
        "shared_training_entry": "experiment_suite.run_variant_seed -> train.main",
        "shared_threshold_policy": suite_config.get("threshold_policy", es.STRICT_THRESHOLD_POLICY),
        "shared_split_id": suite_config.get("split_id", es.STRICT_SPLIT_ID),
        "shared_strict_split_config": dict(es.STRICT_SPLIT_CFG),
        "shared_decoder_statement": "Only encoder_backbone changes across rows; other model logic stays unchanged.",
        "variant_keys": suite_config.get("variants", DEFAULT_VARIANT_KEYS),
        "fairness_requirements": [
            "Same strict split configuration across all backbones.",
            "Same threshold freezing policy across all backbones.",
            "Same decoder and ingredient path configuration across all backbones.",
            "Only encoder_backbone differs across rows.",
        ],
    }
    path = os.path.join(paths["root"], "strict_backbone_fairness_contract.json")
    _write_json(path, payload)
    return path


def generate_report(paths: dict[str, str]) -> tuple[dict[str, str], pd.DataFrame]:
    run_df, aggregated_df, suite_config = es.collect_run_data(paths)
    if not aggregated_df.empty:
        aggregated_df = aggregated_df[aggregated_df["variant"].isin(DEFAULT_VARIANT_KEYS)].copy()
        for metric in ("AUC", "AUPRC", "F1", "Precision", "Recall", "ACC"):
            if metric in aggregated_df.columns and f"{metric}_std" in aggregated_df.columns:
                seed_counts = pd.to_numeric(aggregated_df.get("completed_seed_count"), errors="coerce").fillna(0)
                ci95 = []
                for mean_value, std_value, n in zip(aggregated_df[metric], aggregated_df[f"{metric}_std"], seed_counts):
                    if pd.notna(std_value) and int(n) > 1:
                        ci95.append(float(1.96 * float(std_value) / np.sqrt(int(n))))
                    else:
                        ci95.append(0.0)
                aggregated_df[f"{metric}_ci95"] = ci95
    if not run_df.empty:
        run_df = run_df[run_df["variant"].isin(DEFAULT_VARIANT_KEYS)].copy()

    outputs = {}
    run_csv = os.path.join(paths["root"], "strict_backbone_runs.csv")
    agg_csv = os.path.join(paths["root"], "strict_backbone_results.csv")
    run_df.to_csv(run_csv, index=False, encoding="utf-8-sig")
    aggregated_df.to_csv(agg_csv, index=False, encoding="utf-8-sig")
    outputs["strict_backbone_runs.csv"] = run_csv
    outputs["strict_backbone_results.csv"] = agg_csv

    env_path = os.path.join(paths["root"], "strict_backbone_environment.json")
    _write_json(env_path, _environment_snapshot())
    outputs["strict_backbone_environment.json"] = env_path
    provenance_path = _export_code_provenance(paths)
    outputs["strict_backbone_code_provenance.json"] = provenance_path
    fairness_path = _export_fairness_contract(paths, suite_config)
    outputs["strict_backbone_fairness_contract.json"] = fairness_path

    integrity_path = _export_integrity_audit(run_df, suite_config, paths)
    if integrity_path is not None:
        outputs["strict_backbone_integrity_audit.csv"] = integrity_path

    summary_path = _export_report_summary(aggregated_df, suite_config, paths)
    outputs["strict_backbone_report_summary.json"] = summary_path

    manifest = []
    for metric, ylabel in (("AUC", "AUC"), ("F1", "F1 Score")):
        figure_path = _plot_metric(aggregated_df, paths, metric=metric, ylabel=ylabel)
        if figure_path is not None:
            manifest.append({
                "figure_id": f"strict_backbone_{metric.lower()}",
                "title": f"Strict Backbone Comparison ({metric})",
                "path": figure_path,
            })

    manifest_path = os.path.join(paths["figures"], "strict_backbone_figure_manifest.json")
    _write_json(manifest_path, manifest)
    outputs["strict_backbone_figure_manifest.json"] = manifest_path

    significance_path = _export_significance(aggregated_df, run_df, paths)
    if significance_path is not None:
        outputs["backbone_significance_tests.csv"] = significance_path

    return outputs, aggregated_df


def run_suite(
    output_root: str,
    seeds: list[int],
    variant_keys: list[str],
    min_seed_count: int = DEFAULT_MIN_TOP_JOURNAL_SEEDS,
) -> list[dict]:
    paths = ensure_dirs(output_root)
    write_suite_config(paths, seeds=seeds, variants=variant_keys, min_seed_count=min_seed_count)
    results = []
    for variant_key in variant_keys:
        variant = es.VARIANT_INDEX[variant_key]
        print(f"\n{'=' * 72}")
        print(f"Backbone variant: {variant_key} | Seeds: {seeds}")
        print(f"{'=' * 72}")
        for current_seed in seeds:
            result = es.run_variant_seed(variant, paths, seed=current_seed)
            results.append(result)
            if result.get("status") == "ok":
                print(
                    f"[Done] variant={variant_key} seed={current_seed} "
                    f"AUC={float(result.get('test_AUC', np.nan)):.4f} "
                    f"F1={float(result.get('test_F1', np.nan)):.4f}"
                )
            else:
                print(f"[Failed] variant={variant_key} seed={current_seed}: {result.get('error', 'Unknown error')}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a strict backbone comparison without changing the current model logic."
    )
    parser.add_argument(
        "--mode",
        choices=["run", "report", "all"],
        default="all",
        help="run: launch the selected variants, report: aggregate existing outputs, all: run then aggregate.",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=list(DEFAULT_VARIANT_KEYS),
        help="Backbone variant keys from experiment_suite.py",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Single seed. Use --seeds for multi-seed runs.",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Seed list, e.g. --seeds 42 43 44 45 46",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory used to save runs, tables, and figures.",
    )
    parser.add_argument(
        "--min-seed-count",
        type=int,
        default=DEFAULT_MIN_TOP_JOURNAL_SEEDS,
        help="Minimum seed count required to mark a row as top-journal-ready.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit with a non-zero status if any expected row is incomplete or below the minimum seed count.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    invalid_variants = [item for item in args.variants if item not in es.VARIANT_INDEX]
    if invalid_variants:
        raise ValueError(f"Unknown variants: {invalid_variants}")

    seeds = es.normalize_seeds(seeds=args.seeds, seed=args.seed)
    paths = ensure_dirs(args.output_root)

    if args.mode in ("run", "all"):
        run_suite(
            output_root=args.output_root,
            seeds=seeds,
            variant_keys=list(args.variants),
            min_seed_count=int(args.min_seed_count),
        )
    if args.mode in ("report", "all"):
        outputs, aggregated_df = generate_report(paths)
        print("\nSaved outputs:")
        for _, path in sorted(outputs.items()):
            print(f"  {path}")
        if args.require_complete:
            if aggregated_df.empty or "strict_ready" not in aggregated_df.columns or "completed_seed_count" not in aggregated_df.columns:
                raise SystemExit("Strict backbone report is incomplete: no strict-ready rows available.")
            ready_mask = aggregated_df["strict_ready"] & (aggregated_df["completed_seed_count"] >= int(args.min_seed_count))
            if int(ready_mask.sum()) != int(len(aggregated_df)):
                raise SystemExit("Strict backbone report failed the completeness gate.")


if __name__ == "__main__":
    main()
