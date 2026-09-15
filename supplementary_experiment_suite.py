from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Optional

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "checkpoints", "supplementary_experiments")
DEFAULT_SEEDS = [42]
DEFAULT_DATA_PATH = os.path.join(SCRIPT_DIR, "processed", "hetero_graph.pt")


DISPLAY_NAME_MAP = {
    "proposed_model": "Proposed",
    "hgt_baseline": "Vanilla HGT",
    "static_similarity_hgt": "Static Similarity HGT",
    "strong_herb_only_strict": "Strong herb-only",
    "HTINet2 strict": "HTINet2",
    "htinet2_strict": "HTINet2",
}


def _write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def ensure_dirs(output_root: str) -> dict[str, str]:
    root = os.path.abspath(output_root)
    paths = {
        "root": root,
        "figures": os.path.join(root, "figures"),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths


def normalize_seeds(seeds: Optional[list[int]] = None, seed: Optional[int] = None) -> list[int]:
    if seeds:
        seed_list = [int(item) for item in seeds]
    elif seed is not None:
        seed_list = [int(seed)]
    else:
        seed_list = list(DEFAULT_SEEDS)
    normalized = []
    seen = set()
    for item in seed_list:
        if item in seen:
            continue
        seen.add(item)
        normalized.append(int(item))
    return normalized


def _script_path(name: str) -> str:
    return os.path.join(SCRIPT_DIR, name)


def _run_command(command: list[str], cwd: str) -> dict:
    proc = subprocess.Popen(command, cwd=cwd)
    return {
        "command": command,
        "returncode": int(proc.wait()),
        "stdout": "",
        "stderr": "",
    }


def _component_catalog(output_root: str, data_path: str, seeds: list[int], min_seed_count: int, require_complete: bool) -> list[dict]:
    seed_args = [str(item) for item in seeds]
    require_flag = ["--require-complete"] if require_complete else []
    representative_seed = int(seeds[0]) if seeds else 42
    protocol_root = os.path.join(output_root, "protocol_sensitivity")
    protocol_run_dir = os.path.join(
        protocol_root,
        "runs",
        "strict_herb_cold_start",
        "proposed_model",
        f"seed_{representative_seed}",
    )
    return [
        {
            "component": "protocol_sensitivity",
            "title": "Protocol Sensitivity",
            "script": _script_path("protocol_sensitivity_suite.py"),
            "output_root": protocol_root,
            "result_csv": os.path.join(protocol_root, "protocol_sensitivity_results.csv"),
            "category": "internal_protocol",
            "run_args": [
                "--mode", "all",
                "--seeds", *seed_args,
                "--min-seed-count", str(min_seed_count),
                "--output-root", protocol_root,
                *require_flag,
            ],
        },
        {
            "component": "strong_herb_only_strict",
            "title": "Strong Herb-only Strict Baseline",
            "script": _script_path("strong_herb_only_strict.py"),
            "output_root": os.path.join(output_root, "strong_herb_only_strict"),
            "result_csv": os.path.join(output_root, "strong_herb_only_strict", "strong_herb_only_results.csv"),
            "category": "internal_control",
            "run_args": [
                "--mode", "all",
                "--data-path", data_path,
                "--seeds", *seed_args,
                "--min-seed-count", str(min_seed_count),
                "--output-root", os.path.join(output_root, "strong_herb_only_strict"),
                *require_flag,
            ],
        },
        {
            "component": "htinet2_strict",
            "title": "HTINet2 Strict External Baseline",
            "script": _script_path("external_htinet2_strict.py"),
            "output_root": os.path.join(output_root, "external_htinet2_strict"),
            "result_csv": os.path.join(output_root, "external_htinet2_strict", "htinet2_results.csv"),
            "category": "external_main",
            "run_args": [
                "--mode", "all",
                "--data-path", data_path,
                "--seeds", *seed_args,
                "--min-seed-count", str(min_seed_count),
                "--output-root", os.path.join(output_root, "external_htinet2_strict"),
                *require_flag,
            ],
        },
        {
            "component": "ingredient_validation",
            "title": "Ingredient Validation",
            "script": _script_path("supplementary_ingredient_validation.py"),
            "output_root": os.path.join(output_root, "ingredient_validation"),
            "result_csv": os.path.join(output_root, "ingredient_validation", "supplementary_ingredient_validation.csv"),
            "category": "internal_mechanism",
            "run_args": [
                "--project-root", SCRIPT_DIR,
                "--run-dir", protocol_run_dir,
                "--data-path", data_path,
                "--output-dir", os.path.join(output_root, "ingredient_validation"),
                "--seed", str(representative_seed),
            ],
        },
    ]


def run_all_components(paths: dict[str, str], output_root: str, data_path: str, seeds: list[int], min_seed_count: int, require_complete: bool) -> list[dict]:
    catalog = _component_catalog(output_root=output_root, data_path=data_path, seeds=seeds, min_seed_count=min_seed_count, require_complete=require_complete)
    results = []
    for item in catalog:
        command = [sys.executable, "-u", item["script"], *item["run_args"]]
        print(f"\n{'=' * 88}")
        print(f"Running {item['component']} -> {item['title']}")
        print(" ".join(command))
        print(f"{'=' * 88}")
        result = _run_command(command, cwd=SCRIPT_DIR)
        result["component"] = item["component"]
        result["title"] = item["title"]
        result["category"] = item["category"]
        result["output_root"] = item["output_root"]
        result["result_csv"] = item["result_csv"]
        results.append(result)

        print(result["stdout"])
        if result["stderr"]:
            print(result["stderr"], file=sys.stderr)
        if result["returncode"] != 0:
            print(f"[Failed] {item['component']} exited with code {result['returncode']}", file=sys.stderr)
            if require_complete:
                break
    log_path = os.path.join(paths["root"], "supplementary_run_log.json")
    _write_json(log_path, results)
    return results


def _safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path)


def _apply_figure_style() -> None:
    if plt is None:
        return
    plt.rcParams.update({
        "figure.dpi": 180,
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
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, f"{name}.pdf")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    return pdf_path


def _resolve_metric_col(frame: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for col in candidates:
        if col in frame.columns:
            return col
    return None


def _resolve_label(row: pd.Series) -> str:
    for key in ("display_name", "title", "variant", "component_title", "component"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return DISPLAY_NAME_MAP.get(value, value)
    return "Unknown"


def _make_main_table_figure(main_table_df: pd.DataFrame, out_dir: str) -> Optional[str]:
    if plt is None or main_table_df.empty:
        return None
    auc_col = _resolve_metric_col(main_table_df, ["AUC", "test_AUC"])
    f1_col = _resolve_metric_col(main_table_df, ["F1", "test_F1"])
    if auc_col is None and f1_col is None:
        return None

    plot_df = main_table_df.copy()
    plot_df["plot_label"] = plot_df.apply(_resolve_label, axis=1)
    order = ["Vanilla HGT", "Static Similarity HGT", "Strong herb-only", "HTINet2", "Proposed"]
    plot_df["plot_order"] = plot_df["plot_label"].apply(lambda x: order.index(x) if x in order else len(order))
    plot_df = plot_df.sort_values(["plot_order", "plot_label"]).reset_index(drop=True)

    _apply_figure_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    x = np.arange(len(plot_df))
    colors = ["#6C8EBF" if label != "Proposed" else "#D35400" for label in plot_df["plot_label"]]

    if auc_col is not None:
        y = pd.to_numeric(plot_df[auc_col], errors="coerce").fillna(np.nan).to_numpy()
        err_col = f"{auc_col}_ci95" if f"{auc_col}_ci95" in plot_df.columns else None
        yerr = pd.to_numeric(plot_df[err_col], errors="coerce").fillna(0.0).to_numpy() if err_col else None
        axes[0].bar(x, y, color=colors, edgecolor="black", linewidth=0.6, yerr=yerr, capsize=2)
        axes[0].set_title("Main Comparison: AUC")
        axes[0].set_ylabel("AUC")
        axes[0].set_xticks(x, plot_df["plot_label"], rotation=18, ha="right")
        axes[0].set_ylim(max(0.0, np.nanmin(y) - 0.05), min(1.0, np.nanmax(y) + 0.05))
        axes[0].grid(axis="y", linestyle="--", alpha=0.3)
    else:
        axes[0].axis("off")

    if f1_col is not None:
        y = pd.to_numeric(plot_df[f1_col], errors="coerce").fillna(np.nan).to_numpy()
        err_col = f"{f1_col}_ci95" if f"{f1_col}_ci95" in plot_df.columns else None
        yerr = pd.to_numeric(plot_df[err_col], errors="coerce").fillna(0.0).to_numpy() if err_col else None
        axes[1].bar(x, y, color=colors, edgecolor="black", linewidth=0.6, yerr=yerr, capsize=2)
        axes[1].set_title("Main Comparison: F1")
        axes[1].set_ylabel("F1")
        axes[1].set_xticks(x, plot_df["plot_label"], rotation=18, ha="right")
        axes[1].set_ylim(max(0.0, np.nanmin(y) - 0.05), min(1.0, np.nanmax(y) + 0.05))
        axes[1].grid(axis="y", linestyle="--", alpha=0.3)
    else:
        axes[1].axis("off")

    fig.tight_layout(w_pad=1.6)
    return _save_figure(fig, out_dir, "Figure_top_journal_main_comparison")


def _make_protocol_figure(combined_df: pd.DataFrame, out_dir: str) -> Optional[str]:
    if plt is None or combined_df.empty:
        return None
    protocol_df = combined_df[combined_df["component"] == "protocol_sensitivity"].copy()
    if protocol_df.empty or "protocol_id" not in protocol_df.columns or "variant" not in protocol_df.columns:
        return None
    auc_col = _resolve_metric_col(protocol_df, ["AUC", "test_AUC"])
    if auc_col is None:
        return None

    protocol_order = ["random_edge", "node_split", "strict_herb_cold_start"]
    variant_order = ["hgt_baseline", "static_similarity_hgt", "proposed_model"]
    protocol_labels = {
        "random_edge": "Random edge",
        "node_split": "Node split",
        "strict_herb_cold_start": "Strict herb cold-start",
    }
    subset = protocol_df[protocol_df["protocol_id"].isin(protocol_order) & protocol_df["variant"].isin(variant_order)].copy()
    if subset.empty:
        return None

    _apply_figure_style()
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    x = np.arange(len(protocol_order))
    width = 0.24
    colors = {
        "hgt_baseline": "#7F8C8D",
        "static_similarity_hgt": "#5DADE2",
        "proposed_model": "#D35400",
    }
    for idx, variant in enumerate(variant_order):
        values = []
        errors = []
        for protocol_id in protocol_order:
            row = subset[(subset["protocol_id"] == protocol_id) & (subset["variant"] == variant)]
            if row.empty:
                values.append(np.nan)
                errors.append(0.0)
                continue
            values.append(float(pd.to_numeric(row.iloc[0][auc_col], errors="coerce")))
            ci_col = f"{auc_col}_ci95"
            errors.append(float(pd.to_numeric(row.iloc[0].get(ci_col, 0.0), errors="coerce")))
        ax.bar(x + (idx - 1) * width, values, width=width, color=colors[variant], edgecolor="black", linewidth=0.6, yerr=errors, capsize=2, label=DISPLAY_NAME_MAP.get(variant, variant))

    ax.set_title("Protocol Sensitivity Under Different Split Definitions")
    ax.set_ylabel("AUC")
    ax.set_xticks(x, [protocol_labels[item] for item in protocol_order], rotation=10, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.tight_layout()
    return _save_figure(fig, out_dir, "Figure_protocol_sensitivity_summary")


def _make_mechanism_figure(mechanism_df: pd.DataFrame, out_dir: str) -> Optional[str]:
    if plt is None or mechanism_df.empty or "group" not in mechanism_df.columns:
        return None
    metric_col = _resolve_metric_col(mechanism_df, ["F1", "test_F1"])
    if metric_col is None:
        return None

    group_order = [
        "baseline",
        "ingredient_replacement",
        "attention_pruning:top",
        "attention_pruning:bottom",
        "attention_pruning:random",
    ]
    labels = {
        "baseline": "Baseline",
        "ingredient_replacement": "Ingredient replacement",
        "attention_pruning:top": "Top pruning",
        "attention_pruning:bottom": "Bottom pruning",
        "attention_pruning:random": "Random pruning",
    }
    plot_df = mechanism_df[mechanism_df["group"].isin(group_order)].copy()
    if plot_df.empty:
        return None
    plot_df["plot_order"] = plot_df["group"].apply(group_order.index)
    plot_df = plot_df.sort_values("plot_order").reset_index(drop=True)

    _apply_figure_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))

    x = np.arange(len(plot_df))
    f1_vals = pd.to_numeric(plot_df[metric_col], errors="coerce").fillna(np.nan).to_numpy()
    colors = ["#2E86C1", "#48C9B0", "#CB4335", "#85929E", "#AF7AC5"][: len(plot_df)]
    axes[0].bar(x, f1_vals, color=colors, edgecolor="black", linewidth=0.6)
    axes[0].set_title("Ingredient Validation: F1")
    axes[0].set_ylabel("F1")
    axes[0].set_xticks(x, [labels.get(item, item) for item in plot_df["group"]], rotation=18, ha="right")
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)

    if "avg_pos_logit_drop" in plot_df.columns:
        drop_vals = pd.to_numeric(plot_df["avg_pos_logit_drop"], errors="coerce").fillna(0.0).to_numpy()
        axes[1].bar(x, drop_vals, color=colors, edgecolor="black", linewidth=0.6)
        axes[1].set_title("Positive Logit Drop")
        axes[1].set_ylabel("Average logit drop")
        axes[1].set_xticks(x, [labels.get(item, item) for item in plot_df["group"]], rotation=18, ha="right")
        axes[1].grid(axis="y", linestyle="--", alpha=0.3)
    else:
        axes[1].axis("off")

    fig.tight_layout(w_pad=1.4)
    return _save_figure(fig, out_dir, "Figure_ingredient_mechanism_summary")


def _export_summary_figures(paths: dict[str, str], combined_df: pd.DataFrame, main_table_df: pd.DataFrame, mechanism_df: pd.DataFrame) -> dict[str, Optional[str]]:
    figure_dir = paths["figures"]
    outputs = {
        "main_comparison": _make_main_table_figure(main_table_df, figure_dir),
        "protocol_sensitivity": _make_protocol_figure(combined_df, figure_dir),
        "ingredient_mechanism": _make_mechanism_figure(mechanism_df, figure_dir),
    }
    manifest_path = os.path.join(figure_dir, "supplementary_suite_figure_manifest.json")
    _write_json(manifest_path, outputs)
    outputs["manifest"] = manifest_path
    return outputs


def aggregate_outputs(paths: dict[str, str], output_root: str, data_path: str, seeds: list[int], min_seed_count: int, require_complete: bool) -> dict[str, str]:
    catalog = _component_catalog(output_root=output_root, data_path=data_path, seeds=seeds, min_seed_count=min_seed_count, require_complete=require_complete)
    combined_rows = []
    for item in catalog:
        frame = _safe_read_csv(item["result_csv"])
        if frame.empty:
            continue
        frame = frame.copy()
        frame["component"] = item["component"]
        frame["component_title"] = item["title"]
        frame["component_category"] = item["category"]
        combined_rows.append(frame)

    combined_df = pd.concat(combined_rows, ignore_index=True) if combined_rows else pd.DataFrame()
    combined_csv = os.path.join(paths["root"], "supplementary_combined_results.csv")
    combined_json = os.path.join(paths["root"], "supplementary_combined_results.json")
    combined_df.to_csv(combined_csv, index=False, encoding="utf-8-sig")
    _write_json(combined_json, combined_df.to_dict(orient="records") if not combined_df.empty else [])

    main_rows = []
    if not combined_df.empty:
        protocol_df = combined_df[combined_df["component"] == "protocol_sensitivity"].copy()
        if not protocol_df.empty and "protocol_id" in protocol_df.columns:
            protocol_df = protocol_df[protocol_df["protocol_id"] == "strict_herb_cold_start"].copy()
            preferred_variants = {"proposed_model", "hgt_baseline", "static_similarity_hgt"}
            if "variant" in protocol_df.columns:
                protocol_df = protocol_df[protocol_df["variant"].isin(preferred_variants)].copy()
            if not protocol_df.empty:
                protocol_df["main_table_source"] = "internal_strict"
                main_rows.append(protocol_df)

        herb_only_df = combined_df[combined_df["component"] == "strong_herb_only_strict"].copy()
        if not herb_only_df.empty:
            herb_only_df["main_table_source"] = "internal_control"
            main_rows.append(herb_only_df)

        htinet2_df = combined_df[combined_df["component"] == "htinet2_strict"].copy()
        if not htinet2_df.empty:
            htinet2_df["main_table_source"] = "external_strict"
            main_rows.append(htinet2_df)

    main_table_df = pd.concat(main_rows, ignore_index=True) if main_rows else pd.DataFrame()
    main_table_csv = os.path.join(paths["root"], "top_journal_main_table_view.csv")
    main_table_json = os.path.join(paths["root"], "top_journal_main_table_view.json")
    main_table_df.to_csv(main_table_csv, index=False, encoding="utf-8-sig")
    _write_json(main_table_json, main_table_df.to_dict(orient="records") if not main_table_df.empty else [])

    supplementary_df = combined_df[
        combined_df["component"].isin(["protocol_sensitivity", "strong_herb_only_strict", "ingredient_validation", "htinet2_strict"])
    ].copy() if not combined_df.empty else pd.DataFrame()
    supplementary_csv = os.path.join(paths["root"], "supplementary_table_view.csv")
    supplementary_json = os.path.join(paths["root"], "supplementary_table_view.json")
    supplementary_df.to_csv(supplementary_csv, index=False, encoding="utf-8-sig")
    _write_json(supplementary_json, supplementary_df.to_dict(orient="records") if not supplementary_df.empty else [])

    mechanism_df = combined_df[
        combined_df["component"].isin(["ingredient_validation"])
    ].copy() if not combined_df.empty else pd.DataFrame()
    mechanism_csv = os.path.join(paths["root"], "mechanism_validation_view.csv")
    mechanism_json = os.path.join(paths["root"], "mechanism_validation_view.json")
    mechanism_df.to_csv(mechanism_csv, index=False, encoding="utf-8-sig")
    _write_json(mechanism_json, mechanism_df.to_dict(orient="records") if not mechanism_df.empty else [])

    control_df = combined_df[
        combined_df["component"].isin(["strong_herb_only_strict"])
    ].copy() if not combined_df.empty else pd.DataFrame()
    control_csv = os.path.join(paths["root"], "control_baseline_view.csv")
    control_json = os.path.join(paths["root"], "control_baseline_view.json")
    control_df.to_csv(control_csv, index=False, encoding="utf-8-sig")
    _write_json(control_json, control_df.to_dict(orient="records") if not control_df.empty else [])

    manifest_rows = []
    for item in catalog:
        manifest_rows.append({
            "component": item["component"],
            "title": item["title"],
            "category": item["category"],
            "result_csv": item["result_csv"],
            "exists": os.path.exists(item["result_csv"]),
        })
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_csv = os.path.join(paths["root"], "supplementary_component_manifest.csv")
    manifest_json = os.path.join(paths["root"], "supplementary_component_manifest.json")
    manifest_df.to_csv(manifest_csv, index=False, encoding="utf-8-sig")
    _write_json(manifest_json, manifest_df.to_dict(orient="records"))

    figure_outputs = _export_summary_figures(paths, combined_df, main_table_df, mechanism_df)

    outputs = {
        "supplementary_combined_results.csv": combined_csv,
        "supplementary_combined_results.json": combined_json,
        "top_journal_main_table_view.csv": main_table_csv,
        "top_journal_main_table_view.json": main_table_json,
        "supplementary_table_view.csv": supplementary_csv,
        "supplementary_table_view.json": supplementary_json,
        "mechanism_validation_view.csv": mechanism_csv,
        "mechanism_validation_view.json": mechanism_json,
        "control_baseline_view.csv": control_csv,
        "control_baseline_view.json": control_json,
        "supplementary_component_manifest.csv": manifest_csv,
        "supplementary_component_manifest.json": manifest_json,
    }
    for key, value in figure_outputs.items():
        if value:
            outputs[f"figure::{key}"] = value
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-command runner for the core supplementary experiments, result aggregation, and figure export."
    )
    parser.add_argument(
        "--mode",
        choices=["run", "report", "all"],
        default="all",
        help="run: execute all component scripts, report: aggregate existing outputs, all: run then aggregate.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="Path to processed hetero_graph.pt used by the component experiments.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Single seed. Use --seeds for a custom seed list.",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Seed list, e.g. --seeds 42 43 44 45 46",
    )
    parser.add_argument(
        "--min-seed-count",
        type=int,
        default=1,
        help="Minimum seed count required by completeness gates.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Stop early when a required component fails its own completeness gate.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help="Top-level directory used to store all supplementary experiment outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = normalize_seeds(seeds=args.seeds, seed=args.seed)
    paths = ensure_dirs(args.output_root)

    if args.mode in ("run", "all"):
        run_all_components(
            paths=paths,
            output_root=args.output_root,
            data_path=args.data_path,
            seeds=seeds,
            min_seed_count=int(args.min_seed_count),
            require_complete=bool(args.require_complete),
        )
    if args.mode in ("report", "all"):
        outputs = aggregate_outputs(
            paths=paths,
            output_root=args.output_root,
            data_path=args.data_path,
            seeds=seeds,
            min_seed_count=int(args.min_seed_count),
            require_complete=bool(args.require_complete),
        )
        print("\nSaved outputs:")
        for _, path in sorted(outputs.items()):
            print(f"  {path}")


if __name__ == "__main__":
    main()



