from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import sys
import traceback
from collections import defaultdict
from typing import Any, Callable, Optional

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
DEFAULT_OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "checkpoints", "protocol_sensitivity")
DEFAULT_VARIANT_KEYS = ["hgt_baseline", "static_similarity_hgt", "proposed_model"]
DEFAULT_PROTOCOLS = ["random_edge", "node_split", "strict_herb_cold_start"]
DEFAULT_MIN_TOP_JOURNAL_SEEDS = 1

PROTOCOL_META = {
    "random_edge": {
        "display_name": "Random Edge Split",
        "order": 10,
        "description": "Random H-T edge split with all herb-ingredient edges visible during message passing.",
    },
    "node_split": {
        "display_name": "Node Split",
        "order": 20,
        "description": "Herb-level split without filtering validation/test herb composition edges from the message graph.",
    },
    "strict_herb_cold_start": {
        "display_name": "Strict Herb-Level Cold Start",
        "order": 30,
        "description": "Herb-level split with ingredient-target masking and train-only herb-ingredient message passing.",
    },
}

_CSS = None
_TORCH = None


def _deps() -> tuple[Any, Any]:
    global _CSS, _TORCH
    if _CSS is None or _TORCH is None:
        import torch
        import cold_start_split as css

        _TORCH = torch
        _CSS = css
    return _CSS, _TORCH


def _write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
        "protocol_sensitivity_suite.py",
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
    path = os.path.join(paths["root"], "protocol_sensitivity_code_provenance.json")
    _write_json(path, rows)
    return path


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


def normalize_seeds(seeds: Optional[list[int]] = None, seed: Optional[int] = None) -> list[int]:
    return es.normalize_seeds(seeds=seeds, seed=seed)


def _protocol_variant_root(paths: dict[str, str], protocol_id: str, variant_key: str) -> str:
    return os.path.join(paths["runs"], protocol_id, variant_key)


def _protocol_seed_dir(paths: dict[str, str], protocol_id: str, variant_key: str, seed: int) -> str:
    return os.path.join(_protocol_variant_root(paths, protocol_id, variant_key), f"seed_{int(seed)}")


def _summary_path(paths: dict[str, str], protocol_id: str, variant_key: str, seed: Optional[int] = None) -> str:
    if seed is None:
        return os.path.join(_protocol_variant_root(paths, protocol_id, variant_key), "summary.json")
    return os.path.join(_protocol_seed_dir(paths, protocol_id, variant_key, seed), "summary.json")


def _error_path(paths: dict[str, str], protocol_id: str, variant_key: str, seed: int) -> str:
    return os.path.join(_protocol_seed_dir(paths, protocol_id, variant_key, seed), "error.json")


def _suite_config_path(paths: dict[str, str]) -> str:
    return os.path.join(paths["root"], "suite_config.json")


def write_suite_config(
    paths: dict[str, str],
    seeds: list[int],
    protocols: list[str],
    variants: list[str],
    min_seed_count: int = DEFAULT_MIN_TOP_JOURNAL_SEEDS,
) -> dict:
    payload = {
        "protocol_version": "protocol_sensitivity_v1",
        "expected_seeds": [int(item) for item in seeds],
        "expected_seed_count": len(seeds),
        "min_top_journal_seed_count": int(min_seed_count),
        "protocols": list(protocols),
        "variants": list(variants),
        "threshold_policy": es.STRICT_THRESHOLD_POLICY,
        "notes": [
            "All protocol rows reuse the same training loop and evaluation policy.",
            "Only the train/validation/test split builder changes across protocols.",
            "The strict protocol reuses cold_start_split.cold_start_split without modification.",
        ],
    }
    _write_json(_suite_config_path(paths), payload)
    return payload


def _find_edge_type(data, src_type: str, dst_type: str):
    for edge_type in data.edge_types:
        s, _, d = edge_type
        if s == src_type and d == dst_type:
            return edge_type
    return None


def _clone_tensor(values: list[int]) -> torch.Tensor:
    _, torch = _deps()
    if not values:
        return torch.empty(0, dtype=torch.long)
    return torch.tensor(values, dtype=torch.long)


def _build_message_graph(
    data,
    hi_etype,
    it_etype,
    train_it_edge,
    herb_to_ings: dict[int, list[int]],
    train_herbs_for_hi: Optional[set[int]],
) -> tuple[object, int, int]:
    _, torch = _deps()
    msg_graph = data.clone()

    hi_edge_full = msg_graph[hi_etype].edge_index
    if train_herbs_for_hi is not None:
        train_herbs_tensor = torch.tensor(sorted(train_herbs_for_hi), dtype=torch.long)
        hi_keep_mask = torch.isin(hi_edge_full[0], train_herbs_tensor)
    else:
        hi_keep_mask = torch.ones(hi_edge_full.size(1), dtype=torch.bool)
    msg_graph[hi_etype].edge_index = hi_edge_full[:, hi_keep_mask]

    msg_graph[it_etype].edge_index = train_it_edge

    for edge_type in list(msg_graph.edge_types):
        s, _, d = edge_type
        if (s == "herb" and d == "target") or (s == "target" and d == "herb"):
            del msg_graph[edge_type]

    existing_pairs = {(s, d) for s, _, d in msg_graph.edge_types}
    for edge_type in list(msg_graph.edge_types):
        s, rel, d = edge_type
        if (d, s) in existing_pairs:
            continue
        rev_rel = f"rev_{rel}"
        msg_graph[d, rev_rel, s].edge_index = msg_graph[edge_type].edge_index.flip(0)
        existing_pairs.add((d, s))

    kept = int(hi_keep_mask.sum().item())
    removed = int((~hi_keep_mask).sum().item())
    return msg_graph, kept, removed


def _split_herbs(all_herbs: list[int], val_ratio: float, test_ratio: float, py_rng: np.random.Generator):
    herbs = list(all_herbs)
    py_rng.shuffle(herbs)
    n_val = max(1, int(len(herbs) * val_ratio))
    n_test = max(1, int(len(herbs) * test_ratio))
    n_train = len(herbs) - n_val - n_test
    if n_train <= 0:
        raise ValueError(f"Invalid herb split: total={len(herbs)} val={n_val} test={n_test}")
    train_herbs = set(herbs[:n_train])
    val_herbs = set(herbs[n_train:n_train + n_val])
    test_herbs = set(herbs[n_train + n_val:])
    return train_herbs, val_herbs, test_herbs


def _core_components(data, *, seed: int, it_mask_ratio: float, use_it_mask: bool):
    css, torch = _deps()
    np_rng = np.random.default_rng(int(seed))
    py_rng = np.random.default_rng(int(seed) + 17)

    hi_etype = _find_edge_type(data, "herb", "ingredient")
    it_etype = _find_edge_type(data, "ingredient", "target")
    if hi_etype is None or it_etype is None:
        raise ValueError(f"Missing herb-ingredient or ingredient-target edge type: {data.edge_types}")

    hi_edge = data[hi_etype].edge_index
    it_edge = data[it_etype].edge_index

    herb_to_ings_set = css._build_mapping(hi_edge)
    herb_to_ings = {int(h): sorted(int(ing) for ing in ings) for h, ings in herb_to_ings_set.items()}

    n_it = int(it_edge.size(1))
    perm = np_rng.permutation(n_it)
    if use_it_mask and n_it > 0:
        n_masked = max(1, int(n_it * float(it_mask_ratio)))
        n_train = max(0, n_it - n_masked)
        train_it_idx = torch.from_numpy(perm[:n_train].copy())
        masked_it_idx = torch.from_numpy(perm[n_train:].copy())
        train_it_edge = it_edge[:, train_it_idx]
        masked_it_edge = it_edge[:, masked_it_idx]
    else:
        n_masked = 0
        n_train = n_it
        train_it_edge = it_edge
        masked_it_edge = it_edge[:, :0]

    train_ing_to_tgts = css._build_mapping(train_it_edge)
    masked_ing_to_tgts = css._build_mapping(masked_it_edge)
    full_ing_to_tgts = css._build_mapping(it_edge)

    train_ht_all = css._derive_ht(herb_to_ings_set, train_ing_to_tgts)
    all_ht_pos = css._derive_ht(herb_to_ings_set, full_ing_to_tgts)
    masked_ht_all = css._derive_ht(herb_to_ings_set, masked_ing_to_tgts) if use_it_mask else all_ht_pos
    pure_test_ht = masked_ht_all - train_ht_all

    return {
        "np_rng": np_rng,
        "py_rng": py_rng,
        "hi_etype": hi_etype,
        "it_etype": it_etype,
        "herb_to_ings": herb_to_ings,
        "herb_to_ings_set": herb_to_ings_set,
        "train_it_edge": train_it_edge,
        "train_ing_to_tgts": train_ing_to_tgts,
        "train_ht_all": train_ht_all,
        "all_ht_pos": all_ht_pos,
        "masked_ht_all": masked_ht_all,
        "pure_test_ht": pure_test_ht,
        "n_train_it": n_train,
        "n_masked_it": n_masked,
        "num_targets": int(data["target"].num_nodes),
    }


def _bundle_from_pairs(
    *,
    data,
    components: dict,
    train_ht: list[tuple[int, int]],
    val_ht: list[tuple[int, int]],
    test_ht: list[tuple[int, int]],
    train_herbs: set[int],
    val_herbs: set[int],
    test_herbs: set[int],
    neg_ratio: float,
    strict_eval_neg_filter: bool,
    val_hard_neg_ratio: float,
    val_pop_neg_ratio: float,
    test_hard_neg_ratio: float,
    test_pop_neg_ratio: float,
    filter_hi_to_train_herbs: bool,
    split_id: str,
):
    css, _ = _deps()
    herb_to_ings = components["herb_to_ings"]
    all_ht_pos = components["all_ht_pos"]
    train_ht_all = components["train_ht_all"]
    train_ing_to_tgts = components["train_ing_to_tgts"]
    num_targets = components["num_targets"]
    np_rng = components["np_rng"]
    py_rng = components["py_rng"]

    val_pos_filter = all_ht_pos if strict_eval_neg_filter else train_ht_all
    test_pos_filter = all_ht_pos if strict_eval_neg_filter else train_ht_all

    val_neg = css._sample_ht_negatives(
        pos_pairs=val_ht,
        all_pos_global=val_pos_filter,
        herb_set=val_herbs,
        herb_to_ings=herb_to_ings,
        ing_to_tgts=train_ing_to_tgts,
        num_tgts=num_targets,
        neg_ratio=neg_ratio,
        hard_neg_ratio=val_hard_neg_ratio,
        pop_neg_ratio=val_pop_neg_ratio,
        rng=np_rng,
        py_rng=py_rng,
        name=f"{split_id}_val",
    )
    test_neg = css._sample_ht_negatives(
        pos_pairs=test_ht,
        all_pos_global=test_pos_filter,
        herb_set=test_herbs,
        herb_to_ings=herb_to_ings,
        ing_to_tgts=train_ing_to_tgts,
        num_tgts=num_targets,
        neg_ratio=neg_ratio,
        hard_neg_ratio=test_hard_neg_ratio,
        pop_neg_ratio=test_pop_neg_ratio,
        rng=np_rng,
        py_rng=py_rng,
        name=f"{split_id}_test",
    )

    hi_train_filter = train_herbs if filter_hi_to_train_herbs else None
    msg_graph, hi_kept, hi_removed = _build_message_graph(
        data=data,
        hi_etype=components["hi_etype"],
        it_etype=components["it_etype"],
        train_it_edge=components["train_it_edge"],
        herb_to_ings=herb_to_ings,
        train_herbs_for_hi=hi_train_filter,
    )

    val_neg_false_neg = sum(1 for pair in val_neg if pair in all_ht_pos)
    test_neg_false_neg = sum(1 for pair in test_neg if pair in all_ht_pos)
    stats = {
        "split_protocol": split_id,
        "num_train_herbs": len(train_herbs),
        "num_val_herbs": len(val_herbs),
        "num_test_herbs": len(test_herbs),
        "num_train_ht": len(train_ht),
        "num_val_ht": len(val_ht),
        "num_test_ht": len(test_ht),
        "num_val_neg": len(val_neg),
        "num_test_neg": len(test_neg),
        "num_train_it_edges": int(components["n_train_it"]),
        "num_masked_it_edges": int(components["n_masked_it"]),
        "num_hi_kept": hi_kept,
        "num_hi_removed": hi_removed,
        "val_neg_false_neg": int(val_neg_false_neg),
        "test_neg_false_neg": int(test_neg_false_neg),
        "val_neg_false_neg_rate": float(val_neg_false_neg / max(1, len(val_neg))),
        "test_neg_false_neg_rate": float(test_neg_false_neg / max(1, len(test_neg))),
        "val_pos_neg_ratio": float(len(val_ht) / max(1, len(val_neg))),
        "test_pos_neg_ratio": float(len(test_ht) / max(1, len(test_neg))),
    }

    return css.HTSplitBundle(
        msg_graph=msg_graph,
        train_pos_h=_clone_tensor([h for h, _ in train_ht]),
        train_pos_t=_clone_tensor([t for _, t in train_ht]),
        val_pos_h=_clone_tensor([h for h, _ in val_ht]),
        val_pos_t=_clone_tensor([t for _, t in val_ht]),
        val_neg_h=_clone_tensor([h for h, _ in val_neg]),
        val_neg_t=_clone_tensor([t for _, t in val_neg]),
        test_pos_h=_clone_tensor([h for h, _ in test_ht]),
        test_pos_t=_clone_tensor([t for _, t in test_ht]),
        test_neg_h=_clone_tensor([h for h, _ in test_neg]),
        test_neg_t=_clone_tensor([t for _, t in test_neg]),
        herb_to_ings=herb_to_ings,
        all_ht_pos=all_ht_pos,
        train_herbs=train_herbs,
        val_herbs=val_herbs,
        test_herbs=test_herbs,
        stats=stats,
    )


def _node_split_builder(original_builder: Callable):
    def builder(
        data,
        *,
        val_ratio: float,
        test_ratio: float,
        neg_ratio: float,
        hard_neg_ratio: float,
        pop_neg_ratio: float,
        it_mask_ratio: float,
        seed: int,
        strict_eval_neg_filter: bool,
        val_hard_neg_ratio: float,
        val_pop_neg_ratio: float,
        test_hard_neg_ratio: float,
        test_pop_neg_ratio: float,
        min_pos_per_herb_val: int,
        min_pos_per_herb_test: int,
        max_fallback_per_herb: int,
        use_it_mask: bool = True,
        filter_hi_to_train_herbs: bool = True,
    ):
        css, _ = _deps()
        components = _core_components(
            data,
            seed=int(seed),
            it_mask_ratio=float(it_mask_ratio),
            use_it_mask=bool(use_it_mask),
        )
        train_herb_to_tgts = defaultdict(set)
        for herb, target in components["train_ht_all"]:
            train_herb_to_tgts[int(herb)].add(int(target))

        all_herbs = sorted(train_herb_to_tgts.keys())
        train_herbs, val_herbs, test_herbs = _split_herbs(
            all_herbs,
            val_ratio=float(val_ratio),
            test_ratio=float(test_ratio),
            py_rng=components["py_rng"],
        )

        train_ht = [(h, t) for h in train_herbs for t in train_herb_to_tgts.get(h, set())]
        pure_test_by_herb = defaultdict(set)
        masked_by_herb = defaultdict(set)
        for h, t in components["pure_test_ht"]:
            pure_test_by_herb[int(h)].add(int(t))
        for h, t in components["masked_ht_all"]:
            masked_by_herb[int(h)].add(int(t))

        val_ht = [(h, t) for h in val_herbs for t in pure_test_by_herb.get(h, set())]
        test_ht = [(h, t) for h in test_herbs for t in pure_test_by_herb.get(h, set())]
        if not val_ht:
            val_ht = [(h, t) for h in val_herbs for t in masked_by_herb.get(h, set())]
        if not test_ht:
            test_ht = [(h, t) for h in test_herbs for t in masked_by_herb.get(h, set())]

        val_ht, _ = css._enforce_min_pos_per_herb(
            base_pairs=val_ht,
            herb_set=val_herbs,
            pure_by_herb=pure_test_by_herb,
            masked_by_herb=masked_by_herb,
            train_ht_all=components["train_ht_all"],
            min_pos_per_herb=min_pos_per_herb_val,
            max_fallback_per_herb=max_fallback_per_herb,
        )
        test_ht, _ = css._enforce_min_pos_per_herb(
            base_pairs=test_ht,
            herb_set=test_herbs,
            pure_by_herb=pure_test_by_herb,
            masked_by_herb=masked_by_herb,
            train_ht_all=components["train_ht_all"],
            min_pos_per_herb=min_pos_per_herb_test,
            max_fallback_per_herb=max_fallback_per_herb,
        )

        return _bundle_from_pairs(
            data=data,
            components=components,
            train_ht=train_ht,
            val_ht=val_ht,
            test_ht=test_ht,
            train_herbs=train_herbs,
            val_herbs=val_herbs,
            test_herbs=test_herbs,
            neg_ratio=float(neg_ratio),
            strict_eval_neg_filter=bool(strict_eval_neg_filter),
            val_hard_neg_ratio=float(val_hard_neg_ratio),
            val_pop_neg_ratio=float(val_pop_neg_ratio),
            test_hard_neg_ratio=float(test_hard_neg_ratio),
            test_pop_neg_ratio=float(test_pop_neg_ratio),
            filter_hi_to_train_herbs=False,
            split_id="node_split",
        )

    return builder


def _random_edge_builder(original_builder: Callable):
    def builder(
        data,
        *,
        val_ratio: float,
        test_ratio: float,
        neg_ratio: float,
        hard_neg_ratio: float,
        pop_neg_ratio: float,
        it_mask_ratio: float,
        seed: int,
        strict_eval_neg_filter: bool,
        val_hard_neg_ratio: float,
        val_pop_neg_ratio: float,
        test_hard_neg_ratio: float,
        test_pop_neg_ratio: float,
        min_pos_per_herb_val: int,
        min_pos_per_herb_test: int,
        max_fallback_per_herb: int,
        use_it_mask: bool = True,
        filter_hi_to_train_herbs: bool = True,
    ):
        _, _torch = _deps()
        components = _core_components(
            data,
            seed=int(seed),
            it_mask_ratio=float(it_mask_ratio),
            use_it_mask=bool(use_it_mask),
        )
        all_pairs = sorted(components["all_ht_pos"])
        if len(all_pairs) < 3:
            raise ValueError(f"Not enough H-T positives for random edge split: {len(all_pairs)}")

        perm = components["np_rng"].permutation(len(all_pairs))
        n_val = max(1, int(len(all_pairs) * float(val_ratio)))
        n_test = max(1, int(len(all_pairs) * float(test_ratio)))
        n_train = len(all_pairs) - n_val - n_test
        if n_train <= 0:
            raise ValueError(f"Invalid random-edge split sizes: total={len(all_pairs)}")

        train_idx = perm[:n_train]
        val_idx = perm[n_train:n_train + n_val]
        test_idx = perm[n_train + n_val:]

        train_ht = [all_pairs[int(i)] for i in train_idx.tolist()]
        val_ht = [all_pairs[int(i)] for i in val_idx.tolist()]
        test_ht = [all_pairs[int(i)] for i in test_idx.tolist()]

        train_herbs = {int(h) for h, _ in train_ht}
        val_herbs = {int(h) for h, _ in val_ht}
        test_herbs = {int(h) for h, _ in test_ht}

        return _bundle_from_pairs(
            data=data,
            components=components,
            train_ht=train_ht,
            val_ht=val_ht,
            test_ht=test_ht,
            train_herbs=train_herbs,
            val_herbs=val_herbs,
            test_herbs=test_herbs,
            neg_ratio=float(neg_ratio),
            strict_eval_neg_filter=bool(strict_eval_neg_filter),
            val_hard_neg_ratio=float(val_hard_neg_ratio),
            val_pop_neg_ratio=float(val_pop_neg_ratio),
            test_hard_neg_ratio=float(test_hard_neg_ratio),
            test_pop_neg_ratio=float(test_pop_neg_ratio),
            filter_hi_to_train_herbs=False,
            split_id="random_edge",
        )

    return builder


def get_protocol_builder(protocol_id: str, original_builder: Callable):
    if protocol_id == "strict_herb_cold_start":
        return original_builder
    if protocol_id == "node_split":
        return _node_split_builder(original_builder)
    if protocol_id == "random_edge":
        return _random_edge_builder(original_builder)
    raise KeyError(f"Unsupported protocol: {protocol_id}")


@contextlib.contextmanager
def patch_cold_start_split(protocol_id: str):
    import train

    original_builder = train.cold_start_split
    patched_builder = get_protocol_builder(protocol_id, original_builder)
    train.cold_start_split = patched_builder
    try:
        yield
    finally:
        train.cold_start_split = original_builder


def run_protocol_variant_seed(protocol_id: str, variant: dict, paths: dict[str, str], seed: int) -> dict:
    import train

    run_dir = _protocol_seed_dir(paths, protocol_id, variant["variant"], seed)
    os.makedirs(run_dir, exist_ok=True)

    train.reset_cfg()
    overrides = dict(variant.get("overrides", {}))
    overrides.update({
        "save_dir": run_dir,
        "seed": int(seed),
        "curve_alias": f"{protocol_id}_{variant['variant']}",
        "experiment_name": variant["variant"],
        "experiment_group": variant["group"],
        "experiment_title": variant["title"],
        "rl_save_path": os.path.join(run_dir, "rl_agent.pt"),
    })
    overrides.update(es.STRICT_SPLIT_CFG)
    train.merge_cfg(overrides)

    try:
        with patch_cold_start_split(protocol_id):
            summary = train.main()
        summary.update({
            "protocol_id": protocol_id,
            "protocol_display_name": PROTOCOL_META[protocol_id]["display_name"],
            "protocol_order": PROTOCOL_META[protocol_id]["order"],
            "display_name": variant["display_name"],
            "status": "ok",
            "source": "internal",
            "order": variant["order"],
            "seed": int(seed),
            "strict_protocol_version": "protocol_sensitivity_v1",
            "split_id": protocol_id,
            "threshold_policy": es.STRICT_THRESHOLD_POLICY,
        })
        _write_json(_summary_path(paths, protocol_id, variant["variant"], seed), summary)
        return summary
    except Exception as exc:
        payload = {
            "variant": variant["variant"],
            "group": variant["group"],
            "title": variant["title"],
            "display_name": variant["display_name"],
            "protocol_id": protocol_id,
            "protocol_display_name": PROTOCOL_META[protocol_id]["display_name"],
            "protocol_order": PROTOCOL_META[protocol_id]["order"],
            "save_dir": run_dir,
            "status": "failed",
            "source": "internal",
            "order": variant["order"],
            "seed": int(seed),
            "strict_protocol_version": "protocol_sensitivity_v1",
            "split_id": protocol_id,
            "threshold_policy": es.STRICT_THRESHOLD_POLICY,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(_error_path(paths, protocol_id, variant["variant"], seed), payload)
        return payload


def _aggregate(records: list[dict], expected_seeds: list[int]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()

    frame = pd.DataFrame(records).copy()
    frame = es._normalize_metric_columns(frame)
    aggregated_rows = []

    for (protocol_id, variant_key), group in frame.groupby(["protocol_id", "variant"], sort=False):
        group = group.sort_values(["seed", "order"], na_position="last").reset_index(drop=True)
        ok_group = group[group["status"] == "ok"].copy()
        base_row = ok_group.iloc[0] if not ok_group.empty else group.iloc[0]
        completed_seeds = sorted({int(item) for item in ok_group["seed"].dropna().astype(int).tolist()})
        strict_ready = completed_seeds == sorted(expected_seeds)
        row = {
            "protocol_id": protocol_id,
            "protocol_display_name": base_row.get("protocol_display_name", protocol_id),
            "protocol_order": int(base_row.get("protocol_order", 999)),
            "variant": variant_key,
            "display_name": base_row.get("display_name", variant_key),
            "group": base_row.get("group", ""),
            "title": base_row.get("title", ""),
            "order": float(base_row.get("order", 9999)),
            "status": "ok" if not ok_group.empty else "failed",
            "threshold_policy": es.STRICT_THRESHOLD_POLICY,
            "expected_seeds": ",".join(str(item) for item in expected_seeds),
            "completed_seeds": ",".join(str(item) for item in completed_seeds),
            "strict_ready": bool(strict_ready),
            "error_count": int((group["status"] != "ok").sum()),
        }
        for metric in es.STRICT_METRIC_COLUMNS:
            values = pd.to_numeric(ok_group.get(metric, pd.Series(dtype=float)), errors="coerce").dropna()
            if values.empty:
                continue
            row[metric] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            if len(values) > 1:
                ci95 = 1.96 * row[f"{metric}_std"] / np.sqrt(len(values))
            else:
                ci95 = 0.0
            row[f"{metric}_ci95"] = float(ci95)
            row[f"{metric}_mean_std"] = es._format_mean_std(row[metric], row[f"{metric}_std"])
        aggregated_rows.append(row)

    aggregated_df = pd.DataFrame(aggregated_rows)
    if aggregated_df.empty:
        return aggregated_df
    return aggregated_df.sort_values(["protocol_order", "order", "variant"]).reset_index(drop=True)


def _export_tables(run_df: pd.DataFrame, aggregated_df: pd.DataFrame, paths: dict[str, str]) -> dict[str, str]:
    outputs = {}

    run_csv = os.path.join(paths["root"], "protocol_sensitivity_runs.csv")
    run_json = os.path.join(paths["root"], "protocol_sensitivity_runs.json")
    run_df.to_csv(run_csv, index=False, encoding="utf-8-sig")
    _write_json(run_json, run_df.to_dict(orient="records"))
    outputs["protocol_sensitivity_runs.csv"] = run_csv
    outputs["protocol_sensitivity_runs.json"] = run_json

    agg_csv = os.path.join(paths["root"], "protocol_sensitivity_results.csv")
    agg_json = os.path.join(paths["root"], "protocol_sensitivity_results.json")
    aggregated_df.to_csv(agg_csv, index=False, encoding="utf-8-sig")
    _write_json(agg_json, aggregated_df.to_dict(orient="records"))
    outputs["protocol_sensitivity_results.csv"] = agg_csv
    outputs["protocol_sensitivity_results.json"] = agg_json
    return outputs


def _export_integrity_audit(run_df: pd.DataFrame, aggregated_df: pd.DataFrame, suite_config: dict, paths: dict[str, str]) -> Optional[str]:
    expected_protocols = [str(item) for item in suite_config.get("protocols", DEFAULT_PROTOCOLS)]
    expected_variants = [str(item) for item in suite_config.get("variants", DEFAULT_VARIANT_KEYS)]
    expected_seeds = [int(item) for item in suite_config.get("expected_seeds", [42])]
    min_seed_count = int(suite_config.get("min_top_journal_seed_count", DEFAULT_MIN_TOP_JOURNAL_SEEDS))

    rows = []
    for protocol_id in expected_protocols:
        for variant_key in expected_variants:
            ok_group = run_df[
                (run_df.get("protocol_id") == protocol_id)
                & (run_df.get("variant") == variant_key)
                & (run_df.get("status") == "ok")
            ].copy() if not run_df.empty else pd.DataFrame()
            completed_seeds = sorted({
                int(item)
                for item in pd.to_numeric(ok_group.get("seed", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).tolist()
            })
            row = {
                "protocol_id": protocol_id,
                "protocol_display_name": PROTOCOL_META[protocol_id]["display_name"],
                "variant": variant_key,
                "display_name": es.VARIANT_INDEX[variant_key]["display_name"],
                "expected_seeds": ",".join(str(item) for item in expected_seeds),
                "completed_seeds": ",".join(str(item) for item in completed_seeds),
                "expected_seed_count": len(expected_seeds),
                "completed_seed_count": len(completed_seeds),
                "complete": completed_seeds == sorted(expected_seeds),
                "meets_min_seed_count": len(completed_seeds) >= min_seed_count,
                "top_journal_ready": (completed_seeds == sorted(expected_seeds)) and (len(completed_seeds) >= min_seed_count),
                "threshold_policy": suite_config.get("threshold_policy", es.STRICT_THRESHOLD_POLICY),
            }
            rows.append(row)

    audit_df = pd.DataFrame(rows)
    audit_path = os.path.join(paths["root"], "protocol_sensitivity_integrity_audit.csv")
    audit_df.to_csv(audit_path, index=False, encoding="utf-8-sig")
    return audit_path


def _export_report_summary(aggregated_df: pd.DataFrame, suite_config: dict, paths: dict[str, str]) -> str:
    expected_protocols = [str(item) for item in suite_config.get("protocols", DEFAULT_PROTOCOLS)]
    expected_variants = [str(item) for item in suite_config.get("variants", DEFAULT_VARIANT_KEYS)]
    min_seed_count = int(suite_config.get("min_top_journal_seed_count", DEFAULT_MIN_TOP_JOURNAL_SEEDS))
    payload = {
        "protocol_count": len(expected_protocols),
        "variant_count": len(expected_variants),
        "expected_rows": len(expected_protocols) * len(expected_variants),
        "observed_rows": int(len(aggregated_df)),
        "strict_ready_rows": int(aggregated_df["strict_ready"].sum()) if (not aggregated_df.empty and "strict_ready" in aggregated_df.columns) else 0,
        "min_top_journal_seed_count": min_seed_count,
        "top_journal_ready_rows": int(((aggregated_df["completed_seed_count"] >= min_seed_count) & (aggregated_df["strict_ready"])).sum())
        if (not aggregated_df.empty and {"completed_seed_count", "strict_ready"}.issubset(aggregated_df.columns))
        else 0,
    }
    path = os.path.join(paths["root"], "protocol_sensitivity_report_summary.json")
    _write_json(path, payload)
    return path


def _export_significance(run_df: pd.DataFrame, paths: dict[str, str]) -> Optional[str]:
    if run_df.empty or "status" not in run_df.columns:
        return None
    subset = run_df[run_df["status"] == "ok"].copy()
    if subset.empty:
        return None

    rows = []
    comparisons = [
        ("strict_herb_cold_start", "random_edge"),
        ("strict_herb_cold_start", "node_split"),
    ]
    for variant_key in DEFAULT_VARIANT_KEYS:
        variant_df = subset[subset["variant"] == variant_key].copy()
        for lhs_protocol, rhs_protocol in comparisons:
            lhs = variant_df[variant_df["protocol_id"] == lhs_protocol]
            rhs = variant_df[variant_df["protocol_id"] == rhs_protocol]
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
                    "variant": variant_key,
                    "display_name": es.VARIANT_INDEX[variant_key]["display_name"],
                    "lhs_protocol": lhs_protocol,
                    "lhs_display_name": PROTOCOL_META[lhs_protocol]["display_name"],
                    "rhs_protocol": rhs_protocol,
                    "rhs_display_name": PROTOCOL_META[rhs_protocol]["display_name"],
                    "metric": metric,
                    "mean_delta": float(diffs.mean()),
                    "paired_cohens_d": effect_size,
                    "p_value": float(p_value) if pd.notna(p_value) else np.nan,
                    "seed_count": int(mask.sum()),
                })

    if not rows:
        return None
    path = os.path.join(paths["root"], "protocol_sensitivity_significance_tests.csv")
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _export_fairness_contract(paths: dict[str, str], suite_config: dict) -> str:
    payload = {
        "task_definition": "protocol_sensitivity_for_herb_target_prediction",
        "shared_training_entry": "train.main",
        "shared_threshold_policy": suite_config.get("threshold_policy", es.STRICT_THRESHOLD_POLICY),
        "shared_strict_split_config": dict(es.STRICT_SPLIT_CFG),
        "shared_variant_keys": suite_config.get("variants", DEFAULT_VARIANT_KEYS),
        "protocols": {
            protocol_id: {
                "display_name": meta["display_name"],
                "description": meta["description"],
            }
            for protocol_id, meta in PROTOCOL_META.items()
            if protocol_id in set(suite_config.get("protocols", DEFAULT_PROTOCOLS))
        },
        "fairness_requirements": [
            "Same training loop and optimizer policy across protocols.",
            "Same threshold freezing policy across protocols.",
            "Only the split builder changes across protocols.",
            "All rows must satisfy the same expected seed list before being marked strict-ready.",
        ],
    }
    path = os.path.join(paths["root"], "protocol_sensitivity_fairness_contract.json")
    _write_json(path, payload)
    return path


def _plot_metric(aggregated_df: pd.DataFrame, paths: dict[str, str], metric: str, ylabel: str) -> Optional[str]:
    if aggregated_df.empty or "strict_ready" not in aggregated_df.columns:
        return None
    subset = aggregated_df[aggregated_df["strict_ready"]].copy()
    if subset.empty or metric not in subset.columns:
        return None

    protocol_ids = [item for item in DEFAULT_PROTOCOLS if item in set(subset["protocol_id"].tolist())]
    variant_ids = [item for item in DEFAULT_VARIANT_KEYS if item in set(subset["variant"].tolist())]
    if not protocol_ids or not variant_ids:
        return None

    x = np.arange(len(protocol_ids))
    width = 0.22
    colors = {
        "hgt_baseline": "#2F7ED8",
        "static_similarity_hgt": "#F39C12",
        "proposed_model": "#1D9E75",
    }

    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    for idx, variant_key in enumerate(variant_ids):
        values = []
        errors = []
        label = es.VARIANT_INDEX[variant_key]["display_name"]
        for protocol_id in protocol_ids:
            row = subset[(subset["protocol_id"] == protocol_id) & (subset["variant"] == variant_key)]
            if row.empty:
                values.append(np.nan)
                errors.append(0.0)
            else:
                values.append(float(row.iloc[0].get(metric, np.nan)))
                errors.append(float(row.iloc[0].get(f"{metric}_std", 0.0) or 0.0))
        offset = (idx - (len(variant_ids) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            values,
            width=width,
            label=label,
            color=colors.get(variant_key, "#777777"),
            yerr=errors,
            capsize=3,
            alpha=0.95,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([PROTOCOL_META[item]["display_name"] for item in protocol_ids])
    ax.set_ylabel(ylabel)
    ax.set_title(f"Protocol Sensitivity Comparison ({metric})")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False, ncol=len(variant_ids))

    figure_path = os.path.join(paths["figures"], f"Figure_protocol_sensitivity_{metric.lower()}.png")
    fig.savefig(figure_path)
    plt.close(fig)
    return figure_path


def generate_report(paths: dict[str, str]) -> tuple[dict[str, str], pd.DataFrame]:
    suite_config = _load_json(_suite_config_path(paths)) or {}
    run_records = []
    expected_seeds = [int(item) for item in suite_config.get("expected_seeds", [42])]

    for protocol_id in suite_config.get("protocols", DEFAULT_PROTOCOLS):
        for variant_key in suite_config.get("variants", DEFAULT_VARIANT_KEYS):
            variant_root = _protocol_variant_root(paths, protocol_id, variant_key)
            if not os.path.exists(variant_root):
                continue
            for entry in sorted(os.listdir(variant_root)):
                seed_dir = os.path.join(variant_root, entry)
                if not entry.startswith("seed_") or not os.path.isdir(seed_dir):
                    continue
                payload = _load_json(os.path.join(seed_dir, "summary.json")) or _load_json(os.path.join(seed_dir, "error.json"))
                if payload is not None:
                    run_records.append(dict(payload))

    run_df = pd.DataFrame(run_records)
    aggregated_df = _aggregate(run_records, expected_seeds=expected_seeds)
    outputs = _export_tables(run_df, aggregated_df, paths)
    env_path = os.path.join(paths["root"], "protocol_sensitivity_environment.json")
    _write_json(env_path, _environment_snapshot())
    outputs["protocol_sensitivity_environment.json"] = env_path
    provenance_path = _export_code_provenance(paths)
    outputs["protocol_sensitivity_code_provenance.json"] = provenance_path
    fairness_path = _export_fairness_contract(paths, suite_config)
    outputs["protocol_sensitivity_fairness_contract.json"] = fairness_path

    integrity_path = _export_integrity_audit(run_df, aggregated_df, suite_config, paths)
    if integrity_path is not None:
        outputs["protocol_sensitivity_integrity_audit.csv"] = integrity_path

    summary_path = _export_report_summary(aggregated_df, suite_config, paths)
    outputs["protocol_sensitivity_report_summary.json"] = summary_path

    significance_path = _export_significance(run_df, paths)
    if significance_path is not None:
        outputs["protocol_sensitivity_significance_tests.csv"] = significance_path

    manifest = []
    for metric, ylabel in (("AUC", "AUC"), ("F1", "F1 Score")):
        figure_path = _plot_metric(aggregated_df, paths, metric=metric, ylabel=ylabel)
        if figure_path is not None:
            manifest.append({
                "figure_id": f"protocol_sensitivity_{metric.lower()}",
                "title": f"Protocol Sensitivity Comparison ({metric})",
                "path": figure_path,
            })

    manifest_path = os.path.join(paths["figures"], "protocol_sensitivity_figure_manifest.json")
    _write_json(manifest_path, manifest)
    outputs["protocol_sensitivity_figure_manifest.json"] = manifest_path
    return outputs, aggregated_df


def run_suite(
    output_root: str,
    protocols: list[str],
    variant_keys: list[str],
    seeds: list[int],
    min_seed_count: int = DEFAULT_MIN_TOP_JOURNAL_SEEDS,
) -> list[dict]:
    paths = ensure_dirs(output_root)
    write_suite_config(paths, seeds=seeds, protocols=protocols, variants=variant_keys, min_seed_count=min_seed_count)
    results = []

    for protocol_id in protocols:
        for variant_key in variant_keys:
            variant = es.VARIANT_INDEX[variant_key]
            print(f"\n{'=' * 78}")
            print(f"Protocol: {protocol_id} | Variant: {variant_key} | Seeds: {seeds}")
            print(f"{'=' * 78}")
            for current_seed in seeds:
                result = run_protocol_variant_seed(protocol_id, variant, paths, seed=current_seed)
                results.append(result)
                if result.get("status") == "ok":
                    print(
                        f"[Done] protocol={protocol_id} variant={variant_key} seed={current_seed} "
                        f"AUC={float(result.get('test_AUC', np.nan)):.4f} "
                        f"F1={float(result.get('test_F1', np.nan)):.4f}"
                    )
                else:
                    print(
                        f"[Failed] protocol={protocol_id} variant={variant_key} seed={current_seed}: "
                        f"{result.get('error', 'Unknown error')}"
                    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run protocol sensitivity experiments without changing the current model logic."
    )
    parser.add_argument(
        "--mode",
        choices=["run", "report", "all"],
        default="all",
        help="run: launch experiments, report: aggregate existing outputs, all: run then aggregate.",
    )
    parser.add_argument(
        "--protocols",
        nargs="*",
        default=list(DEFAULT_PROTOCOLS),
        help="Subset of protocols to run: random_edge node_split strict_herb_cold_start",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=list(DEFAULT_VARIANT_KEYS),
        help="Variant keys from experiment_suite.py, e.g. proposed_model hgt_baseline static_similarity_hgt",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Single seed. Use --seeds for multi-seed strict reporting.",
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
        help="Directory used to save protocol runs, tables, and figures.",
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
    invalid_protocols = [item for item in args.protocols if item not in PROTOCOL_META]
    if invalid_protocols:
        raise ValueError(f"Unsupported protocols: {invalid_protocols}")
    invalid_variants = [item for item in args.variants if item not in es.VARIANT_INDEX]
    if invalid_variants:
        raise ValueError(f"Unknown variants: {invalid_variants}")

    seeds = normalize_seeds(seeds=args.seeds, seed=args.seed)
    paths = ensure_dirs(args.output_root)

    if args.mode in ("run", "all"):
        run_suite(
            output_root=args.output_root,
            protocols=list(args.protocols),
            variant_keys=list(args.variants),
            seeds=seeds,
            min_seed_count=int(args.min_seed_count),
        )
    if args.mode in ("report", "all"):
        outputs, aggregated_df = generate_report(paths)
        print("\nSaved outputs:")
        for _, path in sorted(outputs.items()):
            print(f"  {path}")
        if args.require_complete:
            if aggregated_df.empty or "strict_ready" not in aggregated_df.columns or "completed_seed_count" not in aggregated_df.columns:
                raise SystemExit("Protocol sensitivity report is incomplete: no strict-ready rows available.")
            ready_mask = aggregated_df["strict_ready"] & (aggregated_df["completed_seed_count"] >= int(args.min_seed_count))
            if int(ready_mask.sum()) != int(len(aggregated_df)):
                raise SystemExit("Protocol sensitivity report failed the completeness gate.")


if __name__ == "__main__":
    main()
