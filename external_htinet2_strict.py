from __future__ import annotations

import argparse
import copy
from collections import defaultdict
import hashlib
import json
import math
import os
import platform
import random
import sys
import traceback
from dataclasses import dataclass
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
DEFAULT_OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "checkpoints", "external_htinet2_strict")
DEFAULT_VARIANT = "htinet2_strict"
DEFAULT_SEEDS = [42]
DEFAULT_CFG = {
    "data_path": os.path.join(SCRIPT_DIR, "processed", "hetero_graph.pt"),
    "epochs": 180,
    "batch_size": 512,
    "lr": 2e-4,
    "weight_decay": 1e-5,
    "hidden_dim": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "kge_lambda": 0.10,
    "kge_margin": 1.0,
    "kge_triples_per_step": 1024,
    "patience": 30,
    "min_epochs_before_es": 40,
    "device": "cuda",
    "deterministic": True,
}


def _write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _variant_root(paths: dict[str, str]) -> str:
    return os.path.join(paths["runs"], DEFAULT_VARIANT)


def _seed_dir(paths: dict[str, str], seed: int) -> str:
    return os.path.join(_variant_root(paths), f"seed_{int(seed)}")


def _summary_path(paths: dict[str, str], seed: Optional[int] = None) -> str:
    if seed is None:
        return os.path.join(_variant_root(paths), "summary.json")
    return os.path.join(_seed_dir(paths, seed), "summary.json")


def _error_path(paths: dict[str, str], seed: int) -> str:
    return os.path.join(_seed_dir(paths, seed), "error.json")


def _suite_config_path(paths: dict[str, str]) -> str:
    return os.path.join(paths["root"], "suite_config.json")


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
        payload["torch_version"] = getattr(torch, "__version__", "unknown")
        payload["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            payload["cuda_device_count"] = int(torch.cuda.device_count())
            payload["cuda_device_name"] = str(torch.cuda.get_device_name(0))
    except Exception as exc:
        payload["torch_version"] = f"unavailable: {exc}"
        payload["cuda_available"] = False
    return payload


def _export_code_provenance(paths: dict[str, str]) -> str:
    tracked_files = [
        "train.py",
        "model.py",
        "cold_start_split.py",
        "experiment_suite.py",
        "external_htinet2_strict.py",
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
    path = os.path.join(paths["root"], "htinet2_code_provenance.json")
    _write_json(path, rows)
    return path


def _export_fairness_contract(paths: dict[str, str], suite_config: dict) -> str:
    payload = {
        "external_model": "HTINet2",
        "reproduction_scope": "strict herb-level cold-start under the local HIT project data boundary",
        "shared_data_path": suite_config.get("data_path", DEFAULT_CFG["data_path"]),
        "shared_split_id": es.STRICT_SPLIT_ID,
        "shared_threshold_policy": es.STRICT_THRESHOLD_POLICY,
        "shared_split_config": dict(es.STRICT_SPLIT_CFG),
        "paper_mapping": {
            "knowledge_graph_embedding": "TransE-style typed KGE over the visible strict message graph",
            "residual_graph_representation": "Residual bipartite graph convolution over train herb-target pairs",
            "target_prediction": "Dot-product scoring with BPR supervision",
        },
        "notes": [
            "This implementation keeps the local project data boundary and does not import extra symptom/pathway resources.",
            "The strict split and threshold policy are aligned with the main project for fair comparison.",
            "Only the external baseline model differs; the evaluation protocol is shared.",
        ],
    }
    path = os.path.join(paths["root"], "htinet2_fairness_contract.json")
    _write_json(path, payload)
    return path


def normalize_seeds(seeds: Optional[list[int]] = None, seed: Optional[int] = None) -> list[int]:
    return es.normalize_seeds(seeds=seeds, seed=seed)


def write_suite_config(paths: dict[str, str], cfg: dict, seeds: list[int], min_seed_count: int) -> dict:
    payload = {
        "protocol_version": "htinet2_strict_v1",
        "variant": DEFAULT_VARIANT,
        "expected_seeds": [int(item) for item in seeds],
        "expected_seed_count": len(seeds),
        "min_top_journal_seed_count": int(min_seed_count),
        "split_id": es.STRICT_SPLIT_ID,
        "threshold_policy": es.STRICT_THRESHOLD_POLICY,
        "data_path": cfg["data_path"],
        "model_cfg": {key: cfg[key] for key in sorted(cfg.keys()) if key != "device"},
        "notes": [
            "External HTINet2 reproduction under the same strict split and threshold policy as the main project.",
            "This script is independent and does not modify the current project model logic.",
        ],
    }
    _write_json(_suite_config_path(paths), payload)
    return payload


@dataclass
class KgTriples:
    src_types: list[str]
    src_ids: torch.Tensor
    rel_ids: torch.Tensor
    dst_types: list[str]
    dst_ids: torch.Tensor
    rel_names: list[str]

    @property
    def count(self) -> int:
        return int(self.src_ids.numel())


class ResidualBipartiteLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.herb_self = nn.Linear(hidden_dim, hidden_dim)
        self.herb_msg = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.target_self = nn.Linear(hidden_dim, hidden_dim)
        self.target_msg = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.herb_norm = nn.LayerNorm(hidden_dim)
        self.target_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        herb_x: torch.Tensor,
        target_x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if edge_index.numel() == 0:
            return herb_x, target_x

        src = edge_index[0]
        dst = edge_index[1]
        herb_deg = torch.bincount(src, minlength=herb_x.size(0)).float().clamp_min_(1.0)
        target_deg = torch.bincount(dst, minlength=target_x.size(0)).float().clamp_min_(1.0)
        norm = torch.rsqrt(herb_deg[src] * target_deg[dst]).unsqueeze(-1)

        target_agg = torch.zeros_like(target_x)
        target_agg.index_add_(0, dst, herb_x[src] * norm)

        herb_agg = torch.zeros_like(herb_x)
        herb_agg.index_add_(0, src, target_x[dst] * norm)

        new_herb = self.herb_norm(herb_x + self.dropout(F.gelu(self.herb_self(herb_x) + self.herb_msg(herb_agg))))
        new_target = self.target_norm(target_x + self.dropout(F.gelu(self.target_self(target_x) + self.target_msg(target_agg))))
        return new_herb, new_target


class HTINet2StrictModel(nn.Module):
    def __init__(
        self,
        data,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        relation_names: Optional[list[str]] = None,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)

        self.node_types = [ntype for ntype in data.node_types if ntype in {"herb", "ingredient", "target"}]
        self.node_counts = {ntype: int(data[ntype].num_nodes) for ntype in self.node_types}
        self.feature_dims = {ntype: int(data[ntype].x.size(1)) for ntype in self.node_types}

        self.id_embeddings = nn.ModuleDict({
            ntype: nn.Embedding(self.node_counts[ntype], self.hidden_dim)
            for ntype in self.node_types
        })
        self.feature_projs = nn.ModuleDict({
            ntype: nn.Sequential(
                nn.Linear(self.feature_dims[ntype], self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
            )
            for ntype in self.node_types
        })

        rel_names = sorted(set(relation_names or []))
        self.relation_to_id = {name: idx for idx, name in enumerate(rel_names)}
        self.rel_embeddings = nn.Embedding(max(1, len(rel_names)), self.hidden_dim)

        self.layers = nn.ModuleList([
            ResidualBipartiteLayer(hidden_dim=self.hidden_dim, dropout=self.dropout)
            for _ in range(self.num_layers)
        ])
        self.herb_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.target_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )

        self.ht_edge_index = torch.empty((2, 0), dtype=torch.long)
        self._kg_triples: Optional[KgTriples] = None

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for emb in self.id_embeddings.values():
            nn.init.xavier_uniform_(emb.weight)
        for module in self.feature_projs.values():
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.rel_embeddings.weight)

    def set_ht_graph(self, train_pos_h: torch.Tensor, train_pos_t: torch.Tensor) -> None:
        if train_pos_h.numel() == 0:
            self.ht_edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            self.ht_edge_index = torch.stack([train_pos_h.cpu(), train_pos_t.cpu()], dim=0)

    def set_kg_triples(self, msg_graph) -> None:
        src_types = []
        src_ids = []
        rel_ids = []
        dst_types = []
        dst_ids = []
        rel_names = []

        for edge_type in msg_graph.edge_types:
            src_type, rel, dst_type = edge_type
            if src_type not in self.node_types or dst_type not in self.node_types:
                continue
            edge_index = msg_graph[edge_type].edge_index.cpu()
            rel_name = f"{src_type}:{rel}:{dst_type}"
            if rel_name not in self.relation_to_id:
                continue
            rel_id = self.relation_to_id[rel_name]
            for idx in range(edge_index.size(1)):
                src_types.append(src_type)
                src_ids.append(int(edge_index[0, idx].item()))
                rel_ids.append(rel_id)
                dst_types.append(dst_type)
                dst_ids.append(int(edge_index[1, idx].item()))
                rel_names.append(rel_name)

        if not src_ids:
            self._kg_triples = KgTriples([], torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), [], torch.empty(0, dtype=torch.long), [])
            return

        self._kg_triples = KgTriples(
            src_types=src_types,
            src_ids=torch.tensor(src_ids, dtype=torch.long),
            rel_ids=torch.tensor(rel_ids, dtype=torch.long),
            dst_types=dst_types,
            dst_ids=torch.tensor(dst_ids, dtype=torch.long),
            rel_names=rel_names,
        )

    def _base_embeddings(self, data, device: torch.device) -> dict[str, torch.Tensor]:
        embeds = {}
        for ntype in self.node_types:
            features = data[ntype].x.to(device)
            ids = torch.arange(self.node_counts[ntype], device=device)
            embeds[ntype] = self.feature_projs[ntype](features) + self.id_embeddings[ntype](ids)
        return embeds

    def encode(self, data) -> dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        base = self._base_embeddings(data, device=device)

        herb_x = base["herb"]
        target_x = base["target"]
        edge_index = self.ht_edge_index.to(device)

        for layer in self.layers:
            herb_x, target_x = layer(herb_x, target_x, edge_index)

        base["herb"] = self.herb_head(herb_x)
        base["target"] = self.target_head(target_x)
        return base

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
        x_dict = self.encode(data)
        herb_emb = x_dict["herb"][herb_ids]
        target_emb = x_dict["target"][target_ids]
        scores = (herb_emb * target_emb).sum(dim=-1)
        aux = {"herb_emb": herb_emb, "target_emb": target_emb}
        if return_attention and return_aux:
            return scores, None, aux
        if return_attention:
            return scores, None
        if return_aux:
            return scores, aux
        return scores

    def compute_kge_loss(self, batch_size: int, margin: float = 1.0) -> torch.Tensor:
        if self._kg_triples is None or self._kg_triples.count == 0:
            return next(self.parameters()).new_tensor(0.0)

        device = next(self.parameters()).device
        batch_size = min(int(batch_size), self._kg_triples.count)
        if batch_size <= 0:
            return next(self.parameters()).new_tensor(0.0)

        perm = torch.randint(0, self._kg_triples.count, (batch_size,), device=device)
        src_ids = self._kg_triples.src_ids[perm.cpu()].to(device)
        dst_ids = self._kg_triples.dst_ids[perm.cpu()].to(device)
        rel_ids = self._kg_triples.rel_ids[perm.cpu()].to(device)
        src_types = [self._kg_triples.src_types[int(idx)] for idx in perm.cpu().tolist()]
        dst_types = [self._kg_triples.dst_types[int(idx)] for idx in perm.cpu().tolist()]

        pos_src = torch.stack([
            self.id_embeddings[src_types[i]](src_ids[i])
            for i in range(batch_size)
        ], dim=0)
        pos_dst = torch.stack([
            self.id_embeddings[dst_types[i]](dst_ids[i])
            for i in range(batch_size)
        ], dim=0)
        rel_emb = self.rel_embeddings(rel_ids)

        neg_dst_ids = []
        for dst_type in dst_types:
            neg_dst_ids.append(random.randrange(self.node_counts[dst_type]))
        neg_dst_ids = torch.tensor(neg_dst_ids, dtype=torch.long, device=device)
        neg_dst = torch.stack([
            self.id_embeddings[dst_types[i]](neg_dst_ids[i])
            for i in range(batch_size)
        ], dim=0)

        pos_score = torch.linalg.norm(pos_src + rel_emb - pos_dst, ord=1, dim=-1)
        neg_score = torch.linalg.norm(pos_src + rel_emb - neg_dst, ord=1, dim=-1)
        return F.relu(float(margin) + pos_score - neg_score).mean()


def build_relation_names(msg_graph) -> list[str]:
    rel_names = []
    for src_type, rel, dst_type in msg_graph.edge_types:
        if src_type in {"herb", "ingredient", "target"} and dst_type in {"herb", "ingredient", "target"}:
            rel_names.append(f"{src_type}:{rel}:{dst_type}")
    return sorted(set(rel_names))


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(pos_scores - neg_scores).mean()


def build_neg_sampler(bundle, data, seed: int):
    it_etype = next(et for et in bundle.msg_graph.edge_types if et[0] == "ingredient" and et[2] == "target")
    train_it_edge = bundle.msg_graph[it_etype].edge_index
    train_ing_to_tgts = defaultdict(set)
    for idx in range(train_it_edge.size(1)):
        ing = int(train_it_edge[0, idx].item())
        tgt = int(train_it_edge[1, idx].item())
        train_ing_to_tgts[ing].add(tgt)

    train_ht_set = set(zip(bundle.train_pos_h.tolist(), bundle.train_pos_t.tolist()))
    return css.OnlineHTNegSampler(
        all_pos_global=train_ht_set,
        herb_to_ings=bundle.herb_to_ings,
        ing_to_tgts=dict(train_ing_to_tgts),
        num_tgts=int(data["target"].num_nodes),
        hard_ratio=0.0,
        pop_ratio=0.0,
        seed=int(seed),
    )


def train_one_epoch_htinet2(
    model: HTINet2StrictModel,
    msg_graph,
    train_pos_h: torch.Tensor,
    train_pos_t: torch.Tensor,
    neg_sampler,
    optimizer,
    batch_size: int,
    device: torch.device,
    kge_lambda: float,
    kge_margin: float,
    kge_triples_per_step: int,
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
        neg_h = neg_h[:batch_h.size(0)]
        neg_t = neg_t[:batch_h.size(0)]

        pos_scores = model(
            msg_graph,
            batch_h.to(device),
            batch_t.to(device),
            None,
            None,
        )
        neg_scores = model(
            msg_graph,
            neg_h.to(device),
            neg_t.to(device),
            None,
            None,
        )
        loss_bpr = bpr_loss(pos_scores, neg_scores)
        loss_kge = model.compute_kge_loss(batch_size=int(kge_triples_per_step), margin=float(kge_margin))
        loss = loss_bpr + float(kge_lambda) * loss_kge

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item()) * int(batch_h.size(0))

    return float(total_loss / total_count)


def _build_run_summary(
    cfg: dict,
    val_metrics: dict,
    frozen_val_metrics: dict,
    test_metrics: dict,
    bundle_stats: dict,
    best_epoch: int,
    best_threshold: float,
    save_dir: str,
) -> dict:
    summary = {
        "variant": DEFAULT_VARIANT,
        "experiment_name": DEFAULT_VARIANT,
        "group": "EXTERNAL",
        "title": "HTINet2 Strict Reproduction",
        "display_name": "HTINet2",
        "source": "external",
        "external_model": "HTINet2",
        "save_dir": save_dir,
        "best_epoch": int(best_epoch),
        "best_threshold": float(best_threshold),
        "seed": int(cfg["seed"]),
        "hidden_dim": int(cfg["hidden_dim"]),
        "num_layers": int(cfg["num_layers"]),
        "dropout": float(cfg["dropout"]),
        "kge_lambda": float(cfg["kge_lambda"]),
        "kge_margin": float(cfg["kge_margin"]),
        "threshold_policy": es.STRICT_THRESHOLD_POLICY,
        "split_id": es.STRICT_SPLIT_ID,
    }
    for prefix, metrics in (
        ("val", val_metrics),
        ("frozen_val", frozen_val_metrics),
        ("test", test_metrics),
    ):
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                summary[f"{prefix}_{key}"] = float(value)
    for key, value in (bundle_stats or {}).items():
        if isinstance(value, (int, float, bool)):
            summary[f"split_{key}"] = float(value) if isinstance(value, bool) else value
    return summary


def _curve_payload(labels: list[float], probs: list[float], metrics: dict) -> dict:
    labels_np = np.asarray(labels, dtype=np.float32)
    probs_np = np.asarray(probs, dtype=np.float32)
    fpr, tpr, _ = train.roc_curve(labels_np, probs_np)
    precision, recall, _ = train.precision_recall_curve(labels_np, probs_np)
    return {
        "labels": labels,
        "probs": probs,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "precision_curve": precision.tolist(),
        "recall_curve": recall.tolist(),
        "auc": float(metrics.get("AUC", np.nan)),
        "auprc": float(metrics.get("AUPRC", np.nan)),
    }


def run_seed(cfg: dict, paths: dict[str, str], seed: int) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg["seed"] = int(seed)
    run_dir = _seed_dir(paths, seed=int(seed))
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

        relation_names = build_relation_names(bundle.msg_graph)
        model = HTINet2StrictModel(
            data=bundle.msg_graph,
            hidden_dim=int(cfg["hidden_dim"]),
            num_layers=int(cfg["num_layers"]),
            dropout=float(cfg["dropout"]),
            relation_names=relation_names,
        ).to(device)
        model.set_ht_graph(bundle.train_pos_h, bundle.train_pos_t)
        model.set_kg_triples(bundle.msg_graph)

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
            train_loss = train_one_epoch_htinet2(
                model=model,
                msg_graph=bundle.msg_graph,
                train_pos_h=bundle.train_pos_h,
                train_pos_t=bundle.train_pos_t,
                neg_sampler=neg_sampler,
                optimizer=optimizer,
                batch_size=int(cfg["batch_size"]),
                device=device,
                kge_lambda=float(cfg["kge_lambda"]),
                kge_margin=float(cfg["kge_margin"]),
                kge_triples_per_step=int(cfg["kge_triples_per_step"]),
                perm_generator=perm_generator,
            )

            raw_val_metrics = train.evaluate(
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
                lagrange_lambdas=None,
                adaptive_controller=None,
            )
            best_temperature = train.fit_temperature(
                logits=np.asarray(raw_val_metrics.get("logits", []), dtype=np.float32),
                labels=np.asarray(raw_val_metrics.get("labels", []), dtype=np.float32),
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
                lagrange_lambdas=None,
                adaptive_controller=None,
            )

            rank_score = float(es.STRICT_METRIC_COLUMNS.index("AUC") >= 0)  # keep deterministic shape
            rank_score = 0.45 * float(val_metrics["AUC"]) + 0.55 * float(val_metrics["AUPRC"])

            if rank_score > best_rank:
                best_rank = rank_score
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                best_epoch = int(epoch)
                best_threshold = float(val_metrics.get("best_threshold", 0.5))
                best_val_metrics = dict(val_metrics)
                patience = 0
                torch.save({"model_state_dict": best_state, "cfg": cfg}, os.path.join(run_dir, "best_model.pt"))
            else:
                patience += 1

            print(
                f"[HTINet2] seed={seed} epoch={epoch:03d} "
                f"loss={train_loss:.4f} val_auc={float(val_metrics['AUC']):.4f} "
                f"val_auprc={float(val_metrics['AUPRC']):.4f} val_f1={float(val_metrics['F1']):.4f}"
            )

            if epoch >= int(cfg["min_epochs_before_es"]) and patience >= int(cfg["patience"]):
                break

        if best_state is None:
            raise RuntimeError("HTINet2 training did not produce a valid checkpoint.")

        model.load_state_dict(best_state)

        frozen_val_metrics = train.evaluate(
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
            lagrange_lambdas=None,
            adaptive_controller=None,
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
            lagrange_lambdas=None,
            adaptive_controller=None,
        )

        summary = _build_run_summary(
            cfg=cfg,
            val_metrics=best_val_metrics or {},
            frozen_val_metrics=frozen_val_metrics,
            test_metrics=test_metrics,
            bundle_stats=bundle.stats,
            best_epoch=best_epoch,
            best_threshold=best_threshold,
            save_dir=run_dir,
        )
        summary.update({
            "status": "ok",
            "order": 10,
            "strict_protocol_version": "htinet2_strict_v1",
        })

        curve_payload = _curve_payload(
            labels=test_metrics.get("labels", []),
            probs=test_metrics.get("probs", []),
            metrics=test_metrics,
        )
        _write_json(os.path.join(run_dir, "best_model_curves.json"), curve_payload)
        _write_json(_summary_path(paths, seed=int(seed)), summary)
        return summary
    except Exception as exc:
        payload = {
            "variant": DEFAULT_VARIANT,
            "display_name": "HTINet2",
            "source": "external",
            "status": "failed",
            "seed": int(seed),
            "split_id": es.STRICT_SPLIT_ID,
            "threshold_policy": es.STRICT_THRESHOLD_POLICY,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(_error_path(paths, seed=int(seed)), payload)
        return payload


def _collect_run_rows(paths: dict[str, str]) -> pd.DataFrame:
    variant_root = _variant_root(paths)
    rows = []
    if not os.path.exists(variant_root):
        return pd.DataFrame()
    for entry in sorted(os.listdir(variant_root)):
        seed_dir = os.path.join(variant_root, entry)
        if not entry.startswith("seed_") or not os.path.isdir(seed_dir):
            continue
        payload = _load_json(os.path.join(seed_dir, "summary.json")) or _load_json(os.path.join(seed_dir, "error.json"))
        if payload is None:
            continue
        rows.append(dict(payload))
    return pd.DataFrame(rows)


def _aggregate(run_df: pd.DataFrame, expected_seeds: list[int]) -> pd.DataFrame:
    if run_df.empty:
        return pd.DataFrame()
    frame = es._normalize_metric_columns(run_df.copy())
    ok_group = frame[frame["status"] == "ok"].copy()
    completed_seeds = sorted({
        int(item)
        for item in pd.to_numeric(ok_group.get("seed", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).tolist()
    })
    row = {
        "variant": DEFAULT_VARIANT,
        "display_name": "HTINet2",
        "source": "external",
        "status": "ok" if not ok_group.empty else "failed",
        "order": 10,
        "split_id": es.STRICT_SPLIT_ID,
        "threshold_policy": es.STRICT_THRESHOLD_POLICY,
        "expected_seeds": ",".join(str(item) for item in expected_seeds),
        "completed_seeds": ",".join(str(item) for item in completed_seeds),
        "expected_seed_count": len(expected_seeds),
        "completed_seed_count": len(completed_seeds),
        "strict_ready": completed_seeds == sorted(expected_seeds),
        "error_count": int((frame["status"] != "ok").sum()),
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


def _export_integrity_audit(run_df: pd.DataFrame, suite_config: dict, paths: dict[str, str]) -> str:
    expected_seeds = [int(item) for item in suite_config.get("expected_seeds", DEFAULT_SEEDS)]
    min_seed_count = int(suite_config.get("min_top_journal_seed_count", 5))
    ok_group = run_df[run_df["status"] == "ok"].copy() if (not run_df.empty and "status" in run_df.columns) else pd.DataFrame()
    completed_seeds = sorted({
        int(item)
        for item in pd.to_numeric(ok_group.get("seed", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).tolist()
    })
    audit_df = pd.DataFrame([{
        "variant": DEFAULT_VARIANT,
        "display_name": "HTINet2",
        "expected_seeds": ",".join(str(item) for item in expected_seeds),
        "completed_seeds": ",".join(str(item) for item in completed_seeds),
        "expected_seed_count": len(expected_seeds),
        "completed_seed_count": len(completed_seeds),
        "complete": completed_seeds == sorted(expected_seeds),
        "meets_min_seed_count": len(completed_seeds) >= min_seed_count,
        "top_journal_ready": (completed_seeds == sorted(expected_seeds)) and (len(completed_seeds) >= min_seed_count),
    }])
    path = os.path.join(paths["root"], "htinet2_integrity_audit.csv")
    audit_df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _export_significance(run_df: pd.DataFrame, paths: dict[str, str]) -> Optional[str]:
    subset = run_df[run_df["status"] == "ok"].copy() if (not run_df.empty and "status" in run_df.columns) else pd.DataFrame()
    if subset.empty:
        return None
    rows = []
    for metric in es.STRICT_METRIC_COLUMNS:
        metric_data = subset.get(metric)
        if metric_data is None or (isinstance(metric_data, pd.Series) and metric_data.empty):
            continue
        if not isinstance(metric_data, pd.Series):
            metric_data = pd.Series([metric_data])
        values = pd.to_numeric(metric_data, errors="coerce").dropna()
        if values.empty:
            continue
        mean_value = float(values.mean())
        std_value = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append({
            "metric": metric,
            "mean": mean_value,
            "std": std_value,
            "ci95": float(1.96 * std_value / np.sqrt(len(values))) if len(values) > 1 else 0.0,
            "seed_count": int(len(values)),
        })
    if not rows:
        return None
    path = os.path.join(paths["root"], "htinet2_seed_statistics.csv")
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _export_report_summary(aggregated_df: pd.DataFrame, suite_config: dict, paths: dict[str, str]) -> str:
    min_seed_count = int(suite_config.get("min_top_journal_seed_count", 5))
    payload = {
        "expected_rows": 1,
        "observed_rows": int(len(aggregated_df)),
        "strict_ready_rows": int(aggregated_df["strict_ready"].sum()) if (not aggregated_df.empty and "strict_ready" in aggregated_df.columns) else 0,
        "min_top_journal_seed_count": min_seed_count,
        "top_journal_ready_rows": int(((aggregated_df["completed_seed_count"] >= min_seed_count) & (aggregated_df["strict_ready"])).sum())
        if (not aggregated_df.empty and {"completed_seed_count", "strict_ready"}.issubset(aggregated_df.columns))
        else 0,
    }
    path = os.path.join(paths["root"], "htinet2_report_summary.json")
    _write_json(path, payload)
    return path


def _plot_metric(aggregated_df: pd.DataFrame, paths: dict[str, str], metric: str, ylabel: str) -> Optional[str]:
    if aggregated_df.empty or "strict_ready" not in aggregated_df.columns or metric not in aggregated_df.columns:
        return None
    subset = aggregated_df[aggregated_df["strict_ready"]].copy()
    if subset.empty:
        return None
    value = float(subset.iloc[0][metric])
    error = float(subset.iloc[0].get(f"{metric}_std", 0.0) or 0.0)

    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    ax.bar(["HTINet2"], [value], yerr=[error], capsize=3, color="#2F7ED8", alpha=0.95)
    ax.set_ylabel(ylabel)
    ax.set_title(f"HTINet2 Strict Result ({metric})")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    figure_path = os.path.join(paths["figures"], f"Figure_htinet2_strict_{metric.lower()}.png")
    fig.savefig(figure_path)
    plt.close(fig)
    return figure_path


def generate_report(paths: dict[str, str]) -> tuple[dict[str, str], pd.DataFrame]:
    suite_config = _load_json(_suite_config_path(paths)) or {}
    run_df = _collect_run_rows(paths)
    aggregated_df = _aggregate(run_df, expected_seeds=[int(item) for item in suite_config.get("expected_seeds", DEFAULT_SEEDS)])

    outputs = {}
    run_csv = os.path.join(paths["root"], "htinet2_runs.csv")
    agg_csv = os.path.join(paths["root"], "htinet2_results.csv")
    run_df.to_csv(run_csv, index=False, encoding="utf-8-sig")
    aggregated_df.to_csv(agg_csv, index=False, encoding="utf-8-sig")
    outputs["htinet2_runs.csv"] = run_csv
    outputs["htinet2_results.csv"] = agg_csv

    env_path = os.path.join(paths["root"], "htinet2_environment.json")
    _write_json(env_path, _environment_snapshot())
    outputs["htinet2_environment.json"] = env_path
    outputs["htinet2_code_provenance.json"] = _export_code_provenance(paths)
    outputs["htinet2_fairness_contract.json"] = _export_fairness_contract(paths, suite_config)
    outputs["htinet2_integrity_audit.csv"] = _export_integrity_audit(run_df, suite_config, paths)
    outputs["htinet2_report_summary.json"] = _export_report_summary(aggregated_df, suite_config, paths)

    stats_path = _export_significance(run_df, paths)
    if stats_path is not None:
        outputs["htinet2_seed_statistics.csv"] = stats_path

    manifest = []
    for metric, ylabel in (("AUC", "AUC"), ("F1", "F1 Score")):
        figure_path = _plot_metric(aggregated_df, paths, metric=metric, ylabel=ylabel)
        if figure_path is not None:
            manifest.append({
                "figure_id": f"htinet2_strict_{metric.lower()}",
                "title": f"HTINet2 Strict Result ({metric})",
                "path": figure_path,
            })
    manifest_path = os.path.join(paths["figures"], "htinet2_figure_manifest.json")
    _write_json(manifest_path, manifest)
    outputs["htinet2_figure_manifest.json"] = manifest_path
    return outputs, aggregated_df


def run_suite(paths: dict[str, str], cfg: dict, seeds: list[int], min_seed_count: int) -> list[dict]:
    write_suite_config(paths, cfg=cfg, seeds=seeds, min_seed_count=min_seed_count)
    results = []
    for seed in seeds:
        print(f"\n{'=' * 74}")
        print(f"Running HTINet2 strict reproduction | seed={seed}")
        print(f"{'=' * 74}")
        result = run_seed(cfg=cfg, paths=paths, seed=int(seed))
        results.append(result)
        if result.get("status") == "ok":
            print(
                f"[Done] seed={seed} "
                f"AUC={float(result.get('test_AUC', np.nan)):.4f} "
                f"F1={float(result.get('test_F1', np.nan)):.4f}"
            )
        else:
            print(f"[Failed] seed={seed}: {result.get('error', 'Unknown error')}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict external reproduction of HTINet2 under the local HIT project protocol."
    )
    parser.add_argument(
        "--mode",
        choices=["run", "report", "all"],
        default="all",
        help="run: launch training, report: aggregate existing outputs, all: run then aggregate.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_CFG["data_path"],
        help="Path to processed hetero_graph.pt",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_CFG["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CFG["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CFG["lr"])
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_CFG["weight_decay"])
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_CFG["hidden_dim"])
    parser.add_argument("--num-layers", type=int, default=DEFAULT_CFG["num_layers"])
    parser.add_argument("--dropout", type=float, default=DEFAULT_CFG["dropout"])
    parser.add_argument("--kge-lambda", type=float, default=DEFAULT_CFG["kge_lambda"])
    parser.add_argument("--kge-margin", type=float, default=DEFAULT_CFG["kge_margin"])
    parser.add_argument("--kge-triples-per-step", type=int, default=DEFAULT_CFG["kge_triples_per_step"])
    parser.add_argument("--patience", type=int, default=DEFAULT_CFG["patience"])
    parser.add_argument("--min-epochs-before-es", type=int, default=DEFAULT_CFG["min_epochs_before_es"])
    parser.add_argument("--device", type=str, default=DEFAULT_CFG["device"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--min-seed-count", type=int, default=1)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit with a non-zero status if the report does not satisfy the completeness gate.",
    )
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
        "kge_lambda": float(args.kge_lambda),
        "kge_margin": float(args.kge_margin),
        "kge_triples_per_step": int(args.kge_triples_per_step),
        "patience": int(args.patience),
        "min_epochs_before_es": int(args.min_epochs_before_es),
        "device": str(args.device),
    })
    return cfg


def main() -> None:
    args = parse_args()
    cfg = build_cfg(args)
    seeds = normalize_seeds(seeds=args.seeds, seed=args.seed)
    paths = ensure_dirs(args.output_root)

    if args.mode in ("run", "all"):
        run_suite(paths=paths, cfg=cfg, seeds=seeds, min_seed_count=int(args.min_seed_count))
    if args.mode in ("report", "all"):
        outputs, aggregated_df = generate_report(paths)
        print("\nSaved outputs:")
        for _, path in sorted(outputs.items()):
            print(f"  {path}")
        if args.require_complete:
            if aggregated_df.empty or "strict_ready" not in aggregated_df.columns or "completed_seed_count" not in aggregated_df.columns:
                raise SystemExit("HTINet2 report is incomplete: no strict-ready rows available.")
            ready_mask = aggregated_df["strict_ready"] & (aggregated_df["completed_seed_count"] >= int(args.min_seed_count))
            if int(ready_mask.sum()) != int(len(aggregated_df)):
                raise SystemExit("HTINet2 report failed the completeness gate.")


if __name__ == "__main__":
    main()

