from __future__ import annotations

"""
train.py
========
训练 / 验证 / 测试 管线
集成：强化学习虚拟边自适应 (VirtualEdgeRLAgent) + 课程学习负采样 + 动态阈值搜索

修改说明（论文图表生成）
-----------------------
1. evaluate() 新增 return_probs=False 参数：
     当 True 时，在结果字典中附带 'probs' 和 'labels' 列表，
     供生成 ROC/PR 曲线（图10、11）使用。

2. main() 的 epoch 循环中新增 RL 动态日志写入：
     每次 RL 动作执行后，向 checkpoints/rl_dynamics_log.csv 追写一行，
     包含 epoch、threshold、topk、val_auc、val_f1、hh_density、
     tt_density、rl_reward、epsilon 字段。
     供生成 RL 动态曲线（图4）使用。

3. main() 测试完成后，额外保存 best_model_curves.json：
     包含最佳模型在测试集上的 fpr/tpr/precision/recall/auc/auprc，
     供生成 ROC/PR 曲线（图10、11）使用。
"""

import os
import copy
import csv
import json
import time
import random
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch_geometric.data import HeteroData
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    roc_curve, precision_recall_curve,
)
from collections import defaultdict

from cold_start_split import (
    cold_start_split,
    build_herb_ing_padded,
    OnlineHTNegSampler,
    HTSplitBundle,
)
from model import IngredientAwareHTModel, VanillaHGTBaselineModel
from rl_agent import VirtualEdgeRLAgent
from processed import (
    augment_graph_with_similarity_features,
    compute_similarity_edges,
)
from config import (
    THRESHOLD_OPTIMIZATION_METRIC,
    THRESHOLD_SEARCH_SPACE,
    THRESHOLD_SEARCH_STEPS,
)
try:
    from paper_figure_exports import (
        save_test_predictions_csv,
        export_attention_case_csv,
    )
except Exception as _paper_export_error:
    save_test_predictions_csv = None
    export_attention_case_csv = None

# ================================================================
#  配置
# ================================================================

CFG = {
    # 数据
    'data_path':   'processed/hetero_graph.pt',
    'save_dir':    'checkpoints/',
    # 划分
    'val_ratio':      0.16,
    'test_ratio':     0.20,
    'neg_ratio':      1.0,
    'strict_eval_neg_filter': True,
    'it_mask_ratio':  0.20,
    'task_preset': 'strict',   # strict / balanced / progressive
    'progressive_switch_epoch': 120,
    'balanced_it_mask_ratio': 0.20,
    'balanced_strict_eval_neg_filter': False,
    'balanced_eval_hard_neg_ratio': 0.10,
    'balanced_eval_pop_neg_ratio': 0.10,
    'strict_eval_hard_neg_ratio': 0.10,
    'strict_eval_pop_neg_ratio': 0.10,
    'min_pos_per_herb_val': 1,
    'min_pos_per_herb_test': 1,
    'max_fallback_per_herb': 2,
    'seed':           42,
    'deterministic':  True,
    'paper_export_fig_data': True,
    'paper_export_top_cases': 3,
    'paper_export_top_ingredients': 12,
    # 模型
    'hidden_dim': 128,
    'num_layers': 3,
    'num_heads':  4,
    'dropout':    0.2,
    # 训练
    'epochs':        300,
    'batch_size':    512,
    'lr':            2e-4,
    'weight_decay':  5e-5,
    'pos_weight':    1.75,
    'neg_weight':    1.0,
    'train_neg_ratio_start': 1.0,
    'train_neg_ratio_end':   1.5,
    'train_neg_ratio_ramp':  140,
    'train_neg_ratio': 2.0,
    'patience':      30,
    'warmup_epochs': 20,
    # 课程学习：hard/pop neg 从 start 线性增长到 end
    'hard_neg_start': 0.05,
    'hard_neg_end':   0.20,
    'hard_neg_ramp':  180,
    'pop_neg_start':  0.05,
    'pop_neg_end':    0.15,
    'pop_neg_ramp':   180,
    # RL 虚拟边配置
    'use_rl_virtual_edge':    True,
    'rl_update_interval':     3,       # 每 N epoch 执行一次 RL 动作
    'rl_action_start_epoch':  15,      # 避开5~13轮平台期后再让RL改图
    'virtual_edge_threshold': 0.75,    # 初始相似度阈值
    'virtual_edge_topk':      5,       # 初始 Top-K
    'virtual_edge_types':     ['herb', 'ingredient', 'target'],  # 对哪些节点建虚拟边
    'virtual_edge_thresholds': {
        'herb': 0.75,
        'ingredient': 0.75,
        'target': 0.75,
    },
    'virtual_edge_topks': {
        'herb': 5,
        'ingredient': 5,
        'target': 5,
    },
    'virtual_edge_threshold': 0.58,
    'virtual_edge_topk':      6,
    'virtual_edge_similarity_schema': 'professional_v1',
    'virtual_edge_mutual_knn': True,
    'virtual_edge_thresholds': {
        'herb': 0.52,
        'ingredient': 0.60,
        'target': 0.58,
    },
    'virtual_edge_topks': {
        'herb': 6,
        'ingredient': 8,
        'target': 6,
    },
    'rl_batch_size':          32,
    'rl_buffer_warmup':       32,      # buffer 积累多少条后才开始 RL 更新
    'rl_initial_epsilon':     0.7,
    'rl_epsilon_decay':       0.999,
    'rl_save_path':           'checkpoints/rl_agent.pt',
    'resume_rl':              False,
    'rl_reward_w_composite':  0.70,
    'rl_reward_w_auc':        0.08,
    'rl_reward_w_auprc':      0.10,
    'rl_reward_w_f1':         0.10,
    'rl_reward_w_precision':  0.05,
    'rl_reward_w_recall':     0.20,
    'rl_reward_w_acc':        0.00,
    'rl_reward_w_pr_gap':     0.05,
    'rl_reward_w_density':    0.015,
    'rl_reward_clip':         0.20,
    'rl_rollback_tolerance':  0.010,
    'min_epochs_before_es':   50,
    'threshold_min_precision': 0.85,
    'constraint_target_precision': 0.88,
    'constraint_target_recall': 0.94,
    'label_smoothing': 0.02,
    'constraint_eta': 0.05,
    'constraint_max_lambda': 3.0,
    'adaptive_threshold_enabled': True,
    'adaptive_threshold_target_recall': 0.94,
    'adaptive_threshold_target_precision': 0.90,
    'adaptive_threshold_step': 0.03,
    'adaptive_threshold_max_delta': 0.05,
    'temperature_scaling_enabled': True,
    'temperature_fit_steps': 80,
    'temperature_fit_lr': 0.02,
    'freq_bias_beta': 0.08,
    'rl_constraint_penalty': 0.03,
    'rl_force_rollback_on_violation': False,
    # Early stopping 监控组合指标（仅用阈值无关指标，避免阈值波动影响模型选择）
    'es_auc_weight':  0.45,
    'es_prc_weight':  0.55,
    'device':         'cuda',
    'attention_activation': 'sigmoid',
    'attention_normalization': 'length_mean',
    'model_variant': 'proposed',
    'encoder_backbone': 'hgt',
    'use_spatial_encoder': True,
    'spatial_dim': 64,
    'spatial_dropout': 0.1,
    'spatial_max_atoms': 64,
    'spatial_dist_hidden_dim': 64,
    'spatial_centroid_hidden_dim': 16,
    'spatial_semantic_bias': 1.5,
    'use_ranking_loss': True,
    'ranking_lambda': 0.30,
    'use_contrastive_loss': True,
    'contrastive_lambda': 0.10,
    'contrastive_temperature': 0.15,
    'decoder_use_ingredient_path': True,
    'decoder_use_tri_attention': True,
    'decoder_use_global_path': True,
    'use_it_mask': True,
    'filter_hi_to_train_herbs': True,
    'ingredient_feature_mode': 'all',
    'curve_alias': 'proposed_model',
    'experiment_name': 'proposed_model',
    'experiment_group': 'CORE',
    'experiment_title': 'Proposed Model',
}


DEFAULT_CFG = copy.deepcopy(CFG)


def reset_cfg():
    CFG.clear()
    CFG.update(copy.deepcopy(DEFAULT_CFG))


def merge_cfg(overrides: Optional[dict] = None):
    if not overrides:
        return
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(CFG.get(key), dict):
            CFG[key] = {**CFG[key], **value}
        else:
            CFG[key] = value


def _safe_name(value: str) -> str:
    text = str(value).strip().replace(" ", "_")
    keep = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    cleaned = "".join(keep).strip("._")
    return cleaned or "experiment"


def _apply_ingredient_feature_mode(data: HeteroData, mode: str) -> HeteroData:
    mode = str(mode or "all").lower()
    if mode in ("all", "full", "default"):
        return data

    if "ingredient" not in data.node_types:
        return data

    store = data["ingredient"]
    modality_map = {
        "bert": getattr(store, "bert_x", None),
        "gpt": getattr(store, "gpt_x", None),
        "fp": getattr(store, "fp_x", None),
    }
    selected_map = {
        "all": ("bert", "gpt", "fp"),
        "no_bert": ("gpt", "fp"),
        "no_gpt": ("bert", "fp"),
        "no_fp": ("bert", "gpt"),
        "bert_only": ("bert",),
        "gpt_only": ("gpt",),
        "fp_only": ("fp",),
        "bert_gpt": ("bert", "gpt"),
        "bert_fp": ("bert", "fp"),
        "gpt_fp": ("gpt", "fp"),
    }
    if mode not in selected_map:
        raise ValueError(f"Unsupported ingredient_feature_mode: {mode}")

    selected_keys = selected_map[mode]
    selected_tensors = [
        modality_map[key].clone()
        for key in selected_keys
        if modality_map.get(key) is not None
    ]
    if not selected_tensors:
        raise RuntimeError(
            f"Ingredient feature mode '{mode}' requires bert_x/gpt_x/fp_x in processed data."
        )

    store.x = torch.cat(selected_tensors, dim=1).to(dtype=torch.float32)
    for key, tensor in modality_map.items():
        if tensor is None:
            continue
        patched = tensor.clone() if key in selected_keys else torch.zeros_like(tensor)
        setattr(store, f"{key}_x", patched)
    return data


def _build_run_summary(
    test_metrics: dict,
    val_metrics: dict,
    frozen_val_metrics: dict,
    bundle_stats: dict,
    ckpt_epoch: int,
    saved_threshold: float,
    save_dir: str,
) -> dict:
    summary = {
        "variant": str(CFG.get("curve_alias", CFG.get("experiment_name", "experiment"))),
        "experiment_name": str(CFG.get("experiment_name", "experiment")),
        "group": str(CFG.get("experiment_group", "")),
        "title": str(CFG.get("experiment_title", CFG.get("experiment_name", "Experiment"))),
        "save_dir": save_dir,
        "best_epoch": int(ckpt_epoch),
        "best_threshold": float(saved_threshold),
        "seed": int(CFG.get("seed", 42)),
        "ingredient_feature_mode": str(CFG.get("ingredient_feature_mode", "all")),
        "use_it_mask": bool(CFG.get("use_it_mask", True)),
        "filter_hi_to_train_herbs": bool(CFG.get("filter_hi_to_train_herbs", True)),
        "model_variant": str(CFG.get("model_variant", "proposed")),
        "encoder_backbone": str(CFG.get("encoder_backbone", "hgt")),
        "decoder_use_ingredient_path": bool(CFG.get("decoder_use_ingredient_path", True)),
        "decoder_use_tri_attention": bool(CFG.get("decoder_use_tri_attention", True)),
        "decoder_use_global_path": bool(CFG.get("decoder_use_global_path", True)),
        "use_rl_virtual_edge": bool(CFG.get("use_rl_virtual_edge", True)),
        "num_layers": int(CFG.get("num_layers", 0)),
        "use_spatial_encoder": bool(CFG.get("use_spatial_encoder", True)),
        "use_ranking_loss": bool(CFG.get("use_ranking_loss", True)),
        "use_contrastive_loss": bool(CFG.get("use_contrastive_loss", True)),
    }
    for prefix, metrics in (
        ("test", test_metrics),
        ("val", val_metrics),
        ("frozen_val", frozen_val_metrics),
    ):
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                summary[f"{prefix}_{key}"] = float(value)
    for key, value in (bundle_stats or {}).items():
        if isinstance(value, (int, float, bool)):
            summary[f"split_{key}"] = float(value) if isinstance(value, bool) else value
    return summary


def set_global_seed(seed: int, deterministic: bool = True):
    seed = int(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)


# ================================================================
#  数据加载
# ================================================================

def load_data(path: str) -> HeteroData:
    data = torch.load(path, weights_only=False)
    data = augment_graph_with_similarity_features(data)
    print(f"\n=== 数据加载: {path} ===")
    for ntype in data.node_types:
        print(f"  {ntype}: {data[ntype].num_nodes} nodes, "
              f"x.shape={data[ntype].x.shape}")
    for etype in data.edge_types:
        print(f"  {etype}: {data[etype].edge_index.shape[1]} edges")
    return data


def get_virtual_edge_types() -> list[str]:
    raw_types = CFG.get('virtual_edge_types', ['herb', 'ingredient', 'target'])
    seen = set()
    ordered = []
    for ntype in raw_types:
        if not isinstance(ntype, str):
            continue
        key = ntype.strip()
        if key and key not in seen:
            ordered.append(key)
            seen.add(key)
    return ordered


def build_virtual_edge_param_defaults() -> dict[str, dict[str, float | int]]:
    edge_types = get_virtual_edge_types()
    threshold_defaults = CFG.get('virtual_edge_thresholds', {}) or {}
    topk_defaults = CFG.get('virtual_edge_topks', {}) or {}
    fallback_threshold = float(CFG.get('virtual_edge_threshold', 0.75))
    fallback_topk = int(CFG.get('virtual_edge_topk', 5))

    params = {}
    for ntype in edge_types:
        params[ntype] = {
            'threshold': float(threshold_defaults.get(ntype, fallback_threshold)),
            'topk': int(topk_defaults.get(ntype, fallback_topk)),
        }
    return params


# ================================================================
#  课程学习：动态 hard_neg_ratio
# ================================================================

def get_hard_neg_ratio(epoch: int) -> float:
    start = CFG['hard_neg_start']
    end   = CFG['hard_neg_end']
    ramp  = CFG['hard_neg_ramp']
    return start + (end - start) * min(epoch / ramp, 1.0)


def get_train_neg_ratio(epoch: int) -> float:
    start = float(CFG.get('train_neg_ratio_start', CFG.get('train_neg_ratio', 1.0)))
    end = float(CFG.get('train_neg_ratio_end', CFG.get('train_neg_ratio', start)))
    ramp = max(1.0, float(CFG.get('train_neg_ratio_ramp', 1.0)))
    return start + (end - start) * min(epoch / ramp, 1.0)


def get_pop_neg_ratio(epoch: int) -> float:
    start = float(CFG.get('pop_neg_start', 0.0))
    end   = float(CFG.get('pop_neg_end', start))
    ramp  = max(1.0, float(CFG.get('pop_neg_ramp', 1.0)))
    return start + (end - start) * min(epoch / ramp, 1.0)


def get_dynamic_class_weights(train_neg_ratio: float, recall_gap: float = 0.0) -> tuple[float, float]:
    base_pos = float(CFG['pos_weight'])
    base_neg = float(CFG['neg_weight'])
    neg_scale = 1.0 / max(1.0, train_neg_ratio)
    dyn_neg = max(0.7, base_neg * neg_scale)

    # 如果召回率低于目标 (0.94)，则呈指数级增加正样本权重
    recall_boost = 1.0 + 2.0 * max(0.0, recall_gap)
    dyn_pos = base_pos * (1.0 + 0.15 * max(0.0, train_neg_ratio - 1.0)) * recall_boost
    return float(dyn_pos), float(dyn_neg)


def resolve_task_stage(epoch: int) -> tuple[str, dict]:
    """根据任务预设返回当前阶段与对应split参数。"""
    preset = str(CFG.get('task_preset', 'strict')).lower()
    switch_ep = int(CFG.get('progressive_switch_epoch', 120))

    if preset == 'balanced':
        stage = 'balanced'
    elif preset == 'progressive':
        stage = 'balanced' if epoch <= switch_ep else 'strict'
    else:
        stage = 'strict'

    if stage == 'balanced':
        split_cfg = {
            'it_mask_ratio': float(CFG.get('balanced_it_mask_ratio', CFG['it_mask_ratio'])),
            'strict_eval_neg_filter': bool(CFG.get('balanced_strict_eval_neg_filter', False)),
            'val_hard_neg_ratio': float(CFG.get('balanced_eval_hard_neg_ratio', 0.20)),
            'val_pop_neg_ratio': float(CFG.get('balanced_eval_pop_neg_ratio', 0.20)),
            'test_hard_neg_ratio': float(CFG.get('balanced_eval_hard_neg_ratio', 0.20)),
            'test_pop_neg_ratio': float(CFG.get('balanced_eval_pop_neg_ratio', 0.20)),
        }
    else:
        split_cfg = {
            'it_mask_ratio': float(CFG.get('it_mask_ratio', 0.05)),
            'strict_eval_neg_filter': bool(CFG.get('strict_eval_neg_filter', True)),
            'val_hard_neg_ratio': float(CFG.get('strict_eval_hard_neg_ratio', 0.30)),
            'val_pop_neg_ratio': float(CFG.get('strict_eval_pop_neg_ratio', 0.30)),
            'test_hard_neg_ratio': float(CFG.get('strict_eval_hard_neg_ratio', 0.30)),
            'test_pop_neg_ratio': float(CFG.get('strict_eval_pop_neg_ratio', 0.30)),
        }

    split_cfg['min_pos_per_herb_val'] = int(CFG.get('min_pos_per_herb_val', 1))
    split_cfg['min_pos_per_herb_test'] = int(CFG.get('min_pos_per_herb_test', 1))
    split_cfg['max_fallback_per_herb'] = int(CFG.get('max_fallback_per_herb', 2))
    return stage, split_cfg


# ================================================================
#  虚拟边管理器
# ================================================================

class VirtualEdgeManager:
    """
    管理虚拟边的动态添加与更新。

    RL 智能体决定 (Δthreshold, ΔtopK) →
    重新计算相似度虚拟边 →
    注入消息传递图。

    支持节点类型：herb-herb / ingredient-ingredient / target-target
    """

    def __init__(self, base_graph: HeteroData, data: HeteroData):
        self.base_graph = base_graph
        self.data = data
        self.edge_types = tuple(
            ntype for ntype in get_virtual_edge_types()
            if ntype in self.data.node_types
        )
        self.edge_params = build_virtual_edge_param_defaults()
        self.current_virtual_counts = {
            ntype: 0 for ntype in self.edge_types
        }

    @property
    def primary_edge_type(self) -> str:
        if 'herb' in self.edge_types:
            return 'herb'
        return self.edge_types[0] if self.edge_types else 'herb'

    def get_threshold(self, ntype: str) -> float:
        return float(
            self.edge_params.get(ntype, {}).get(
                'threshold', CFG.get('virtual_edge_threshold', 0.75)
            )
        )

    def get_topk(self, ntype: str) -> int:
        return int(
            self.edge_params.get(ntype, {}).get(
                'topk', CFG.get('virtual_edge_topk', 5)
            )
        )

    def get_threshold_norm(self, ntype: str) -> float:
        return float(np.clip((self.get_threshold(ntype) - 0.10) / (0.99 - 0.10), 0.0, 1.0))

    def get_topk_norm(self, ntype: str) -> float:
        return float(np.clip((self.get_topk(ntype) - 1.0) / (10.0 - 1.0), 0.0, 1.0))

    def get_rl_state_dim(self) -> int:
        return 4 + 3 * len(self.edge_types)

    def get_edge_params_snapshot(self) -> dict[str, dict[str, float | int]]:
        return {
            ntype: {
                'threshold': float(self.get_threshold(ntype)),
                'topk': int(self.get_topk(ntype)),
            }
            for ntype in self.edge_types
        }

    def load_edge_params(self, saved_params=None, legacy_threshold=None, legacy_topk=None):
        if isinstance(saved_params, dict) and saved_params:
            for ntype in self.edge_types:
                if ntype not in saved_params:
                    continue
                params = saved_params.get(ntype, {})
                self.edge_params[ntype] = {
                    'threshold': float(params.get('threshold', self.get_threshold(ntype))),
                    'topk': int(params.get('topk', self.get_topk(ntype))),
                }
            return

        for ntype in self.edge_types:
            self.edge_params[ntype] = {
                'threshold': float(
                    legacy_threshold if legacy_threshold is not None else self.get_threshold(ntype)
                ),
                'topk': int(
                    legacy_topk if legacy_topk is not None else self.get_topk(ntype)
                ),
            }

    def apply_action(self, action_deltas):
        """根据 RL 动作为每类虚拟边分别更新 threshold / topk。"""
        if isinstance(action_deltas, dict):
            deltas = action_deltas
        else:
            deltas = {
                ntype: action_deltas[idx]
                for idx, ntype in enumerate(self.edge_types)
            }

        for ntype in self.edge_types:
            delta_threshold, delta_topk = deltas.get(ntype, (0.0, 0))
            self.edge_params[ntype] = {
                'threshold': float(np.clip(
                    self.get_threshold(ntype) + float(delta_threshold), 0.10, 0.99
                )),
                'topk': int(np.clip(
                    self.get_topk(ntype) + int(delta_topk), 1, 10
                )),
            }

    def build_virtual_graph(self, device: torch.device) -> HeteroData:
        """
        在当前 threshold / topk 下重建带虚拟边的消息传递图。

        虚拟边类型：
            herb       --herb_similar_herb-->             herb
            ingredient --ingredient_similar_ingredient--> ingredient
            target     --target_similar_target-->         target
        """
        g = self.base_graph.clone()
        self.current_virtual_counts = {
            ntype: 0 for ntype in self.edge_types
        }

        for ntype in self.edge_types:
            node_store = self.data[ntype]
            feat = node_store.x
            if feat is None:
                continue

            edge_index, edge_weight = compute_similarity_edges(
                feat,
                threshold=self.get_threshold(ntype),
                top_k=self.get_topk(ntype),
                node_type=ntype,
                node_store=node_store,
                return_weights=True,
                mutual=bool(CFG.get('virtual_edge_mutual_knn', True)),
            )

            if edge_index is not None and edge_index.size(1) > 0:
                rel  = f'{ntype}_similar_{ntype}'
                etype = (ntype, rel, ntype)
                g[etype].edge_index = edge_index.to(device)
                g[etype].edge_weight = edge_weight.to(device)

                # 反向边
                rev_rel  = f'rev_{rel}'
                rev_etype = (ntype, rev_rel, ntype)
                g[rev_etype].edge_index = edge_index.flip(0).to(device)
                g[rev_etype].edge_weight = edge_weight.to(device)

                self.current_virtual_counts[ntype] = edge_index.size(1)
                print(f"   🔗 虚拟边 [{ntype}]: {edge_index.size(1)} 条 "
                      f"(thresh={self.get_threshold(ntype):.3f}, topk={self.get_topk(ntype)})")
            else:
                self.current_virtual_counts[ntype] = 0

        return g

    def get_edge_densities(self) -> dict[str, float]:
        """返回各类虚拟边密度，密度 = 边数 / 节点数。"""
        densities = {}
        for ntype in self.edge_types:
            num_nodes = self.data[ntype].num_nodes if ntype in self.data.node_types else 1
            densities[ntype] = float(
                self.current_virtual_counts.get(ntype, 0) / max(num_nodes, 1)
            )
        return densities


# ================================================================
#  RL 状态构建
# ================================================================

def build_rl_state(
    val_metrics:    dict,
    ve_manager:     VirtualEdgeManager,
) -> np.ndarray:
    """
    构建 RL 状态向量：
        [val_auc, val_f1, val_precision, val_recall,
         <per_type_density...>,
         <per_type_threshold_norm, per_type_topk_norm...>]
    """
    densities = ve_manager.get_edge_densities()
    state = [
        val_metrics.get('AUC',  0.0),
        val_metrics.get('F1',   0.0),
        val_metrics.get('Precision', 0.0),
        val_metrics.get('Recall', 0.0),
    ]
    for ntype in ve_manager.edge_types:
        state.append(float(densities.get(ntype, 0.0)))
    for ntype in ve_manager.edge_types:
        state.append(float(ve_manager.get_threshold_norm(ntype)))
        state.append(float(ve_manager.get_topk_norm(ntype)))
    return np.array(state, dtype=np.float32)


def attach_virtual_edge_metrics(metrics: dict, ve_manager: VirtualEdgeManager) -> dict:
    densities = ve_manager.get_edge_densities()
    for ntype in ve_manager.edge_types:
        metrics[f'{ntype}_density'] = float(densities.get(ntype, 0.0))

    metrics['hh_density'] = float(densities.get('herb', 0.0))
    metrics['ii_density'] = float(densities.get('ingredient', 0.0))
    metrics['tt_density'] = float(densities.get('target', 0.0))
    return metrics


def summarize_virtual_edge_action(
    ve_manager: VirtualEdgeManager,
    action_deltas: dict[str, tuple[float, int]],
) -> str:
    parts = []
    for ntype in ve_manager.edge_types:
        delta_threshold, delta_topk = action_deltas.get(ntype, (0.0, 0))
        parts.append(
            f"{ntype}:Δτ={float(delta_threshold):+0.3f},Δk={int(delta_topk):+d}"
            f"->τ={ve_manager.get_threshold(ntype):.3f},k={ve_manager.get_topk(ntype)}"
        )
    return " | ".join(parts)


class AdaptiveThresholdController:
    """轻量阈值自适应控制器：根据验证集 Precision/Recall 缺口微调阈值。"""

    def __init__(self, step: float, max_delta: float, search_space: tuple):
        self.step = float(step)
        self.max_delta = float(max_delta)
        lo, hi = float(search_space[0]), float(search_space[1])
        self.lo = min(lo, hi)
        self.hi = max(lo, hi)

    def adjust(self, threshold: float, precision: float, recall: float, target_precision: float, target_recall: float) -> tuple[float, float]:
        recall_gap = float(target_recall - recall)
        precision_gap = float(target_precision - precision)

        delta = self.step * recall_gap
        delta -= 0.45 * self.step * max(0.0, precision_gap)
        delta = float(np.clip(delta, -self.max_delta, self.max_delta))

        adjusted = float(np.clip(float(threshold) - delta, self.lo, self.hi))
        effective_delta = adjusted - float(threshold)
        return adjusted, float(effective_delta)


def update_constraint_lambdas(lambdas: dict, metrics: dict, eta: float, max_lambda: float) -> dict:
    """拉格朗日乘子在线更新：lambda <- max(0, lambda + eta*(target-metric))."""
    updated = dict(lambdas)
    for key, metric_key in (('precision', 'Precision'), ('recall', 'Recall')):
        target_key = f'target_{key}'
        lam_key = f'lambda_{key}'
        gap = float(updated.get(target_key, 0.0) - metrics.get(metric_key, 0.0))
        updated[lam_key] = float(np.clip(updated.get(lam_key, 0.0) + eta * gap, 0.0, max_lambda))
    return updated


def fit_temperature(logits: np.ndarray, labels: np.ndarray, steps: int, lr: float) -> float:
    """在验证集拟合单参数温度T（NLL最小化）。"""
    if logits.size == 0:
        return 1.0

    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.float32)
    log_t = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([log_t], lr=float(lr))

    for _ in range(max(1, int(steps))):
        temp = torch.exp(log_t).clamp_min(1e-4)
        loss = F.binary_cross_entropy_with_logits(logits_t / temp, labels_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    t = float(torch.exp(log_t.detach()).item())
    return float(np.clip(t, 0.2, 5.0))


# ================================================================
#  动态最优阈值搜索
# ================================================================

def compute_threshold_objective(
    labels:    np.ndarray,
    probs:     np.ndarray,
    threshold: float,
    metric:    str,
    min_precision: float = None,
    beta: float = 1.5,
    w_p: float = 0.25,
    w_r: float = 0.10,
    w_gap: float = 0.15,
    lagrange_lambdas: Optional[dict] = None,
) -> float:
    preds = (probs >= threshold).astype(int)
    if preds.sum() == 0 or preds.sum() == len(preds):
        return -1.0

    p  = precision_score(labels, preds, zero_division=0)
    r  = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    f_beta = ((1 + beta * beta) * p * r / max(beta * beta * p + r, 1e-12)) if (p > 0 or r > 0) else 0.0

    tn = float(((preds == 0) & (labels == 0)).sum())
    tp = float(((preds == 1) & (labels == 1)).sum())
    acc = (tp + tn) / float(len(labels)) if len(labels) > 0 else 0.0

    if metric == 'f1':
        return float(f1)
    if metric == 'youden':
        return float(r + acc - 1.0)
    if metric == 'balanced':
        min_p = float(CFG.get('threshold_min_precision', 0.0)) if min_precision is None else float(min_precision)
        soft_penalty = 1.5 * max(0.0, min_p - p)
        score = (f_beta + w_p * p + w_r * r - w_gap * abs(p - r))

        if lagrange_lambdas is not None:
            lp = float(lagrange_lambdas.get('lambda_precision', 0.0))
            lr = float(lagrange_lambdas.get('lambda_recall', 0.0))
            tpv = float(lagrange_lambdas.get('target_precision', CFG.get('constraint_target_precision', min_p)))
            trv = float(lagrange_lambdas.get('target_recall', CFG.get('constraint_target_recall', 0.0)))
            constraint_penalty = (
                lp * max(0.0, tpv - p)
                + lr * max(0.0, trv - r)
            )
            score -= constraint_penalty

        return float(score - soft_penalty)
    if metric == 'recall':
        return float(r if p >= 0.6 else 0.0)
    if metric == 'precision':
        return float(p if r >= 0.6 else 0.0)

    raise ValueError(f"不支持的 metric: {metric}")


def find_best_threshold(
    labels:       np.ndarray,
    probs:        np.ndarray,
    metric:       str = THRESHOLD_OPTIMIZATION_METRIC,
    search_space: tuple = None,
    search_step:  float = None,
    lagrange_lambdas: Optional[dict] = None,
) -> tuple:
    """
    两阶段阈值搜索：先全局粗搜，再在最优附近细搜。
    """
    if search_space is None:
        search_space = tuple(THRESHOLD_SEARCH_SPACE)
    if search_step is None:
        search_step = float(THRESHOLD_SEARCH_STEPS)

    lo, hi = float(search_space[0]), float(search_space[1])
    lo = max(0.0, min(lo, 1.0))
    hi = max(0.0, min(hi, 1.0))
    if lo > hi:
        lo, hi = hi, lo

    coarse_step = max(float(search_step), 1e-6)
    coarse_thresholds = np.arange(lo, hi + coarse_step * 0.5, coarse_step, dtype=np.float64)
    if coarse_thresholds.size == 0:
        coarse_thresholds = np.array([0.5], dtype=np.float64)

    best_thresh = 0.5
    best_score = -1.0

    for thresh in coarse_thresholds:
        score = compute_threshold_objective(
            labels,
            probs,
            float(thresh),
            metric,
            lagrange_lambdas=lagrange_lambdas,
        )
        if score > best_score:
            best_score = float(score)
            best_thresh = float(thresh)

    # 局部细搜：在粗搜最优点附近用更小步长搜索，缓解边界吸附
    fine_half_window = max(coarse_step, 0.02)
    fine_lo = max(lo, best_thresh - fine_half_window)
    fine_hi = min(hi, best_thresh + fine_half_window)
    fine_step = max(coarse_step / 10.0, 1e-4)
    fine_thresholds = np.arange(fine_lo, fine_hi + fine_step * 0.5, fine_step, dtype=np.float64)

    for thresh in fine_thresholds:
        score = compute_threshold_objective(
            labels,
            probs,
            float(thresh),
            metric,
            lagrange_lambdas=lagrange_lambdas,
        )
        if score > best_score:
            best_score = float(score)
            best_thresh = float(thresh)

    return best_thresh, best_score


def compute_ranking_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """带有动态边距的 BPR 损失，强制正样本显著高于负样本。"""
    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        return pos_scores.new_tensor(0.0)
    num_pairs = min(pos_scores.numel(), neg_scores.numel())
    # 增加一个硬 margin，使得不仅仅是 pos > neg，而是 pos > neg + margin
    diff = pos_scores[:num_pairs] - neg_scores[:num_pairs] - margin
    return F.softplus(-diff).mean()


def compute_contrastive_loss(model, aux: dict, batch_size: int, temperature: float = 0.15) -> torch.Tensor:
    """Herb-conditioned InfoNCE：anchor=[herb_emb, ingredient_context], 正样本target靠近，负样本target远离。"""
    if aux is None or batch_size <= 0:
        device = next(model.parameters()).device
        return torch.tensor(0.0, device=device)

    herb_emb = aux['herb_emb']
    target_emb = aux['target_emb']
    ingredient_context = aux['ingredient_context']
    total = herb_emb.size(0)
    if total < batch_size * 2:
        return herb_emb.new_tensor(0.0)

    pos_herb = herb_emb[:batch_size]
    pos_ctx = ingredient_context[:batch_size]
    pos_tgt = target_emb[:batch_size]
    neg_tgt = target_emb[batch_size:batch_size * 2]

    anchor = model.herb_context_projector(torch.cat([pos_herb, pos_ctx], dim=-1))
    pos_key = model.target_projector(pos_tgt)
    neg_key = model.target_projector(neg_tgt)

    anchor = F.normalize(anchor, dim=-1)
    pos_key = F.normalize(pos_key, dim=-1)
    neg_key = F.normalize(neg_key, dim=-1)

    pos_logits = (anchor * pos_key).sum(dim=-1, keepdim=True) / temperature
    neg_logits = (anchor * neg_key).sum(dim=-1, keepdim=True) / temperature
    logits = torch.cat([pos_logits, neg_logits], dim=1)
    labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


# ================================================================
#  训练一个 Epoch
# ================================================================

def train_one_epoch(
    model,
    msg_graph,          # 当前消息传递图（可能含虚拟边）
    train_pos_h,
    train_pos_t,
    herb_to_ings,
    neg_sampler,
    optimizer,
    batch_size,
    device,
    pos_weight:      float = 1.0,
    neg_weight:      float = 1.0,
    train_neg_ratio: float = 1.0,
    hard_neg_ratio:  float = 0.3,
    perm_generator:  torch.Generator = None,
) -> float:
    model.train()

    # 动态更新 neg_sampler 的 hard_ratio（课程学习）
    neg_sampler.hard_ratio = hard_neg_ratio

    total_loss = 0.0
    N    = train_pos_h.size(0)
    if perm_generator is not None:
        perm = torch.randperm(N, generator=perm_generator)
    else:
        perm = torch.randperm(N)
    train_h = train_pos_h[perm]
    train_t = train_pos_t[perm]

    for start in range(0, N, batch_size):
        end     = min(start + batch_size, N)
        batch_h = train_h[start:end]
        batch_t = train_t[start:end]
        B       = batch_h.size(0)

        neg_h, neg_t = neg_sampler.sample(batch_h, batch_t, neg_ratio=float(train_neg_ratio))

        all_h  = torch.cat([batch_h, neg_h])
        all_t  = torch.cat([batch_t, neg_t])
        labels = torch.cat([
            torch.ones(B),
            torch.zeros(neg_h.size(0)),
        ]).to(device)

        padded, mask = build_herb_ing_padded(all_h, herb_to_ings)
        out = model(
            msg_graph,
            all_h.to(device),
            all_t.to(device),
            padded.to(device),
            mask.to(device),
            return_aux=True,
        )
        scores, aux = out

        sample_weights = torch.where(
            labels > 0.5,
            torch.full_like(labels, float(pos_weight)),
            torch.full_like(labels, float(neg_weight)),
        )
        smooth = float(CFG.get('label_smoothing', 0.0))
        if smooth > 0.0:
            labels_smooth = labels * (1.0 - smooth) + 0.5 * smooth
        else:
            labels_smooth = labels
        loss_bce = F.binary_cross_entropy_with_logits(
            scores, labels_smooth, weight=sample_weights
        )

        pos_scores = scores[:B]
        neg_scores = scores[B:B + B]
        loss_rank = compute_ranking_loss(pos_scores, neg_scores, margin=1.2) if CFG.get('use_ranking_loss', True) else scores.new_tensor(0.0)
        loss_con = compute_contrastive_loss(
            model,
            aux,
            batch_size=B,
            temperature=float(CFG.get('contrastive_temperature', 0.15)),
        ) if CFG.get('use_contrastive_loss', True) else scores.new_tensor(0.0)

        loss = (
            loss_bce
            + float(CFG.get('ranking_lambda', 0.30)) * loss_rank
            + float(CFG.get('contrastive_lambda', 0.10)) * loss_con
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * B

    return total_loss / N


# ================================================================
#  评估
# ================================================================

@torch.no_grad()
def evaluate(
    model,
    msg_graph,
    pos_h, pos_t,
    neg_h, neg_t,
    herb_to_ings,
    batch_size,
    device,
    threshold:         float = None,
    thresh_metric:     str   = THRESHOLD_OPTIMIZATION_METRIC,
    threshold_search_space: tuple = None,
    threshold_search_step:  float = None,
    return_threshold:  bool  = False,
    return_probs:      bool  = False,
    return_logits:     bool  = False,
    temperature:       float = 1.0,
    lagrange_lambdas:  Optional[dict] = None,
    adaptive_controller: Optional[AdaptiveThresholdController] = None,
    adaptive_targets:  Optional[dict] = None,
) -> dict:
    """评估模型性能（支持温度缩放、约束阈值搜索、自适应阈值校正）。"""
    model.eval()

    def _score(h_ids, t_ids):
        scores = []
        for s in range(0, h_ids.size(0), batch_size):
            e  = min(s + batch_size, h_ids.size(0))
            bh = h_ids[s:e]
            bt = t_ids[s:e]
            padded, mask = build_herb_ing_padded(bh, herb_to_ings)
            sc = model(msg_graph, bh.to(device), bt.to(device),
                       padded.to(device), mask.to(device))
            scores.append(sc.cpu())
        return torch.cat(scores)

    pos_s = _score(pos_h, pos_t)
    neg_s = _score(neg_h, neg_t)

    logits_all = torch.cat([pos_s, neg_s]).numpy()
    labels = np.concatenate([
        np.ones(pos_s.size(0)),
        np.zeros(neg_s.size(0)),
    ])

    safe_temp = float(np.clip(temperature, 1e-3, 10.0))
    probs = 1.0 / (1.0 + np.exp(-(logits_all / safe_temp)))

    if threshold is None:
        best_thresh, best_thresh_score = find_best_threshold(
            labels,
            probs,
            metric=thresh_metric,
            search_space=threshold_search_space,
            search_step=threshold_search_step,
            lagrange_lambdas=lagrange_lambdas,
        )
        threshold_source = 'val_search'

        if not np.isfinite(best_thresh_score) or best_thresh_score < 0:
            best_thresh = 0.5
            best_thresh_score = compute_threshold_objective(
                labels,
                probs,
                threshold=best_thresh,
                metric=thresh_metric,
                lagrange_lambdas=lagrange_lambdas,
            )
            threshold_source = 'fallback_0.5'
    else:
        best_thresh = float(threshold)
        best_thresh_score = compute_threshold_objective(
            labels,
            probs,
            threshold=best_thresh,
            metric=thresh_metric,
            lagrange_lambdas=lagrange_lambdas,
        )
        threshold_source = 'fixed_input'

    preds = (probs >= best_thresh).astype(int)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    tn = float(((preds == 0) & (labels == 0)).sum())
    tp = float(((preds == 1) & (labels == 1)).sum())
    acc = (tp + tn) / float(len(labels)) if len(labels) > 0 else 0.0

    adaptive_delta = 0.0
    if adaptive_controller is not None and adaptive_targets is not None and threshold is None:
        target_p = float(adaptive_targets.get('target_precision', CFG.get('adaptive_threshold_target_precision', 0.82)))
        target_r = float(adaptive_targets.get('target_recall', CFG.get('adaptive_threshold_target_recall', 0.88)))
        adjusted_thresh, adaptive_delta = adaptive_controller.adjust(
            threshold=best_thresh,
            precision=precision,
            recall=recall,
            target_precision=target_p,
            target_recall=target_r,
        )
        if abs(adaptive_delta) > 1e-12:
            best_thresh = adjusted_thresh
            preds = (probs >= best_thresh).astype(int)
            precision = precision_score(labels, preds, zero_division=0)
            recall = recall_score(labels, preds, zero_division=0)
            tn = float(((preds == 0) & (labels == 0)).sum())
            tp = float(((preds == 1) & (labels == 1)).sum())
            acc = (tp + tn) / float(len(labels)) if len(labels) > 0 else 0.0
            best_thresh_score = compute_threshold_objective(
                labels,
                probs,
                threshold=best_thresh,
                metric=thresh_metric,
                lagrange_lambdas=lagrange_lambdas,
            )
            threshold_source = 'val_search+adaptive'

    pr_gap = abs(precision - recall)

    if lagrange_lambdas is None:
        target_precision = float(CFG.get('constraint_target_precision', 0.0))
        target_recall = float(CFG.get('constraint_target_recall', 0.0))
    else:
        target_precision = float(lagrange_lambdas.get('target_precision', CFG.get('constraint_target_precision', 0.0)))
        target_recall = float(lagrange_lambdas.get('target_recall', CFG.get('constraint_target_recall', 0.0)))

    constraint_violation = (
        max(0.0, target_precision - precision)
        + max(0.0, target_recall - recall)
    )

    result = {
        'AUC':       roc_auc_score(labels, probs),
        'AUPRC':     average_precision_score(labels, probs),
        'F1':        f1_score(labels, preds, zero_division=0),
        'Precision': precision,
        'Recall':    recall,
        'ACC':       acc,
        'PR_gap':    pr_gap,
        'pos_mean':  float(pos_s.mean()),
        'neg_mean':  float(neg_s.mean()),
        'temperature': float(safe_temp),
        'adaptive_threshold_delta': float(adaptive_delta),
        'constraint_violation': float(constraint_violation),
    }
    if return_threshold:
        search_space = threshold_search_space if threshold_search_space is not None else tuple(THRESHOLD_SEARCH_SPACE)
        search_step = threshold_search_step if threshold_search_step is not None else float(THRESHOLD_SEARCH_STEPS)
        result['best_threshold'] = float(best_thresh)
        result['best_threshold_score'] = float(best_thresh_score)
        result['threshold_metric'] = thresh_metric
        result['threshold_search_space'] = [float(search_space[0]), float(search_space[1])]
        result['threshold_search_step'] = float(search_step)
        result['threshold_source'] = threshold_source

    if return_probs:
        result['probs']  = probs.tolist()
        result['labels'] = labels.tolist()
    if return_logits:
        result['logits'] = logits_all.tolist()

    return result


# ================================================================
#  RL 奖励计算
# ================================================================

def compute_rl_reward(
    prev_metrics: dict,
    curr_metrics: dict,
) -> float:
    """平衡导向RL奖励：优先使用阈值无关排序能力增量，弱化阈值搜索噪声。"""
    prev_rank = CFG['es_auc_weight'] * prev_metrics.get('AUC', 0.0) + CFG['es_prc_weight'] * prev_metrics.get('AUPRC', 0.0)
    curr_rank = CFG['es_auc_weight'] * curr_metrics.get('AUC', 0.0) + CFG['es_prc_weight'] * curr_metrics.get('AUPRC', 0.0)
    delta_composite = curr_rank - prev_rank

    delta_auc = curr_metrics.get('AUC', 0.0) - prev_metrics.get('AUC', 0.0)
    delta_auprc = curr_metrics.get('AUPRC', 0.0) - prev_metrics.get('AUPRC', 0.0)
    delta_f1  = curr_metrics.get('F1', 0.0) - prev_metrics.get('F1', 0.0)
    delta_precision = curr_metrics.get('Precision', 0.0) - prev_metrics.get('Precision', 0.0)
    delta_recall = curr_metrics.get('Recall', 0.0) - prev_metrics.get('Recall', 0.0)

    pr_gap = curr_metrics.get('PR_gap', 0.0)
    density_penalty = 0.0
    for ntype in get_virtual_edge_types():
        density_penalty += curr_metrics.get(f'{ntype}_density', 0.0)
    constraint_violation = curr_metrics.get('constraint_violation', 0.0)

    reward = (
        CFG.get('rl_reward_w_composite', 0.0) * delta_composite
        + CFG['rl_reward_w_auc'] * delta_auc
        + CFG['rl_reward_w_auprc'] * delta_auprc
        + CFG['rl_reward_w_f1'] * delta_f1
        + CFG['rl_reward_w_precision'] * delta_precision
        + CFG['rl_reward_w_recall'] * delta_recall
        - CFG['rl_reward_w_pr_gap'] * pr_gap
        - CFG['rl_reward_w_density'] * density_penalty
        - CFG.get('rl_constraint_penalty', 0.0) * constraint_violation
    )

    reward = float(np.clip(reward, -CFG['rl_reward_clip'], CFG['rl_reward_clip']))
    return reward


# ================================================================
#  辅助：保存 ROC / PR 曲线数据
# ================================================================

def _save_curves_json(
    probs:        np.ndarray,
    labels:       np.ndarray,
    auc:          float,
    auprc:        float,
    save_path:    str,
    sample_step:  int = 5,
):
    """
    将 ROC 和 PR 曲线数据保存为 JSON 文件，供图10、11绘图脚本读取。

    参数
    ----
    sample_step : int
        对曲线数据点降采样（每 sample_step 个点保留一个），
        降低文件体积。对于 >10 000 个正样本通常取 5~10。
    """
    fpr, tpr, _ = roc_curve(labels, probs)
    pre, rec, _ = precision_recall_curve(labels, probs)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump({
            'fpr':       fpr[::sample_step].tolist(),
            'tpr':       tpr[::sample_step].tolist(),
            'precision': pre[::sample_step].tolist(),
            'recall':    rec[::sample_step].tolist(),
            'auc':       float(auc),
            'auprc':     float(auprc),
        }, f)
    print(f"  📈 曲线数据已保存: {save_path}")


# ================================================================
#  主流程
# ================================================================

def main():
    set_global_seed(CFG['seed'], deterministic=CFG['deterministic'])
    os.makedirs(CFG['save_dir'], exist_ok=True)
    device = torch.device(
        CFG['device'] if torch.cuda.is_available() else 'cpu'
    )
    print(f"Device: {device}")

    # ── 1. 加载数据 ──────────────────────────────────────────
    data = load_data(CFG['data_path'])
    data = _apply_ingredient_feature_mode(
        data,
        CFG.get('ingredient_feature_mode', 'all'),
    )

    # ── 2. 冷启动划分 ────────────────────────────────────────
    task_stage, split_cfg = resolve_task_stage(epoch=1)
    bundle: HTSplitBundle = cold_start_split(
        data,
        val_ratio      = CFG['val_ratio'],
        test_ratio     = CFG['test_ratio'],
        neg_ratio      = CFG['neg_ratio'],
        hard_neg_ratio = CFG['hard_neg_start'],
        pop_neg_ratio  = split_cfg['val_pop_neg_ratio'],
        it_mask_ratio  = split_cfg['it_mask_ratio'],
        seed           = CFG['seed'],
        strict_eval_neg_filter = split_cfg['strict_eval_neg_filter'],
        val_hard_neg_ratio = split_cfg['val_hard_neg_ratio'],
        val_pop_neg_ratio = split_cfg['val_pop_neg_ratio'],
        test_hard_neg_ratio = split_cfg['test_hard_neg_ratio'],
        test_pop_neg_ratio = split_cfg['test_pop_neg_ratio'],
        min_pos_per_herb_val = split_cfg['min_pos_per_herb_val'],
        min_pos_per_herb_test = split_cfg['min_pos_per_herb_test'],
        max_fallback_per_herb = split_cfg['max_fallback_per_herb'],
        use_it_mask = bool(CFG.get('use_it_mask', True)),
        filter_hi_to_train_herbs = bool(CFG.get('filter_hi_to_train_herbs', True)),
    )

    base_msg_graph = bundle.msg_graph.to(device)
    if hasattr(data['ingredient'], 'pos') and data['ingredient'].pos is not None:
        base_msg_graph['ingredient'].pos = data['ingredient'].pos.to(device)

    # ── 3. 重建 ing_to_tgts ─────────────────────────────────
    it_etype = next(
        et for et in base_msg_graph.edge_types
        if et[0] == 'ingredient' and et[2] == 'target'
    )
    train_it_edge     = base_msg_graph[it_etype].edge_index
    train_ing_to_tgts = defaultdict(set)
    for i in range(train_it_edge.size(1)):
        ing = train_it_edge[0, i].item()
        tgt = train_it_edge[1, i].item()
        train_ing_to_tgts[ing].add(tgt)

    train_ht_set = set(
        zip(bundle.train_pos_h.tolist(), bundle.train_pos_t.tolist())
    )

    num_ing = int(data['ingredient'].num_nodes)
    num_tgt = int(data['target'].num_nodes)
    ing_degree = torch.zeros(num_ing, dtype=torch.float32)
    tgt_degree = torch.zeros(num_tgt, dtype=torch.float32)
    for i in range(train_it_edge.size(1)):
        ing = int(train_it_edge[0, i].item())
        tgt = int(train_it_edge[1, i].item())
        if 0 <= ing < num_ing:
            ing_degree[ing] += 1.0
        if 0 <= tgt < num_tgt:
            tgt_degree[tgt] += 1.0

    ingredient_log_freq = torch.log1p(ing_degree)
    target_log_freq = torch.log1p(tgt_degree)

    neg_sampler = OnlineHTNegSampler(
        all_pos_global = train_ht_set,
        herb_to_ings   = bundle.herb_to_ings,
        ing_to_tgts    = dict(train_ing_to_tgts),
        num_tgts       = data['target'].num_nodes,
        hard_ratio     = CFG['hard_neg_start'],
        pop_ratio      = CFG.get('pop_neg_start', 0.0),
        target_popularity = (target_log_freq.cpu().numpy() + 1e-6),
        seed           = CFG['seed'],
    )

    # ── 4. 初始化虚拟边管理器 ────────────────────────────────
    ve_manager = VirtualEdgeManager(
        base_graph = base_msg_graph,
        data       = data,
    )

    # 初始建图（用初始 threshold/topk）
    if CFG['use_rl_virtual_edge'] and len(ve_manager.edge_types) > 0:
        print("\n🔗 [VirtualEdge] 初始化虚拟边...")
        msg_graph = ve_manager.build_virtual_graph(device)
    else:
        msg_graph = base_msg_graph
    if (not CFG['use_rl_virtual_edge']) and len(ve_manager.edge_types) > 0:
        print("\n[VirtualEdge] using fixed virtual edges without RL updates...")
        msg_graph = ve_manager.build_virtual_graph(device)

    # ── 5. 构建模型 ──────────────────────────────────────────
    # 模型的 edge_types / node_types 需包含虚拟边
    model_variant = str(CFG.get('model_variant', 'proposed')).lower()
    encoder_backbone = str(CFG.get('encoder_backbone', 'hgt')).lower()
    if model_variant == 'vanilla_hgt':
        model = VanillaHGTBaselineModel(
            node_types=msg_graph.node_types,
            edge_types=msg_graph.edge_types,
            in_dim_dict={nt: msg_graph[nt].x.size(1) for nt in msg_graph.node_types},
            hidden_dim=CFG['hidden_dim'],
            num_layers=CFG['num_layers'],
            num_heads=CFG['num_heads'],
            dropout=CFG['dropout'],
        ).to(device)
    else:
        model = IngredientAwareHTModel(
            node_types  = msg_graph.node_types,
            edge_types  = msg_graph.edge_types,
            in_dim_dict = {nt: msg_graph[nt].x.size(1) for nt in msg_graph.node_types},
            hidden_dim  = CFG['hidden_dim'],
            num_layers  = CFG['num_layers'],
            num_heads   = CFG['num_heads'],
            dropout     = CFG['dropout'],
            attention_activation = CFG.get('attention_activation', 'softmax'),
            attention_normalization = CFG.get('attention_normalization', 'length_mean'),
            freq_bias_beta = float(CFG.get('freq_bias_beta', 0.0)),
            use_spatial_encoder = bool(CFG.get('use_spatial_encoder', True)),
            spatial_dim = int(CFG.get('spatial_dim', 64)),
            spatial_dropout = float(CFG.get('spatial_dropout', 0.1)),
            spatial_max_atoms = int(CFG.get('spatial_max_atoms', 64)),
            spatial_dist_hidden_dim = int(CFG.get('spatial_dist_hidden_dim', 64)),
            spatial_centroid_hidden_dim = int(CFG.get('spatial_centroid_hidden_dim', 16)),
            spatial_semantic_bias = float(CFG.get('spatial_semantic_bias', 1.5)),
            encoder_backbone = encoder_backbone,
            decoder_use_ingredient_path = bool(CFG.get('decoder_use_ingredient_path', True)),
            decoder_use_tri_attention = bool(CFG.get('decoder_use_tri_attention', True)),
            decoder_use_global_path = bool(CFG.get('decoder_use_global_path', True)),
        ).to(device)

    if hasattr(model, 'decoder') and hasattr(model.decoder, 'set_frequency_bias'):
        model.decoder.set_frequency_bias(ingredient_log_freq, target_log_freq)

    print(f"\n模型参数量: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"   [Variant] model_variant={model_variant}  backbone={encoder_backbone}")
    if model_variant != 'vanilla_hgt':
        print(f"   [Spatial] enabled={CFG.get('use_spatial_encoder', True)}  "
              f"dim={CFG.get('spatial_dim', 64)}  "
              f"ingredient_semantic_dim={base_msg_graph['ingredient'].x.size(1)}")

    # ── 6. RL 智能体 ─────────────────────────────────────────
    rl_agent = None
    if CFG['use_rl_virtual_edge'] and len(ve_manager.edge_types) > 0:
        rl_agent = VirtualEdgeRLAgent(
            state_dim   = ve_manager.get_rl_state_dim(),
            action_dim  = 63,    # 9×7 (Δthreshold, ΔtopK)
            action_heads = len(ve_manager.edge_types),
            action_head_labels = ve_manager.edge_types,
            lr          = 1e-3,
            epsilon     = CFG['rl_initial_epsilon'],
            epsilon_decay = CFG['rl_epsilon_decay'],
            buffer_size = 20000,
            batch_size  = int(CFG.get('rl_batch_size', 32)),
            verbose     = True,
            seed        = CFG['seed'],
        )
        # 尝试加载已有 RL checkpoint
        if CFG.get('resume_rl', False) and os.path.exists(CFG['rl_save_path']):
            rl_agent.load_model(CFG['rl_save_path'])
            print(f"   ✅ RL智能体加载: {CFG['rl_save_path']}")

    # ── 7. 优化器 + Warmup + CosineAnnealing ─────────────────
    optimizer = AdamW(
        model.parameters(),
        lr           = CFG['lr'],
        weight_decay = CFG['weight_decay'],
    )

    def lr_lambda(epoch):
        warmup = CFG['warmup_epochs']
        if epoch < warmup:
            return float(epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, CFG['epochs'] - warmup)
        return max(1e-7 / CFG['lr'],
                   0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    train_perm_generator = torch.Generator(device='cpu')
    train_perm_generator.manual_seed(int(CFG['seed']))

    # ── 8. RL 动态日志初始化（图4）────────────────────────────
    rl_log_path = os.path.join(CFG['save_dir'], 'rl_dynamics_log.csv')
    rl_log_file = open(rl_log_path, 'w', newline='', encoding='utf-8')
    rl_log_fields = [
        'epoch', 'task_stage', 'threshold', 'topk',
        'val_auc', 'val_f1', 'hh_density', 'ii_density', 'tt_density',
        'rl_reward', 'epsilon', 'temperature',
        'adaptive_threshold_delta', 'constraint_violation',
        'lambda_precision', 'lambda_recall',
        'val_herb_coverage', 'val_fallback_used_ratio',
        'train_neg_ratio', 'hard_neg_ratio',
    ]
    for ntype in ve_manager.edge_types:
        rl_log_fields.extend([
            f'{ntype}_threshold',
            f'{ntype}_topk',
            f'{ntype}_density',
        ])
    rl_writer   = csv.DictWriter(rl_log_file, fieldnames=rl_log_fields)
    rl_writer.writeheader()
    rl_log_file.flush()

    # ── 9. 训练循环 ──────────────────────────────────────────
    def rank_selection_score(m):
        return CFG['es_auc_weight'] * m['AUC'] + CFG['es_prc_weight'] * m['AUPRC']

    def composite_score(m):
        # 仅用于观察阈值后表现，不再作为best checkpoint主标准
        rank_score = rank_selection_score(m)
        balanced_score = m.get('best_threshold_score', 0.0)
        return 0.8 * rank_score + 0.2 * balanced_score

    best_rank_score = 0.0
    patience_cnt = 0

    lagrange_lambdas = {
        'lambda_precision': 0.0,
        'lambda_recall': 0.0,
        'target_precision': float(CFG.get('constraint_target_precision', 0.85)),
        'target_recall': float(CFG.get('constraint_target_recall', 0.85)),
    }
    adaptive_controller = AdaptiveThresholdController(
        step=float(CFG.get('adaptive_threshold_step', 0.03)),
        max_delta=float(CFG.get('adaptive_threshold_max_delta', 0.05)),
        search_space=tuple(THRESHOLD_SEARCH_SPACE),
    ) if CFG.get('adaptive_threshold_enabled', True) else None
    adaptive_targets = {
        'target_precision': float(CFG.get('adaptive_threshold_target_precision', lagrange_lambdas['target_precision'])),
        'target_recall': float(CFG.get('adaptive_threshold_target_recall', lagrange_lambdas['target_recall'])),
    }
    calibrated_temperature = 1.0

    # RL 状态追踪
    prev_val_metrics = {
        'AUC': 0.0,
        'F1': 0.0,
        'Precision': 0.0,
        'Recall': 0.0,
        'PR_gap': 1.0,
        'hh_density': 0.0,
        'ii_density': 0.0,
        'tt_density': 0.0,
        'constraint_violation': 0.0,
    }

    print(f"\n{'='*92}")
    print(f"  开始训练  lr={CFG['lr']}  pos_weight={CFG['pos_weight']}  "
          f"dropout={CFG['dropout']}  warmup={CFG['warmup_epochs']}")
    print(f"  阈值配置: metric={THRESHOLD_OPTIMIZATION_METRIC}  "
          f"space={THRESHOLD_SEARCH_SPACE}  step={THRESHOLD_SEARCH_STEPS}")
    print(f"  复现配置: seed={CFG['seed']}  deterministic={CFG['deterministic']}  "
          f"resume_rl={CFG.get('resume_rl', False)}")
    print(f"  课程学习: hard_neg {CFG['hard_neg_start']} → "
          f"{CFG['hard_neg_end']} (ramp {CFG['hard_neg_ramp']} epochs)  "
          f"pop_neg {CFG.get('pop_neg_start', 0.0)} → {CFG.get('pop_neg_end', 0.0)} "
          f"(ramp {CFG.get('pop_neg_ramp', 1)} epochs)")
    print(f"  任务预设: preset={CFG.get('task_preset', 'strict')} "
          f"stage={task_stage} switch_epoch={CFG.get('progressive_switch_epoch', 120)}")
    if len(ve_manager.edge_types) > 0:
        ve_param_text = " | ".join(
            f"{ntype}:τ={ve_manager.get_threshold(ntype):.3f},k={ve_manager.get_topk(ntype)}"
            for ntype in ve_manager.edge_types
        )
        print(f"  RL虚拟边: {ve_param_text}  "
              f"更新间隔={CFG['rl_update_interval']} epochs  "
              f"启动epoch={CFG['rl_action_start_epoch']}  "
              f"eps_init={CFG['rl_initial_epsilon']}")
    print(f"  注意力机制: activation={CFG.get('attention_activation', 'softmax')}  "
          f"norm={CFG.get('attention_normalization', 'length_mean')}")
    print(f"  频次去偏置: beta={CFG.get('freq_bias_beta', 0.0)}")
    print(f"  约束目标: P>={lagrange_lambdas['target_precision']:.2f} "
          f"R>={lagrange_lambdas['target_recall']:.2f}")
    print(f"  自适应阈值: enabled={adaptive_controller is not None}  "
          f"step={CFG.get('adaptive_threshold_step', 0.03)}  "
          f"max_delta={CFG.get('adaptive_threshold_max_delta', 0.05)}")
    print(f"  温度缩放: enabled={CFG.get('temperature_scaling_enabled', True)}")
    print(f"{'='*92}")
    print(f"{'Ep':>5} {'Loss':>7} {'AUC':>7} {'PRC':>7} "
          f"{'F1':>7} {'Pre':>7} {'Rec':>7} {'ACC':>7} {'Gap':>7} "
          f"{'Thr':>6} {'Temp':>6} {'ΔThr':>6} {'Viol':>6} {'λr':>5} "
          f"{'Cov':>5} {'FB':>5} {'Stg':>6} {'pos_μ':>7} {'HNR':>5} {'VE':>4} {'Status':>8}")
    print('-' * 92)

    for epoch in range(1, CFG['epochs'] + 1):
        t0 = time.time()

        curr_stage, curr_split_cfg = resolve_task_stage(epoch=epoch)
        if curr_stage != task_stage:
            task_stage = curr_stage
            print(f"\n🔁 阶段切换: epoch={epoch} -> stage={task_stage}")
            bundle = cold_start_split(
                data,
                val_ratio      = CFG['val_ratio'],
                test_ratio     = CFG['test_ratio'],
                neg_ratio      = CFG['neg_ratio'],
                hard_neg_ratio = CFG['hard_neg_start'],
                pop_neg_ratio  = curr_split_cfg['val_pop_neg_ratio'],
                it_mask_ratio  = curr_split_cfg['it_mask_ratio'],
                seed           = CFG['seed'],
                strict_eval_neg_filter = curr_split_cfg['strict_eval_neg_filter'],
                val_hard_neg_ratio = curr_split_cfg['val_hard_neg_ratio'],
                val_pop_neg_ratio = curr_split_cfg['val_pop_neg_ratio'],
                test_hard_neg_ratio = curr_split_cfg['test_hard_neg_ratio'],
                test_pop_neg_ratio = curr_split_cfg['test_pop_neg_ratio'],
                min_pos_per_herb_val = curr_split_cfg['min_pos_per_herb_val'],
                min_pos_per_herb_test = curr_split_cfg['min_pos_per_herb_test'],
                max_fallback_per_herb = curr_split_cfg['max_fallback_per_herb'],
                use_it_mask = bool(CFG.get('use_it_mask', True)),
                filter_hi_to_train_herbs = bool(CFG.get('filter_hi_to_train_herbs', True)),
            )

            base_msg_graph = bundle.msg_graph.to(device)
            if hasattr(data['ingredient'], 'pos') and data['ingredient'].pos is not None:
                base_msg_graph['ingredient'].pos = data['ingredient'].pos.to(device)
            it_etype = next(
                et for et in base_msg_graph.edge_types
                if et[0] == 'ingredient' and et[2] == 'target'
            )
            train_it_edge = base_msg_graph[it_etype].edge_index
            train_ing_to_tgts = defaultdict(set)
            for i in range(train_it_edge.size(1)):
                ing = train_it_edge[0, i].item()
                tgt = train_it_edge[1, i].item()
                train_ing_to_tgts[ing].add(tgt)

            train_ht_set = set(zip(bundle.train_pos_h.tolist(), bundle.train_pos_t.tolist()))
            ing_degree = torch.zeros(int(data['ingredient'].num_nodes), dtype=torch.float32)
            tgt_degree = torch.zeros(int(data['target'].num_nodes), dtype=torch.float32)
            for i in range(train_it_edge.size(1)):
                ing = int(train_it_edge[0, i].item())
                tgt = int(train_it_edge[1, i].item())
                if 0 <= ing < ing_degree.numel():
                    ing_degree[ing] += 1.0
                if 0 <= tgt < tgt_degree.numel():
                    tgt_degree[tgt] += 1.0
            train_target_popularity = (torch.log1p(tgt_degree).cpu().numpy() + 1e-6)
            neg_sampler = OnlineHTNegSampler(
                all_pos_global=train_ht_set,
                herb_to_ings=bundle.herb_to_ings,
                ing_to_tgts=dict(train_ing_to_tgts),
                num_tgts=data['target'].num_nodes,
                hard_ratio=CFG['hard_neg_start'],
                pop_ratio=CFG.get('pop_neg_start', 0.0),
                target_popularity=train_target_popularity,
                seed=CFG['seed'],
            )
            ve_manager = VirtualEdgeManager(base_graph=base_msg_graph, data=data)
            if len(ve_manager.edge_types) > 0:
                msg_graph = ve_manager.build_virtual_graph(device)
            else:
                msg_graph = base_msg_graph

            if hasattr(model, 'decoder') and hasattr(model.decoder, 'set_frequency_bias'):
                model.decoder.set_frequency_bias(torch.log1p(ing_degree), torch.log1p(tgt_degree))

        # ── 课程学习 ──────────────────────────────────────────
        hard_neg_ratio = get_hard_neg_ratio(epoch)
        pop_neg_ratio = get_pop_neg_ratio(epoch)
        train_neg_ratio = get_train_neg_ratio(epoch)
        neg_sampler.hard_ratio = float(hard_neg_ratio)
        neg_sampler.pop_ratio = float(pop_neg_ratio)

        # 计算召回缺口，用于动态调整正样本权重
        recall_gap = max(0.0, float(CFG.get('adaptive_threshold_target_recall', 0.94)) - prev_val_metrics.get('Recall', 0.0))
        dyn_pos_weight, dyn_neg_weight = get_dynamic_class_weights(train_neg_ratio, recall_gap=recall_gap)

        # ── 训练一个 epoch ────────────────────────────────────
        train_loss = train_one_epoch(
            model          = model,
            msg_graph      = msg_graph,
            train_pos_h    = bundle.train_pos_h,
            train_pos_t    = bundle.train_pos_t,
            herb_to_ings   = bundle.herb_to_ings,
            neg_sampler    = neg_sampler,
            optimizer      = optimizer,
            batch_size     = CFG['batch_size'],
            device         = device,
            pos_weight     = dyn_pos_weight,
            neg_weight     = dyn_neg_weight,
            train_neg_ratio= train_neg_ratio,
            hard_neg_ratio = hard_neg_ratio,
            perm_generator = train_perm_generator,
        )
        scheduler.step()

        # ── 验证（温度校准 + 约束阈值 + 自适应控制）────────────────────
        raw_val_metrics = evaluate(
            model                 = model,
            msg_graph             = msg_graph,
            pos_h                 = bundle.val_pos_h,
            pos_t                 = bundle.val_pos_t,
            neg_h                 = bundle.val_neg_h,
            neg_t                 = bundle.val_neg_t,
            herb_to_ings          = bundle.herb_to_ings,
            batch_size            = CFG['batch_size'],
            device                = device,
            threshold             = None,
            thresh_metric         = THRESHOLD_OPTIMIZATION_METRIC,
            threshold_search_space= tuple(THRESHOLD_SEARCH_SPACE),
            threshold_search_step = float(THRESHOLD_SEARCH_STEPS),
            return_threshold      = True,
            return_logits         = True,
            return_probs          = True,
            temperature           = 1.0,
            lagrange_lambdas      = lagrange_lambdas,
            adaptive_controller   = None,
        )

        if CFG.get('temperature_scaling_enabled', True):
            logits_np = np.array(raw_val_metrics.get('logits', []), dtype=np.float32)
            labels_np = np.array(raw_val_metrics.get('labels', []), dtype=np.float32)
            calibrated_temperature = fit_temperature(
                logits=logits_np,
                labels=labels_np,
                steps=int(CFG.get('temperature_fit_steps', 40)),
                lr=float(CFG.get('temperature_fit_lr', 0.05)),
            )

        val_metrics = evaluate(
            model                 = model,
            msg_graph             = msg_graph,
            pos_h                 = bundle.val_pos_h,
            pos_t                 = bundle.val_pos_t,
            neg_h                 = bundle.val_neg_h,
            neg_t                 = bundle.val_neg_t,
            herb_to_ings          = bundle.herb_to_ings,
            batch_size            = CFG['batch_size'],
            device                = device,
            threshold             = None,
            thresh_metric         = THRESHOLD_OPTIMIZATION_METRIC,
            threshold_search_space= tuple(THRESHOLD_SEARCH_SPACE),
            threshold_search_step = float(THRESHOLD_SEARCH_STEPS),
            return_threshold      = True,
            temperature           = float(calibrated_temperature),
            lagrange_lambdas      = lagrange_lambdas,
            adaptive_controller   = None,
            adaptive_targets      = adaptive_targets,
        )

        lagrange_lambdas = update_constraint_lambdas(
            lambdas=lagrange_lambdas,
            metrics=val_metrics,
            eta=float(CFG.get('constraint_eta', 0.03)),
            max_lambda=float(CFG.get('constraint_max_lambda', 3.0)),
        )

        val_metrics['lambda_precision'] = float(lagrange_lambdas['lambda_precision'])
        val_metrics['lambda_recall'] = float(lagrange_lambdas['lambda_recall'])

        # ── RL 虚拟边更新 ─────────────────────────────────────
        ve_flag = ' '
        if CFG['use_rl_virtual_edge'] and rl_agent is not None:

            # 当前 RL 状态
            val_metrics = attach_virtual_edge_metrics(val_metrics, ve_manager)
            curr_state = build_rl_state(val_metrics, ve_manager)

            # 每 rl_update_interval epoch 让 RL 选择新动作并重建虚拟边
            if epoch >= CFG['rl_action_start_epoch'] and epoch % CFG['rl_update_interval'] == 0:
                pre_action_params = ve_manager.get_edge_params_snapshot()
                pre_action_state = curr_state.copy()
                pre_action_metrics = val_metrics.copy()
                prev_composite_score = composite_score(val_metrics)

                action = np.atleast_1d(rl_agent.select_action(pre_action_state)).astype(np.int64)
                action_deltas = {
                    ntype: rl_agent.actions[int(action[idx])]
                    for idx, ntype in enumerate(ve_manager.edge_types)
                }

                ve_manager.apply_action(action_deltas)
                msg_graph = ve_manager.build_virtual_graph(device)
                ve_flag   = '★'

                post_action_metrics = evaluate(
                    model                 = model,
                    msg_graph             = msg_graph,
                    pos_h                 = bundle.val_pos_h,
                    pos_t                 = bundle.val_pos_t,
                    neg_h                 = bundle.val_neg_h,
                    neg_t                 = bundle.val_neg_t,
                    herb_to_ings          = bundle.herb_to_ings,
                    batch_size            = CFG['batch_size'],
                    device                = device,
                    threshold             = None,
                    thresh_metric         = THRESHOLD_OPTIMIZATION_METRIC,
                    threshold_search_space= tuple(THRESHOLD_SEARCH_SPACE),
                    threshold_search_step = float(THRESHOLD_SEARCH_STEPS),
                    return_threshold      = True,
                    temperature           = float(calibrated_temperature),
                    lagrange_lambdas      = lagrange_lambdas,
                    adaptive_controller   = adaptive_controller,
                    adaptive_targets      = adaptive_targets,
                )
                post_action_metrics = attach_virtual_edge_metrics(post_action_metrics, ve_manager)
                post_action_state = build_rl_state(post_action_metrics, ve_manager)
                action_reward = compute_rl_reward(pre_action_metrics, post_action_metrics)
                new_composite_score = composite_score(post_action_metrics)
                violated = post_action_metrics.get('constraint_violation', 0.0) > 0.0

                if (
                    new_composite_score + CFG['rl_rollback_tolerance'] < prev_composite_score
                    or (CFG.get('rl_force_rollback_on_violation', True) and violated)
                ):
                    rollback_penalty = max(abs(action_reward), 0.03)
                    rl_agent.memory.push(
                        state      = pre_action_state,
                        action     = action,
                        reward     = -float(np.clip(rollback_penalty, 0.0, CFG['rl_reward_clip'])),
                        next_state = post_action_state,
                        done       = False,
                    )
                    ve_manager.load_edge_params(saved_params=pre_action_params)
                    msg_graph = ve_manager.build_virtual_graph(device)
                    ve_flag = 'R'
                    print(f"   RL动作回滚: composite {prev_composite_score:.4f} -> {new_composite_score:.4f}")
                else:
                    rl_agent.memory.push(
                        state      = pre_action_state,
                        action     = action,
                        reward     = action_reward,
                        next_state = post_action_state,
                        done       = False,
                    )
                    val_metrics = post_action_metrics
                    curr_state = post_action_state

                if len(rl_agent.memory) >= int(CFG['rl_buffer_warmup']):
                    rl_agent.update()

                print(f"   🤖 RL动作: {summarize_virtual_edge_action(ve_manager, action_deltas)}")

                # ── RL 动态日志写入（图4数据）──────────────────
                rl_reward = action_reward if ve_flag != 'R' else -max(abs(action_reward), 0.03)
                val_metrics = attach_virtual_edge_metrics(val_metrics, ve_manager)
                primary_edge_type = ve_manager.primary_edge_type
                log_row = {
                    'epoch':       epoch,
                    'task_stage':  task_stage,
                    'threshold':   round(ve_manager.get_threshold(primary_edge_type), 4),
                    'topk':        ve_manager.get_topk(primary_edge_type),
                    'val_auc':     round(val_metrics['AUC'], 4),
                    'val_f1':      round(val_metrics['F1'], 4),
                    'hh_density':  round(float(val_metrics.get('hh_density', 0.0)), 4),
                    'ii_density':  round(float(val_metrics.get('ii_density', 0.0)), 4),
                    'tt_density':  round(float(val_metrics.get('tt_density', 0.0)), 4),
                    'rl_reward':   round(rl_reward, 4),
                    'epsilon':     round(rl_agent.epsilon, 4),
                    'temperature': round(float(val_metrics.get('temperature', calibrated_temperature)), 4),
                    'adaptive_threshold_delta': round(float(val_metrics.get('adaptive_threshold_delta', 0.0)), 4),
                    'constraint_violation': round(float(val_metrics.get('constraint_violation', 0.0)), 4),
                    'lambda_precision': round(float(val_metrics.get('lambda_precision', 0.0)), 4),
                    'lambda_recall': round(float(val_metrics.get('lambda_recall', 0.0)), 4),
                    'val_herb_coverage': round(float(bundle.stats.get('val_herb_coverage', 0.0)), 4),
                    'val_fallback_used_ratio': round(float(bundle.stats.get('val_fallback_used_ratio', 0.0)), 4),
                    'train_neg_ratio': round(float(train_neg_ratio), 4),
                    'hard_neg_ratio': round(float(hard_neg_ratio), 4),
                }
                for ntype in ve_manager.edge_types:
                    log_row[f'{ntype}_threshold'] = round(ve_manager.get_threshold(ntype), 4)
                    log_row[f'{ntype}_topk'] = ve_manager.get_topk(ntype)
                    log_row[f'{ntype}_density'] = round(float(val_metrics.get(f'{ntype}_density', 0.0)), 4)
                rl_writer.writerow(log_row)
                rl_log_file.flush()

            prev_val_metrics = val_metrics.copy()

        # ── Early Stopping & 保存 ────────────────────────────
        rank_score = rank_selection_score(val_metrics)
        comp   = composite_score(val_metrics)
        thresh = val_metrics['best_threshold']
        status = ""

        if rank_score > best_rank_score:
            best_rank_score = rank_score
            patience_cnt = 0
            status = "★ best"
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'val_metrics':      val_metrics,
                'cfg':              CFG,
                've_params':        ve_manager.get_edge_params_snapshot(),
                've_threshold':     ve_manager.get_threshold(ve_manager.primary_edge_type),
                've_topk':          ve_manager.get_topk(ve_manager.primary_edge_type),
                'calibrated_temperature': float(calibrated_temperature),
                'lagrange_lambdas': dict(lagrange_lambdas),
                'selection_metric': {
                    'name': 'rank_selection_auc_auprc',
                    'formula': 'auc_weight*AUC + auprc_weight*AUPRC',
                    'auc_weight': CFG['es_auc_weight'],
                    'auprc_weight': CFG['es_prc_weight'],
                },
            }, os.path.join(CFG['save_dir'], 'best_model.pt'))

            # 同步保存 RL 智能体
            if rl_agent is not None:
                rl_agent.save_model(CFG['rl_save_path'])
        else:
            patience_cnt += 1

        dt = time.time() - t0
        print(f"{epoch:>5d} {train_loss:>7.4f} "
              f"{val_metrics['AUC']:>7.4f} {val_metrics['AUPRC']:>7.4f} "
              f"{val_metrics['F1']:>7.4f} {val_metrics['Precision']:>7.4f} "
              f"{val_metrics['Recall']:>7.4f} {val_metrics['ACC']:>7.4f} "
              f"{val_metrics['PR_gap']:>7.4f} "
              f"{thresh:>6.4f} {val_metrics.get('temperature', 1.0):>6.3f} "
              f"{val_metrics.get('adaptive_threshold_delta', 0.0):>+6.3f} "
              f"{val_metrics.get('constraint_violation', 0.0):>6.3f} "
              f"{val_metrics.get('lambda_recall', 0.0):>5.2f} "
              f"{val_metrics['pos_mean']:>7.4f} "
              f"{hard_neg_ratio:>5.2f} {ve_flag:>4} {status:>8}  "
              f"[neg={train_neg_ratio:>4.2f}, w+={dyn_pos_weight:>4.2f}, w-={dyn_neg_weight:>4.2f}] "
              f"[{dt:.1f}s]")

        if patience_cnt >= CFG['patience'] and epoch >= CFG['min_epochs_before_es']:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    rl_log_file.close()
    print(f"\n  📋 RL动态日志保存: {rl_log_path}")

    # ── 10. 测试前：加载最佳模型并在验证集冻结阈值 ────────────────
    print(f"\n{'='*60}")
    print("  加载最佳模型，并在验证集执行一次阈值冻结...")
    ckpt_path = os.path.join(CFG['save_dir'], 'best_model.pt')
    ckpt = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    # 恢复最佳虚拟边配置
    if len(ve_manager.edge_types) > 0:
        ve_manager.load_edge_params(
            saved_params=ckpt.get('ve_params'),
            legacy_threshold=ckpt.get('ve_threshold', CFG['virtual_edge_threshold']),
            legacy_topk=ckpt.get('ve_topk', CFG['virtual_edge_topk']),
        )
        msg_graph = ve_manager.build_virtual_graph(device)
        restored_text = " | ".join(
            f"{ntype}:τ={ve_manager.get_threshold(ntype):.3f},k={ve_manager.get_topk(ntype)}"
            for ntype in ve_manager.edge_types
        )
        print(f"  恢复虚拟边: {restored_text}")

    frozen_lambdas = ckpt.get('lagrange_lambdas', lagrange_lambdas)
    frozen_temperature = float(ckpt.get('calibrated_temperature', calibrated_temperature))

    frozen_val_metrics = evaluate(
        model                 = model,
        msg_graph             = msg_graph,
        pos_h                 = bundle.val_pos_h,
        pos_t                 = bundle.val_pos_t,
        neg_h                 = bundle.val_neg_h,
        neg_t                 = bundle.val_neg_t,
        herb_to_ings          = bundle.herb_to_ings,
        batch_size            = CFG['batch_size'],
        device                = device,
        threshold             = None,
        thresh_metric         = THRESHOLD_OPTIMIZATION_METRIC,
        threshold_search_space= tuple(THRESHOLD_SEARCH_SPACE),
        threshold_search_step = float(THRESHOLD_SEARCH_STEPS),
        return_threshold      = True,
        temperature           = frozen_temperature,
        lagrange_lambdas      = frozen_lambdas,
        adaptive_controller   = None,
        adaptive_targets      = adaptive_targets,
    )

    frozen_threshold = float(frozen_val_metrics['best_threshold'])
    print(f"  冻结验证阈值: {frozen_threshold:.4f} "
          f"(metric={frozen_val_metrics['threshold_metric']}, "
          f"source={frozen_val_metrics['threshold_source']})")

    ckpt['frozen_val_threshold'] = frozen_threshold
    ckpt['frozen_val_threshold_score'] = float(frozen_val_metrics['best_threshold_score'])
    ckpt['threshold_metric'] = frozen_val_metrics['threshold_metric']
    ckpt['threshold_search_space'] = frozen_val_metrics['threshold_search_space']
    ckpt['threshold_search_step'] = float(frozen_val_metrics['threshold_search_step'])
    ckpt['threshold_source'] = 'frozen_from_validation_once'
    ckpt['frozen_val_metrics'] = frozen_val_metrics
    ckpt['frozen_temperature'] = float(frozen_temperature)
    ckpt['lagrange_lambdas'] = dict(frozen_lambdas)
    torch.save(ckpt, ckpt_path)

    # ── 11. 测试（严格使用冻结验证阈值，不在测试集搜索阈值）────────
    saved_threshold = ckpt.get('frozen_val_threshold', None)
    if saved_threshold is None:
        raise RuntimeError("测试前未找到 frozen_val_threshold，已阻止测试集阈值泄漏风险。")
    print(f"  测试阶段阈值来源: frozen_val_threshold={saved_threshold:.4f}")

    saved_temperature = float(ckpt.get('frozen_temperature', ckpt.get('calibrated_temperature', 1.0)))
    saved_lambdas = ckpt.get('lagrange_lambdas', frozen_lambdas)
    print(f"  测试阶段温度来源: frozen_temperature={saved_temperature:.4f}")

    test_metrics = evaluate(
        model                 = model,
        msg_graph             = msg_graph,
        pos_h                 = bundle.test_pos_h,
        pos_t                 = bundle.test_pos_t,
        neg_h                 = bundle.test_neg_h,
        neg_t                 = bundle.test_neg_t,
        herb_to_ings          = bundle.herb_to_ings,
        batch_size            = CFG['batch_size'],
        device                = device,
        threshold             = float(saved_threshold),
        thresh_metric         = THRESHOLD_OPTIMIZATION_METRIC,
        threshold_search_space= tuple(THRESHOLD_SEARCH_SPACE),
        threshold_search_step = float(THRESHOLD_SEARCH_STEPS),
        return_threshold      = True,
        return_probs          = True,
        return_logits         = True,
        temperature           = float(saved_temperature),
        lagrange_lambdas      = saved_lambdas,
    )

    if test_metrics.get('threshold_source') != 'fixed_input':
        raise RuntimeError("测试阶段阈值来源异常：检测到非固定阈值输入。")

    print(f"\n{'='*60}")
    print(f"  测试集结果 (best epoch={ckpt['epoch']}  "
          f"frozen_threshold={saved_threshold:.4f})")
    print(f"{'='*60}")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"  {k:<24}: {v:.4f}")

    # ── 12. 保存曲线数据（图10、11）──────────────────────────
    probs_arr  = np.array(test_metrics.pop('probs'))
    labels_arr = np.array(test_metrics.pop('labels'))
    logits_arr = np.array(test_metrics.pop('logits')) if 'logits' in test_metrics else None

    curves_dir = CFG['save_dir']
    os.makedirs(curves_dir, exist_ok=True)
    _save_curves_json(
        probs_arr, labels_arr,
        auc   = test_metrics['AUC'],
        auprc = test_metrics['AUPRC'],
        save_path = os.path.join(
            curves_dir,
            f"{_safe_name(CFG.get('curve_alias', 'experiment'))}_curves.json",
        ),
    )

    # ── 13. 保存测试结果 ─────────────────────────────────────
    if CFG.get('paper_export_fig_data', True) and save_test_predictions_csv is not None:
        alias = _safe_name(CFG.get('curve_alias', 'experiment'))
        seed_tag = f"seed{int(CFG.get('seed', 0))}"
        test_herb_ids = torch.cat([bundle.test_pos_h.cpu(), bundle.test_neg_h.cpu()]).numpy()
        test_target_ids = torch.cat([bundle.test_pos_t.cpu(), bundle.test_neg_t.cpu()]).numpy()
        id_name_path = os.path.join('processed', 'index_name.json')
        prediction_csv = os.path.join(curves_dir, f"{alias}_{seed_tag}_test_predictions.csv")
        save_test_predictions_csv(
            save_path=prediction_csv,
            herb_ids=test_herb_ids,
            target_ids=test_target_ids,
            labels=labels_arr,
            probs=probs_arr,
            logits=logits_arr,
            threshold=float(saved_threshold),
            id_name_mapping_path=id_name_path,
        )
        print(f"  [Paper] sample-level predictions saved: {prediction_csv}")

        if export_attention_case_csv is not None and model_variant != 'vanilla_hgt':
            explanation_csv = os.path.join(curves_dir, f"{alias}_{seed_tag}_explanation_cases.csv")
            export_attention_case_csv(
                model=model,
                msg_graph=msg_graph,
                herb_to_ings=bundle.herb_to_ings,
                herb_ids=test_herb_ids,
                target_ids=test_target_ids,
                labels=labels_arr,
                probs=probs_arr,
                save_path=explanation_csv,
                device=device,
                top_cases=int(CFG.get('paper_export_top_cases', 3)),
                top_ingredients=int(CFG.get('paper_export_top_ingredients', 12)),
                id_name_mapping_path=id_name_path,
            )
            print(f"  [Paper] ingredient explanation cases saved: {explanation_csv}")
    ve_snapshot = ve_manager.get_edge_params_snapshot() if ve_manager is not None else {}
    torch.save({
        'test_metrics':   test_metrics,
        'val_metrics':    ckpt['val_metrics'],
        'frozen_val_metrics': frozen_val_metrics,
        'best_threshold': float(saved_threshold),
        'threshold_protocol': {
            'metric': THRESHOLD_OPTIMIZATION_METRIC,
            'search_space': [float(THRESHOLD_SEARCH_SPACE[0]), float(THRESHOLD_SEARCH_SPACE[1])],
            'search_step': float(THRESHOLD_SEARCH_STEPS),
            'policy': 'freeze_once_on_validation_then_apply_on_test',
        },
        'seed':           CFG['seed'],
        'best_epoch':     ckpt['epoch'],
        'cfg':            CFG,
        'bundle_stats':   bundle.stats,
        've_params':      ve_snapshot,
        've_threshold':   ve_manager.get_threshold(ve_manager.primary_edge_type),
        've_topk':        ve_manager.get_topk(ve_manager.primary_edge_type),
    }, os.path.join(CFG['save_dir'], 'results.pt'))

    summary = _build_run_summary(
        test_metrics=test_metrics,
        val_metrics=ckpt['val_metrics'],
        frozen_val_metrics=frozen_val_metrics,
        bundle_stats=bundle.stats,
        ckpt_epoch=ckpt['epoch'],
        saved_threshold=float(saved_threshold),
        save_dir=CFG['save_dir'],
    )
    summary_path = os.path.join(CFG['save_dir'], 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  [Summary] {summary_path}")
    return summary



if __name__ == '__main__':
    main()
