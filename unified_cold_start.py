# -*- coding: utf-8 -*-
from __future__ import annotations
"""
unified_cold_start.py
=====================
冷启动推理：对图中已有的草药，模拟冷启动场景
（移除该草药的 H-I 边，用训练好的 IngredientAwareHTModel 预测靶点）
"""

import argparse
import os
import torch
import json
import numpy as np
from torch_geometric.data import HeteroData
from config import PROCESSED_DIR, MODEL_DIR
from model import IngredientAwareHTModel, VanillaHGTBaselineModel
from cold_start_split import build_herb_ing_padded
from processed import augment_graph_with_similarity_features, compute_similarity_edges
from train import _apply_ingredient_feature_mode

# ==========================================
# 1. 配置
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 冷启动推理配置
COLD_START_CFG = {
    # 目标草药（图中已有，模拟冷启动）
    'query_herbs': ['野菊花'],

    # 遮蔽模式
    # 'full'    : 移除全部 H-I 边（完全冷启动）
    # 'partial' : 随机移除 mask_ratio 比例的 H-I 边
    'mask_mode':  'full',
    'mask_ratio': 1.0,

    # 推理参数
    'top_k':      50,
    'batch_size': 512,

    # 模型配置（与训练时保持一致）
    'hidden_dim': 128,
    'num_layers': 3,
    'num_heads':  4,
    'dropout':    0.1,

    # 文件路径
    'ckpt_path':  os.path.join(MODEL_DIR, 'best_model.pt'),
    'graph_path': os.path.join(PROCESSED_DIR, 'hetero_graph.pt'),
    'map_path':   os.path.join(PROCESSED_DIR, 'index_name.json'),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cold-start inference for trained herb-target models."
    )
    parser.add_argument(
        "--herb",
        type=str,
        default=None,
        help="Single herb name or ID, e.g. --herb 野菊花",
    )
    parser.add_argument(
        "--herbs",
        nargs="*",
        default=None,
        help="Herb names or IDs to query, e.g. --herbs 野菊花 金银花",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help="Path to trained best_model.pt",
    )
    parser.add_argument(
        "--graph-path",
        type=str,
        default=None,
        help="Path to processed hetero_graph.pt",
    )
    parser.add_argument(
        "--map-path",
        type=str,
        default=None,
        help="Path to processed index_name.json",
    )
    parser.add_argument(
        "--mask-mode",
        choices=["full", "partial"],
        default=None,
        help="full removes all H-I edges for the query herb; partial masks a ratio of them",
    )
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=None,
        help="Mask ratio used when --mask-mode partial",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-K targets to return",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=None,
        help="Alias of --top-k",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Inference batch size over targets",
    )
    return parser.parse_args()


def apply_cli_overrides(args):
    if args.herb:
        COLD_START_CFG["query_herbs"] = [str(args.herb)]
    elif args.herbs:
        COLD_START_CFG["query_herbs"] = list(args.herbs)
    if args.ckpt_path:
        COLD_START_CFG["ckpt_path"] = args.ckpt_path
    if args.graph_path:
        COLD_START_CFG["graph_path"] = args.graph_path
    if args.map_path:
        COLD_START_CFG["map_path"] = args.map_path
    if args.mask_mode:
        COLD_START_CFG["mask_mode"] = args.mask_mode
    if args.mask_ratio is not None:
        COLD_START_CFG["mask_ratio"] = float(args.mask_ratio)
    if args.top_k is not None:
        COLD_START_CFG["top_k"] = int(args.top_k)
    elif args.topk is not None:
        COLD_START_CFG["top_k"] = int(args.topk)


def _find_latest_checkpoint() -> str | None:
    preferred = []
    fallback = []
    for root, _, files in os.walk(MODEL_DIR):
        if "best_model.pt" not in files:
            continue
        path = os.path.join(root, "best_model.pt")
        norm = path.replace("\\", "/").lower()
        if "/runs/proposed_model/" in norm:
            preferred.append(path)
        else:
            fallback.append(path)

    candidates = preferred or fallback
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]
    if args.batch_size is not None:
        COLD_START_CFG["batch_size"] = int(args.batch_size)


# ==========================================
# 2. 加载资源
# ==========================================

def load_resources():
    print("[Cold Start] Loading graph and model resources...")

    # 加载图数据
    try:
        data = torch.load(COLD_START_CFG['graph_path'], weights_only=False)
    except TypeError:
        data = torch.load(COLD_START_CFG['graph_path'])
    data = augment_graph_with_similarity_features(data)

    # 加载名称映射
    with open(COLD_START_CFG['map_path'], 'r', encoding='utf-8') as f:
        id_maps = json.load(f)

    # 建立 名字(小写) -> ID 的映射
    name_to_idx = {}
    for ntype, mapping in id_maps.items():
        name_to_idx[ntype] = {
            v.strip().lower(): int(k)
            for k, v in mapping.items()
        }

    return data, id_maps, name_to_idx


def _prepare_message_passing_graph(data: HeteroData) -> HeteroData:
    """
    对齐训练时的 msg_graph 协议：
    1. 删除 herb-target 直连边
    2. 为现有关系边补齐反向边
    """
    out = data.clone()

    for key in list(out.edge_types):
        s, _, d = key
        if (s == 'herb' and d == 'target') or (s == 'target' and d == 'herb'):
            del out[key]

    existing_edge_types = set(out.edge_types)
    for edge_type in list(out.edge_types):
        s, r, d = edge_type
        rev_key = (d, f'rev_{r}', s)
        if rev_key in existing_edge_types:
            continue
        out[rev_key].edge_index = out[edge_type].edge_index.flip(0)
        edge_weight = getattr(out[edge_type], 'edge_weight', None)
        if edge_weight is not None:
            out[rev_key].edge_weight = edge_weight.clone()
        existing_edge_types.add(rev_key)

    return out


def _set_frequency_bias(model, graph: HeteroData):
    if not hasattr(model, 'decoder') or not hasattr(model.decoder, 'set_frequency_bias'):
        return

    it_etype = _find_edge_type(graph, 'ingredient', 'target')
    if it_etype is None:
        return

    num_ing = int(graph['ingredient'].num_nodes)
    num_tgt = int(graph['target'].num_nodes)
    ing_degree = torch.zeros(num_ing, dtype=torch.float32)
    tgt_degree = torch.zeros(num_tgt, dtype=torch.float32)
    edge_index = graph[it_etype].edge_index

    for idx in range(edge_index.size(1)):
        ing = int(edge_index[0, idx].item())
        tgt = int(edge_index[1, idx].item())
        if 0 <= ing < num_ing:
            ing_degree[ing] += 1.0
        if 0 <= tgt < num_tgt:
            tgt_degree[tgt] += 1.0

    model.decoder.set_frequency_bias(torch.log1p(ing_degree), torch.log1p(tgt_degree))


def load_trained_model(data: HeteroData):
    """
    加载训练好的 IngredientAwareHTModel。
    edge_types 排除 H-T 边，与训练时 msg_graph 一致。
    """
    ckpt_path = COLD_START_CFG['ckpt_path']
    if not os.path.exists(ckpt_path):
        auto_ckpt = _find_latest_checkpoint()
        if auto_ckpt is not None:
            ckpt_path = auto_ckpt
            COLD_START_CFG['ckpt_path'] = ckpt_path
            print(f"   Auto-selected checkpoint: {ckpt_path}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    saved_cfg = ckpt['cfg']
    data = _apply_ingredient_feature_mode(
        data,
        saved_cfg.get('ingredient_feature_mode', 'all'),
    )
    base_graph = _prepare_message_passing_graph(data)
    virtual_edge_params = ckpt.get('ve_params')
    if not virtual_edge_params:
        legacy_threshold = ckpt.get('ve_threshold', saved_cfg.get('virtual_edge_threshold', 0.75))
        legacy_topk = ckpt.get('ve_topk', saved_cfg.get('virtual_edge_topk', 5))
        legacy_types = saved_cfg.get('virtual_edge_types', [])
        virtual_edge_params = {
            ntype: {
                'threshold': float(legacy_threshold),
                'topk': int(legacy_topk),
            }
            for ntype in legacy_types
        }

    # edge_types 排除 H-T 边（与训练时 msg_graph 一致）
    edge_types = list(base_graph.edge_types)
    edge_type_set = set(edge_types)
    for ntype in virtual_edge_params.keys():
        rel = (ntype, f'{ntype}_similar_{ntype}', ntype)
        rev_rel = (ntype, f'rev_{ntype}_similar_{ntype}', ntype)
        if rel not in edge_type_set:
            edge_types.append(rel)
            edge_type_set.add(rel)
        if rev_rel not in edge_type_set:
            edge_types.append(rev_rel)
            edge_type_set.add(rev_rel)

    in_dim_dict = {nt: data[nt].x.size(1) for nt in data.node_types}
    model_variant = str(saved_cfg.get('model_variant', 'proposed')).lower()
    encoder_backbone = str(saved_cfg.get('encoder_backbone', 'hgt')).lower()

    if model_variant == 'vanilla_hgt':
        model = VanillaHGTBaselineModel(
            node_types=data.node_types,
            edge_types=edge_types,
            in_dim_dict=in_dim_dict,
            hidden_dim=saved_cfg['hidden_dim'],
            num_layers=saved_cfg['num_layers'],
            num_heads=saved_cfg['num_heads'],
            dropout=saved_cfg['dropout'],
        ).to(DEVICE)
    else:
        model = IngredientAwareHTModel(
            node_types=data.node_types,
            edge_types=edge_types,
            in_dim_dict=in_dim_dict,
            hidden_dim=saved_cfg['hidden_dim'],
            num_layers=saved_cfg['num_layers'],
            num_heads=saved_cfg['num_heads'],
            dropout=saved_cfg['dropout'],
            attention_activation=saved_cfg.get('attention_activation', 'softmax'),
            attention_normalization=saved_cfg.get('attention_normalization', 'length_mean'),
            freq_bias_beta=float(saved_cfg.get('freq_bias_beta', 0.0)),
            use_spatial_encoder=bool(saved_cfg.get('use_spatial_encoder', True)),
            spatial_dim=int(saved_cfg.get('spatial_dim', 64)),
            spatial_dropout=float(saved_cfg.get('spatial_dropout', 0.1)),
            spatial_max_atoms=int(saved_cfg.get('spatial_max_atoms', 64)),
            spatial_dist_hidden_dim=int(saved_cfg.get('spatial_dist_hidden_dim', 64)),
            spatial_centroid_hidden_dim=int(saved_cfg.get('spatial_centroid_hidden_dim', 16)),
            spatial_semantic_bias=float(saved_cfg.get('spatial_semantic_bias', 1.5)),
            encoder_backbone=encoder_backbone,
            decoder_use_ingredient_path=bool(saved_cfg.get('decoder_use_ingredient_path', True)),
            decoder_use_tri_attention=bool(saved_cfg.get('decoder_use_tri_attention', True)),
            decoder_use_global_path=bool(saved_cfg.get('decoder_use_global_path', True)),
        ).to(DEVICE)

    model.load_state_dict(ckpt['model_state_dict'])
    model.virtual_edge_params = virtual_edge_params
    model.inference_temperature = float(
        ckpt.get('frozen_temperature', ckpt.get('calibrated_temperature', 1.0))
    )
    model.inference_threshold = float(ckpt.get('frozen_val_threshold', 0.5))
    _set_frequency_bias(model, base_graph)
    model.eval()

    print(f"   [OK] Model loaded successfully (epoch={ckpt['epoch']}, "
          f"val_AUC={ckpt['val_metrics']['AUC']:.4f})")
    return model


# ==========================================
# 3. 工具函数
# ==========================================

def _find_edge_type(data: HeteroData, src: str, dst: str):
    for et in data.edge_types:
        s, r, d = et
        if s == src and d == dst:
            return et
    return None


def _inject_virtual_edges(
    graph: HeteroData,
    reference_data: HeteroData,
    virtual_edge_params: dict | None,
):
    if not virtual_edge_params:
        return graph

    out = graph.clone()
    for ntype, params in virtual_edge_params.items():
        if ntype not in reference_data.node_types:
            continue
        node_store = reference_data[ntype]
        feat = node_store.x
        if feat is None:
            continue

        edge_index, edge_weight = compute_similarity_edges(
            feat,
            threshold=float(params.get('threshold', 0.75)),
            top_k=int(params.get('topk', 5)),
            node_type=ntype,
            node_store=node_store,
            return_weights=True,
            mutual=True,
        )
        if edge_index is None or edge_index.size(1) == 0:
            continue

        rel = f'{ntype}_similar_{ntype}'
        out[(ntype, rel, ntype)].edge_index = edge_index
        out[(ntype, rel, ntype)].edge_weight = edge_weight
        out[(ntype, f'rev_{rel}', ntype)].edge_index = edge_index.flip(0)
        out[(ntype, f'rev_{rel}', ntype)].edge_weight = edge_weight
    return out


def _build_masked_graph(
    data:       HeteroData,
    herb_id:    int,
    mask_mode:  str,
    mask_ratio: float,
    seed:       int = 42,
):
    """
    构建冷启动消息传递图：移除目标草药的部分/全部 H-I 边。

    Returns:
        masked_graph : 移除边后的图（用于 GNN 消息传递）
        all_ings     : 该草药全量成分 ID 列表
        kept_ings    : 保留的成分 ID 列表
        removed_ings : 被移除的成分 ID 列表
    """
    rng          = np.random.default_rng(seed)
    masked_graph = data.clone()

    hi_etype = _find_edge_type(data, 'herb', 'ingredient')
    assert hi_etype is not None, "找不到 herb→ingredient 边"
    hi_edge = data[hi_etype].edge_index   # [2, N_hi]

    herb_mask = hi_edge[0] == herb_id
    herb_eidx = herb_mask.nonzero(as_tuple=True)[0]
    all_ings  = hi_edge[1, herb_eidx].tolist()

    if mask_mode == 'full':
        keep_mask    = ~herb_mask
        removed_ings = all_ings
        kept_ings    = []

    elif mask_mode == 'partial':
        n_remove   = max(1, int(len(herb_eidx) * mask_ratio))
        remove_pos = rng.choice(len(herb_eidx), size=n_remove, replace=False)
        remove_set = set(herb_eidx[remove_pos].tolist())

        device = hi_edge.device
        edge_keep = torch.ones(hi_edge.size(1), dtype=torch.bool, device=device)
        for pos in remove_set:
            edge_keep[pos] = False

        keep_mask    = edge_keep
        removed_ings = hi_edge[1, ~edge_keep & herb_mask].tolist()
        kept_ings    = hi_edge[1,  edge_keep & herb_mask].tolist()

    else:
        raise ValueError(f"不支持的 mask_mode: {mask_mode}")

    # 更新 H-I 边
    masked_graph[hi_etype].edge_index = hi_edge[:, keep_mask]

    # 同步更新反向边 ingredient→herb（若存在）
    rev_hi_etype = _find_edge_type(masked_graph, 'ingredient', 'herb')
    if rev_hi_etype is not None:
        rev_edge        = data[rev_hi_etype].edge_index
        removed_ing_set = set(removed_ings)
        device = rev_edge.device

        if mask_mode == 'full':
            rev_mask = rev_edge[1] != herb_id
        else:
            rev_mask = torch.tensor([
                not (rev_edge[0, i].item() in removed_ing_set
                     and rev_edge[1, i].item() == herb_id)
                for i in range(rev_edge.size(1))
            ], dtype=torch.bool, device=device)
        masked_graph[rev_hi_etype].edge_index = rev_edge[:, rev_mask]

    print(f"\n   草药 ID={herb_id} 冷启动遮蔽:")
    print(f"     全量成分: {len(all_ings)} 个")
    print(f"     保留成分: {len(kept_ings)} 个")
    print(f"     移除成分: {len(removed_ings)} 个")

    return masked_graph, all_ings, kept_ings, removed_ings


# ==========================================
# 4. 冷启动推理器
# ==========================================

class ColdStartMiner:
    """
    冷启动推理器

    流程：
        1. 解析草药名称 → herb_id
        2. 移除该草药的 H-I 边，构建冷启动图
        3. GNN 在冷启动图上编码（该草药无 H-I 邻居）
        4. 用完整成分列表做成分感知打分（build_herb_ing_padded）
        5. 对全量靶点打分，Top-K 排序
        6. 区分新发现 vs 已知靶点
    """

    def __init__(self, data, id_maps, name_to_idx, model):
        self.data       = data
        self.id_maps    = id_maps
        self.name_map   = name_to_idx   # 全小写 key
        self.model      = model
        self.num_tgts   = data['target'].num_nodes
        self.num_herbs  = data['herb'].num_nodes

        # 构建 herb→成分 完整映射（来自原始图）
        hi_etype = _find_edge_type(data, 'herb', 'ingredient')
        hi_edge  = data[hi_etype].edge_index
        from collections import defaultdict
        self.full_herb_to_ings = defaultdict(list)
        for i in range(hi_edge.size(1)):
            self.full_herb_to_ings[hi_edge[0, i].item()].append(
                hi_edge[1, i].item()
            )

        # 构建已知 H-T 连接（用于区分新发现 vs 已知）
        self.known_ht = set()
        ht_etype = _find_edge_type(data, 'herb', 'target')
        if ht_etype is not None:
            ht_edge = data[ht_etype].edge_index
            for i in range(ht_edge.size(1)):
                self.known_ht.add(
                    (ht_edge[0, i].item(), ht_edge[1, i].item())
                )

    # ------------------------------------------------------------------
    # 草药名称解析（复用原有逻辑）
    # ------------------------------------------------------------------

    def resolve_herb(self, query: str):
        """
        将草药名称解析为节点 ID。
        支持：精确匹配 → 不区分大小写 → 模糊包含匹配 → 直接整数 ID
        """
        q = query.strip().lower()
        herb_map = self.name_map.get('herb', {})

        # 1. 精确匹配（不区分大小写）
        if q in herb_map:
            return herb_map[q]

        # 2. 模糊匹配（包含关系）
        for db_name, idx in herb_map.items():
            if len(q) > 1 and (q in db_name or db_name in q):
                print(f"   ℹ️ 通过模糊匹配 '{db_name}' 找到 '{query}'")
                return idx

        # 3. 直接整数 ID
        try:
            idx = int(query)
            if 0 <= idx < self.num_herbs:
                return idx
        except ValueError:
            pass

        return None

    def get_ingredient_names(self, ing_ids):
        """
        获取成分 ID 列表对应的成分名称（去重）
        """
        ing_map = self.id_maps.get('ingredient', {})
        ing_names = []
        seen = set()
        for ing_id in ing_ids:
            ing_name = ing_map.get(str(ing_id), f'ingredient_{ing_id}')
            if ing_name not in seen:
                ing_names.append(ing_name)
                seen.add(ing_name)
        return ing_names

    # ------------------------------------------------------------------
    # 获取已知 I-T 连接（与原 get_existing_links 对应，改为 H-T）
    # ------------------------------------------------------------------

    def get_known_targets(self, herb_id: int):
        """返回该草药在数据库中已知的靶点 ID 集合"""
        return {t for (h, t) in self.known_ht if h == herb_id}

    # ------------------------------------------------------------------
    # GNN 编码 + 全量靶点打分
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode(self, graph: HeteroData):
        """在给定图上做 GNN 编码，返回所有节点 embedding"""
        graph = _prepare_message_passing_graph(graph)
        graph = _inject_virtual_edges(
            graph,
            self.data,
            getattr(self.model, 'virtual_edge_params', None),
        )
        graph = graph.to(DEVICE)
        # IngredientAwareHTModel 需要实现 encode() 方法
        # 返回 {node_type: embedding_tensor}
        if hasattr(self.model, 'encode'):
            return self.model.encode(graph)
        return self.model.encoder(graph)

    @torch.no_grad()
    def _score_all_targets(self, x_dict, herb_id: int, herb_to_ings: dict):
        """
        对该草药与全量靶点打分。

        Args:
            x_dict       : GNN 编码后的节点 embedding
            herb_id      : 目标草药 ID
            herb_to_ings : 推理时使用的草药→成分映射

        Returns:
            probs : [num_targets]  sigmoid 概率
        """
        batch_size = COLD_START_CFG['batch_size']
        all_scores = []
        herb_tensor = torch.tensor([herb_id], dtype=torch.long, device=DEVICE)

        for start in range(0, self.num_tgts, batch_size):
            end     = min(start + batch_size, self.num_tgts)
            B       = end - start
            tgt_ids = torch.arange(start, end, dtype=torch.long, device=DEVICE)
            herb_ids = herb_tensor.expand(B)

            padded, mask = build_herb_ing_padded(herb_ids, herb_to_ings)

            if hasattr(self.model, 'decode'):
                scores = self.model.decode(
                    x_dict          = x_dict,
                    herb_ids        = herb_ids,
                    target_ids      = tgt_ids,
                    herb_ing_padded = padded.to(DEVICE),
                    herb_ing_mask   = mask.to(DEVICE),
                )
            else:
                herb_emb = self.model.herb_head(x_dict['herb'][herb_ids])
                target_emb = self.model.target_head(x_dict['target'][tgt_ids])
                scores = self.model.scorer(herb_emb, target_emb).squeeze(-1)
            all_scores.append(scores.cpu())

        logits = torch.cat(all_scores)
        safe_temp = float(
            np.clip(getattr(self.model, 'inference_temperature', 1.0), 1e-3, 10.0)
        )
        return torch.sigmoid(logits / safe_temp)   # [num_targets]

    # ------------------------------------------------------------------
    # 主推理入口
    # ------------------------------------------------------------------

    def predict_herb_targets(
        self,
        herb_query: str,
        mask_mode:  str   = None,
        mask_ratio: float = None,
        top_n:      int   = None,
    ):
        """
        对单个草药做冷启动靶点预测。

        Args:
            herb_query : 草药名称或 ID
            mask_mode  : 'full' | 'partial'
            mask_ratio : partial 模式下遮蔽比例
            top_n      : 返回 Top-N 靶点

        Returns:
            new_discoveries : 潜在新靶点列表
            known_hits      : 已知靶点复现列表
        """
        mask_mode  = mask_mode  or COLD_START_CFG['mask_mode']
        mask_ratio = mask_ratio or COLD_START_CFG['mask_ratio']
        top_n      = top_n      or COLD_START_CFG['top_k']

        print(f"\n[Cold Start Inference] Target herb: '{herb_query}'")
        print(f"   遮蔽模式: {mask_mode}  遮蔽比例: {mask_ratio:.0%}")
        print("=" * 75)

        # ── Step 1: 解析草药名称 ──────────────────────────────────
        herb_id = self.resolve_herb(herb_query)
        if herb_id is None:
            print(f"   [ERROR] Herb not found in graph: '{herb_query}'")
            return [], []

        herb_name = self.id_maps['herb'].get(str(herb_id), f'herb_{herb_id}')
        print(f"   [OK] Found: {herb_query} -> ID {herb_id} ({herb_name})")

        known_tgts = self.get_known_targets(herb_id)
        print(f"   [INFO] Known targets for this herb: {len(known_tgts)}")

        # ── Step 2: 构建冷启动图（移除 H-I 边）─────────────────────
        masked_graph, all_ings, kept_ings, removed_ings = _build_masked_graph(
            data       = self.data,
            herb_id    = herb_id,
            mask_mode  = mask_mode,
            mask_ratio = mask_ratio,
        )

        # ── Step 3: GNN 编码（冷启动图，herb 无 H-I 邻居）──────────
        print(f"\n   [INFO] GNN encoding (cold-start graph)...")
        x_dict = self._encode(masked_graph)

        # ── Step 4: 成分感知推理（用完整成分列表）──────────────────
        # 模拟：成分已知（化学分析可得），但图中训练边被遮蔽
        herb_to_ings = {herb_id: all_ings}
        print(f"   [INFO] Ingredients: {len(all_ings)}, Targets: {self.num_tgts}")

        probs = self._score_all_targets(x_dict, herb_id, herb_to_ings)

        # ── 为每个成分预测其作用的靶点 ────────────────────────────
        # 去重成分
        unique_ings = []
        seen = set()
        for ing_id in all_ings:
            if ing_id not in seen:
                unique_ings.append(ing_id)
                seen.add(ing_id)
        
        # 为每个成分预测靶点
        ingredient_targets = {}
        
        for ing_id in unique_ings:
            # 只使用当前成分
            herb_to_single_ing = {herb_id: [ing_id]}
            # 计算该成分对所有靶点的预测概率
            single_ing_probs = self._score_all_targets(x_dict, herb_id, herb_to_single_ing)
            # 获取所有靶点的预测概率
            ing_top_vals, ing_top_idx = torch.topk(single_ing_probs, k=self.num_tgts)
            
            # 存储该成分的靶点预测结果
            ing_name = self.id_maps['ingredient'].get(str(ing_id), f'ingredient_{ing_id}')
            ingredient_targets[ing_name] = []
            
            for prob, tgt_id in zip(ing_top_vals.tolist(), ing_top_idx.tolist()):
                tgt_name = self.id_maps['target'].get(str(tgt_id), f'target_{tgt_id}')
                entry = {
                    'target_id':   tgt_id,
                    'target_name': tgt_name,
                    'prob':        round(prob, 4),
                    'is_known':    tgt_id in known_tgts,
                }
                ingredient_targets[ing_name].append(entry)

        # ── 整理结果 ───────────────────────────────────────────────
        all_results = []
        
        # 按成分分组，收集所有新发现和已知靶点
        for ing_name, targets in ingredient_targets.items():
            for target in targets:
                entry = {
                    'target_id':   target['target_id'],
                    'target_name': target['target_name'],
                    'herb_id':     herb_id,
                    'herb_name':   herb_name,
                    'ingredients': ing_name,
                    'prob':        target['prob'],
                    'is_known':    target['is_known'],
                }
                all_results.append(entry)

        # 按概率排序所有结果
        all_results.sort(key=lambda x: x['prob'], reverse=True)
        
        # 只保留前 top_n 个结果
        top_results = all_results[:top_n]
        
        # 分类为新发现和已知靶点
        new_discoveries = []
        known_hits = []
        
        for entry in top_results:
            if entry['is_known']:
                # 移除 'is_known' 键，因为它不需要在最终结果中
                entry_copy = entry.copy()
                del entry_copy['is_known']
                known_hits.append(entry_copy)
            else:
                entry_copy = entry.copy()
                del entry_copy['is_known']
                new_discoveries.append(entry_copy)

        return new_discoveries, known_hits

    # ------------------------------------------------------------------
    # 对比实验：不同遮蔽程度的预测稳定性
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compare_mask_modes(self, herb_query: str, top_n: int = 20):
        """
        对比 无遮蔽 / partial50% / full 三种模式的 Top-N 预测重叠度。
        重叠度高 → 模型不依赖图拓扑查表，真正学到了成分语义。
        重叠度低 → 模型严重依赖 H-I 边结构信息。
        """
        print(f"\n[Masking Comparison] Herb: '{herb_query}'  Top-{top_n}")
        print("=" * 75)

        herb_id = self.resolve_herb(herb_query)
        if herb_id is None:
            print(f"   [ERROR] Herb not found: '{herb_query}'")
            return

        herb_to_ings = {herb_id: self.full_herb_to_ings[herb_id]}

        def _get_topk_set(graph, mode_name):
            x_dict = self._encode(graph)
            probs  = self._score_all_targets(x_dict, herb_id, herb_to_ings)
            topk   = set(torch.topk(probs, k=top_n).indices.tolist())
            print(f"   [{mode_name}] Top-{top_n} 预测完成")
            return topk

        # 基准：无遮蔽（完整原始图）
        baseline_set = _get_topk_set(self.data.to(DEVICE), '无遮蔽（基准）')

        # partial 50%
        g50, _, _, _ = _build_masked_graph(
            self.data.to(DEVICE), herb_id, 'partial', 0.5
        )
        partial50_set = _get_topk_set(g50, 'partial 50%')

        # full 100%
        g_full, _, _, _ = _build_masked_graph(
            self.data.to(DEVICE), herb_id, 'full', 1.0
        )
        full_set = _get_topk_set(g_full, 'full 100%')

        # 打印重叠度
        print(f"\n   ── Top-{top_n} 预测重叠度 ──")
        for label, pred_set in [('partial 50%', partial50_set),
                                 ('full 100%',  full_set)]:
            overlap = len(baseline_set & pred_set)
            print(f"   baseline ∩ {label}: {overlap}/{top_n} "
                  f"({overlap/top_n*100:.1f}%)")

        overlap_p_f = len(partial50_set & full_set)
        print(f"   partial50% ∩ full:  {overlap_p_f}/{top_n} "
              f"({overlap_p_f/top_n*100:.1f}%)")

        print(f"\n   [Analysis]:")
        print(f"      重叠度高(>70%) → 模型通过成分语义推断，不依赖图拓扑")
        print(f"      重叠度低(<50%) → 模型依赖 H-I 边结构，冷启动泛化能力弱")


# ==========================================
# 5. 主程序
# ==========================================

def main():
    args = parse_args()
    apply_cli_overrides(args)
    query_herbs = COLD_START_CFG['query_herbs']

    print(f"\n[Cold Start Inference] Herb target prediction")
    print(f"   目标草药: {', '.join(query_herbs)}")
    print("=" * 75)

    # 1. 加载资源
    data, id_maps, name_to_idx = load_resources()
    data = data.to(DEVICE)

    # 2. 加载训练好的模型
    model = load_trained_model(data)

    # 3. 初始化推理器
    miner = ColdStartMiner(data, id_maps, name_to_idx, model)

    # 4. 对每个草药做冷启动推理
    for herb_query in query_herbs:
        new_discoveries, known_hits = miner.predict_herb_targets(
            herb_query = herb_query,
            mask_mode  = COLD_START_CFG['mask_mode'],
            mask_ratio = COLD_START_CFG['mask_ratio'],
            top_n      = COLD_START_CFG['top_k'],
        )

        # ── 输出 A: 潜在新靶点 ──────────────────────────────────
        print("\n" + "=" * 100)
        print(f"[New Target Discovery] Herb: {herb_query}")
        print(f"   说明: 冷启动条件下预测的新靶点（数据库未收录）")
        print("=" * 100)
        print(f"{'Rank':<5} | {'Herb':<8} | {'Ingredient':<25} | {'Target':<40} | {'Prob':<8}")
        print("-" * 100)
        top_k = COLD_START_CFG['top_k']
        for i, item in enumerate(new_discoveries[:top_k], 1):
            print(f"{i:<5} | {item['herb_name']:<8} | "
                  f"{item['ingredients']:<25} | {item['target_name']:<40} | {item['prob']:.4f}")

        # ── 输出 B: 已知靶点复现 ────────────────────────────────
        print("\n" + "-" * 100)
        print(f"[Known Target Validation] Herb: {herb_query}")
        print(f"   说明: 模型在冷启动条件下成功复现的已知靶点")
        print("-" * 100)
        if not known_hits:
            print("   (Top-K 预测中未包含已知靶点，冷启动挑战较大)")
        else:
            for i, item in enumerate(known_hits[:min(10, top_k)], 1):
                print(f"{i:<5} | {item['herb_name']:<8} | "
                      f"{item['ingredients']:<25} | {item['target_name']:<40} | {item['prob']:.4f}")

        # 5. 遮蔽对比实验（验证模型是否真正泛化）
        miner.compare_mask_modes(herb_query, top_n=20)

    print("\n[Analysis Suggestions]:")
    print("   1. 关注 Prob > 0.90 的新靶点作为实验验证优先候选。")
    print("   2. 已知靶点复现率高，说明模型冷启动推理可靠。")
    print("   3. 遮蔽对比实验重叠度高，说明预测基于成分语义而非图拓扑查表。")


if __name__ == "__main__":
    main()
