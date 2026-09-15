# -*- coding: utf-8 -*-
from __future__ import annotations
"""
M²-RLHGT 数据预处理模块 (Cold-Start Edition)
在原有基础上新增：保存herb冷启动划分所需的分组信息
"""
import os
import re
import json
import hashlib
from typing import Optional

import torch
import pandas as pd
from torch_geometric.data import HeteroData

try:
    from config import PROCESSED_DIR
except ImportError:
    PROCESSED_DIR = "./processed_data"


SIMILARITY_SCHEMA_VERSION = "professional_v1"
HERB_STRUCT_COLUMNS = (
    "property_flavor",
    "channel_tropism",
    "effect",
    "indication",
)
TEXT_SPLIT_PATTERN = re.compile(r"[，、；;,.。/|]+|\s+")
AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_GROUPS = {
    "hydrophobic": set("AVILMFWYC"),
    "polar": set("STNQYCH"),
    "positive": set("KRH"),
    "negative": set("DE"),
    "aromatic": set("FWYH"),
    "tiny": set("AGSC"),
}
SIMILARITY_RECIPES = {
    "ingredient": (
        ("fp_x", "tanimoto", 0.50),
        ("bert_x", "cosine", 0.18),
        ("gpt_x", "cosine", 0.12),
        ("shape_x", "cosine", 0.20),
    ),
    "herb": (
        ("text_x", "cosine", 0.65),
        ("struct_x", "cosine", 0.35),
    ),
    "target": (
        ("text_x", "cosine", 0.70),
        ("seq_stats_x", "cosine", 0.30),
    ),
}


def load_precomputed_features():
    feat_path = os.path.join(PROCESSED_DIR, "real_features_ultra.pt")
    if os.path.exists(feat_path):
        try:
            return torch.load(feat_path, map_location='cpu', weights_only=False)
        except TypeError:
            return torch.load(feat_path, map_location='cpu')
    raise FileNotFoundError(
        f"❌ 未找到预计算特征文件: {feat_path}！请先运行 generate_data.py"
    )


def align_features(tensor, target_len):
    if tensor is None:
        return None
    current_len = tensor.size(0)
    if current_len == target_len:
        return tensor
    elif current_len > target_len:
        return tensor[:target_len]
    else:
        pad_size = target_len - current_len
        pad_tensor = torch.zeros(
            (pad_size, *tensor.shape[1:]),
            dtype=tensor.dtype,
            device=tensor.device
        )
        return torch.cat([tensor, pad_tensor], dim=0)


def compute_similarity_edges_legacy(feature_tensor, threshold=0.75, top_k=5):
    """供 train.py 中 RL 智能体动态建边的函数"""
    if feature_tensor is None:
        return None
    device = (
        feature_tensor.device
        if feature_tensor.device.type != 'cpu'
        else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    feats = feature_tensor.to(device)
    feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-8)
    sim_matrix = torch.mm(feats, feats.t())
    sim_matrix.fill_diagonal_(0)

    vals, inds = torch.topk(sim_matrix, k=top_k, dim=1)
    mask = vals >= threshold

    src_indices = (
        torch.arange(feats.size(0), device=device)
        .unsqueeze(1)
        .expand_as(inds)
    )
    src = src_indices[mask]
    dst = inds[mask]

    if len(src) == 0:
        return None
    return torch.stack([src, dst], dim=0).cpu()

def _tokenize_cn_terms(value: object) -> list[str]:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    parts = [p.strip() for p in TEXT_SPLIT_PATTERN.split(text) if p.strip()]
    return parts if parts else [text]


def _stable_hash(token: str, dim: int) -> int:
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % max(1, int(dim))


def build_herb_struct_features(df_herb: pd.DataFrame, hash_dim: int = 256) -> torch.Tensor:
    if df_herb is None or len(df_herb) == 0:
        return torch.zeros((0, int(hash_dim)), dtype=torch.float32)

    feats = torch.zeros((len(df_herb), int(hash_dim)), dtype=torch.float32)
    for row_idx, row in df_herb.iterrows():
        token_counts = {}
        for col in HERB_STRUCT_COLUMNS:
            for token in _tokenize_cn_terms(row.get(col, "")):
                key = f"{col}:{token}"
                token_counts[key] = token_counts.get(key, 0.0) + 1.0
        for token, count in token_counts.items():
            feats[row_idx, _stable_hash(token, hash_dim)] += float(count)
    return feats


def build_target_seq_features(df_tgt: pd.DataFrame) -> torch.Tensor:
    if df_tgt is None or len(df_tgt) == 0:
        width = len(AA_VOCAB) + len(AA_GROUPS) + 2
        return torch.zeros((0, width), dtype=torch.float32)

    aa_to_idx = {aa: idx for idx, aa in enumerate(AA_VOCAB)}
    feats = []
    for _, row in df_tgt.iterrows():
        raw_seq = str(row.get("sequence", "")).upper().replace(" ", "").strip()
        seq = "".join(ch for ch in raw_seq if ch in aa_to_idx)
        length = max(len(seq), 1)
        vec = torch.zeros(len(AA_VOCAB) + len(AA_GROUPS) + 2, dtype=torch.float32)

        for aa in seq:
            vec[aa_to_idx[aa]] += 1.0
        vec[:len(AA_VOCAB)] /= float(length)

        offset = len(AA_VOCAB)
        for idx, aa_group in enumerate(AA_GROUPS.values()):
            vec[offset + idx] = sum(1 for aa in seq if aa in aa_group) / float(length)

        aa_comp = vec[:len(AA_VOCAB)]
        valid = aa_comp > 0
        entropy = 0.0
        if bool(valid.any()):
            probs = aa_comp[valid]
            entropy = float((-(probs * torch.log(probs + 1e-8))).sum().item())
        vec[-2] = float(torch.log1p(torch.tensor(float(length))) / 10.0)
        vec[-1] = entropy / 3.5
        feats.append(vec)

    return torch.stack(feats, dim=0)


def build_ingredient_shape_features(coords: torch.Tensor) -> torch.Tensor:
    if coords is None:
        return torch.zeros((0, 24), dtype=torch.float32)
    coords = coords.to(dtype=torch.float32, device="cpu")
    if coords.dim() != 3:
        raise ValueError(f"Expected ingredient coords shape [N, A, 3], got {tuple(coords.shape)}")

    n_nodes, max_atoms, _ = coords.shape
    all_feats = torch.zeros((n_nodes, 24), dtype=torch.float32)
    for idx in range(n_nodes):
        mol = coords[idx]
        mask = mol.abs().sum(dim=-1) > 0
        valid = mol[mask]
        if valid.size(0) == 0:
            continue

        centroid = valid.mean(dim=0, keepdim=True)
        radial = torch.norm(valid - centroid, dim=-1)
        pairwise = torch.cdist(valid.unsqueeze(0), valid.unsqueeze(0)).squeeze(0)
        tri_mask = torch.triu(torch.ones_like(pairwise, dtype=torch.bool), diagonal=1)
        pair_vals = pairwise[tri_mask]
        if pair_vals.numel() == 0:
            pair_vals = torch.zeros(1, dtype=torch.float32)

        feat = torch.zeros(24, dtype=torch.float32)
        feat[0] = float(valid.size(0)) / float(max_atoms)
        feat[1] = float(radial.mean())
        feat[2] = float(radial.std(unbiased=False)) if radial.numel() > 1 else 0.0
        feat[3] = float(radial.max())
        feat[4] = float(pair_vals.mean())
        feat[5] = float(pair_vals.std(unbiased=False)) if pair_vals.numel() > 1 else 0.0
        feat[6] = float(pair_vals.max())

        radial_hist = torch.histc(radial, bins=8, min=0.0, max=8.0)
        pair_hist = torch.histc(pair_vals, bins=9, min=0.0, max=16.0)
        if radial_hist.sum() > 0:
            radial_hist = radial_hist / radial_hist.sum()
        if pair_hist.sum() > 0:
            pair_hist = pair_hist / pair_hist.sum()
        feat[7:15] = radial_hist
        feat[15:24] = pair_hist
        all_feats[idx] = feat
    return all_feats


def _read_table(path: str):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, encoding="gbk")
        except Exception:
            return pd.read_csv(path, encoding="utf-8-sig")


def load_source_tables(required: bool = True):
    file_map = {
        "herb": "herb.xlsx - Sheet1.csv",
        "ingredient": "ingredients.xlsx - Sheet1.csv",
        "target": "target.xlsx - Sheet1.csv",
        "relation": "herb_ingredient_target.xlsx - Sheet1.csv",
    }
    search_dirs = [".", "./data", "./raw", "./data/TCM-PharmBench", "./dataset"]
    dataframes = {}

    for key, fname in file_map.items():
        loaded = None
        for s_dir in search_dirs:
            full_path = os.path.join(s_dir, fname)
            if os.path.exists(full_path):
                print(f"   📨 成功找到 [{key}] 数据: {full_path}")
                loaded = _read_table(full_path)
                break
        if loaded is None:
            xlsx_fname = fname.replace(" - Sheet1.csv", "")
            for s_dir in search_dirs:
                full_path = os.path.join(s_dir, xlsx_fname)
                if os.path.exists(full_path):
                    print(f"   📨 成功找到 [{key}] 数据: {full_path}")
                    loaded = pd.read_excel(full_path)
                    break
        if loaded is None and required:
            raise FileNotFoundError(f"\n❌ 找不到 {key} 的文件 '{fname}'")
        dataframes[key] = loaded
    return dataframes


def augment_graph_with_similarity_features(
    data: HeteroData,
    real_feats: Optional[dict] = None,
    dataframes: Optional[dict] = None,
) -> HeteroData:
    if real_feats is None:
        try:
            real_feats = load_precomputed_features()
        except FileNotFoundError:
            real_feats = {}
    if dataframes is None:
        dataframes = load_source_tables(required=False)

    if "ingredient" in data.node_types:
        store = data["ingredient"]
        n_nodes = store.num_nodes
        if getattr(store, "bert_x", None) is None and real_feats.get("ingredient_bert_features") is not None:
            store.bert_x = align_features(real_feats["ingredient_bert_features"], n_nodes)
        if getattr(store, "gpt_x", None) is None and real_feats.get("ingredient_gpt_features") is not None:
            store.gpt_x = align_features(real_feats["ingredient_gpt_features"], n_nodes)
        if getattr(store, "fp_x", None) is None and real_feats.get("ingredient_fp_features") is not None:
            store.fp_x = align_features(real_feats["ingredient_fp_features"], n_nodes)
        if getattr(store, "pos", None) is None and real_feats.get("ingredient_coords") is not None:
            store.pos = align_features(real_feats["ingredient_coords"], n_nodes)
        if getattr(store, "shape_x", None) is None and getattr(store, "pos", None) is not None:
            store.shape_x = build_ingredient_shape_features(store.pos)

    if "herb" in data.node_types:
        store = data["herb"]
        n_nodes = store.num_nodes
        if getattr(store, "text_x", None) is None:
            herb_text = real_feats.get("herb_ai_features") if real_feats else None
            store.text_x = align_features(herb_text, n_nodes) if herb_text is not None else store.x
        df_herb = dataframes.get("herb") if isinstance(dataframes, dict) else None
        if getattr(store, "struct_x", None) is None and df_herb is not None:
            store.struct_x = align_features(build_herb_struct_features(df_herb), n_nodes)

    if "target" in data.node_types:
        store = data["target"]
        n_nodes = store.num_nodes
        if getattr(store, "text_x", None) is None:
            tgt_text = real_feats.get("target_ai_features") if real_feats else None
            store.text_x = align_features(tgt_text, n_nodes) if tgt_text is not None else store.x
        df_tgt = dataframes.get("target") if isinstance(dataframes, dict) else None
        if getattr(store, "seq_stats_x", None) is None and df_tgt is not None:
            store.seq_stats_x = align_features(build_target_seq_features(df_tgt), n_nodes)

    data.similarity_schema = SIMILARITY_SCHEMA_VERSION
    return data


def _pick_similarity_device(feature_tensor=None, node_store=None) -> torch.device:
    if feature_tensor is not None and getattr(feature_tensor, "device", None) is not None:
        if feature_tensor.device.type != "cpu":
            return feature_tensor.device
    if node_store is not None:
        for attr_name in ("x", "text_x", "bert_x", "gpt_x", "fp_x", "shape_x", "struct_x", "seq_stats_x"):
            tensor = getattr(node_store, attr_name, None)
            if tensor is not None and tensor.device.type != "cpu":
                return tensor.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cosine_similarity_matrix(features: torch.Tensor, device: torch.device):
    if features is None:
        return None
    feats = features.to(device=device, dtype=torch.float32)
    if feats.dim() != 2 or feats.size(0) == 0:
        return None
    feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-8)
    return torch.mm(feats, feats.t()).clamp_(min=0.0, max=1.0)


def _tanimoto_similarity_matrix(features: torch.Tensor, device: torch.device):
    if features is None:
        return None
    feats = (features.to(device=device, dtype=torch.float32) > 0).float()
    if feats.dim() != 2 or feats.size(0) == 0:
        return None
    intersection = torch.mm(feats, feats.t())
    cardinality = feats.sum(dim=1, keepdim=True)
    union = cardinality + cardinality.t() - intersection
    return (intersection / union.clamp_min(1e-8)).clamp_(min=0.0, max=1.0)


def _build_professional_similarity_matrix(node_store, node_type: str, fallback_tensor, device: torch.device):
    recipe = SIMILARITY_RECIPES.get(node_type, ())
    sim_acc = None
    weight_sum = 0.0

    for attr_name, metric, weight in recipe:
        tensor = getattr(node_store, attr_name, None)
        if tensor is None:
            continue
        sim_matrix = (
            _tanimoto_similarity_matrix(tensor, device)
            if metric == "tanimoto"
            else _cosine_similarity_matrix(tensor, device)
        )
        if sim_matrix is None:
            continue
        sim_acc = sim_matrix * float(weight) if sim_acc is None else sim_acc + sim_matrix * float(weight)
        weight_sum += float(weight)

    if sim_acc is None or weight_sum <= 0.0:
        sim_acc = _cosine_similarity_matrix(fallback_tensor, device)
        weight_sum = 1.0 if sim_acc is not None else 0.0

    if sim_acc is None or weight_sum <= 0.0:
        return None
    sim_acc = sim_acc / float(weight_sum)
    sim_acc.fill_diagonal_(0.0)
    return torch.nan_to_num(sim_acc, nan=0.0, posinf=0.0, neginf=0.0)


def _similarity_matrix_to_edges(sim_matrix, threshold: float, top_k: int, mutual: bool = True):
    if sim_matrix is None:
        return None, None

    num_nodes = int(sim_matrix.size(0))
    if num_nodes <= 1:
        return None, None

    k = max(1, min(int(top_k), num_nodes - 1))
    vals, inds = torch.topk(sim_matrix, k=k, dim=1)
    keep = vals >= float(threshold)

    adjacency = torch.zeros_like(sim_matrix, dtype=torch.bool)
    adjacency.scatter_(1, inds, keep)
    adjacency.fill_diagonal_(False)
    if mutual:
        adjacency = adjacency & adjacency.t()

    upper_mask = torch.triu(adjacency, diagonal=1)
    src, dst = upper_mask.nonzero(as_tuple=True)
    if src.numel() == 0:
        return None, None
    edge_index = torch.stack([src, dst], dim=0).cpu()
    edge_weight = sim_matrix[src, dst].to(dtype=torch.float32).cpu()
    return edge_index, edge_weight


def compute_similarity_edges(
    feature_tensor,
    threshold=0.75,
    top_k=5,
    node_type: Optional[str] = None,
    node_store=None,
    return_weights: bool = False,
    mutual: bool = True,
):
    """专业版虚拟边相似性：分类型融合相似度 + mutual-kNN + threshold。"""
    if feature_tensor is None and node_store is None:
        return (None, None) if return_weights else None

    device = _pick_similarity_device(feature_tensor=feature_tensor, node_store=node_store)
    if node_store is not None and node_type is not None:
        sim_matrix = _build_professional_similarity_matrix(
            node_store=node_store,
            node_type=node_type,
            fallback_tensor=feature_tensor,
            device=device,
        )
    else:
        sim_matrix = _cosine_similarity_matrix(feature_tensor, device)
        if sim_matrix is not None:
            sim_matrix.fill_diagonal_(0.0)

    edge_index, edge_weight = _similarity_matrix_to_edges(
        sim_matrix=sim_matrix,
        threshold=threshold,
        top_k=top_k,
        mutual=mutual,
    )
    if return_weights:
        return edge_index, edge_weight
    return edge_index


class HeteroGraphProcessor:
    def __init__(self):
        self.data = HeteroData()
        self.real_feats = None

    def load_data_from_files(self):
        dataframes = load_source_tables(required=True)
        return (
            dataframes['herb'],
            dataframes['ingredient'],
            dataframes['target'],
            dataframes['relation'],
        )

        file_map = {
            'herb':       "herb.xlsx - Sheet1.csv",
            'ingredient': "ingredients.xlsx - Sheet1.csv",
            'target':     "target.xlsx - Sheet1.csv",
            'relation':   "herb_ingredient_target.xlsx - Sheet1.csv"
        }
        search_dirs = ['.', './data', './raw',
                       './data/TCM-PharmBench', './dataset']
        dataframes = {}

        for key, fname in file_map.items():
            loaded = False
            for s_dir in search_dirs:
                full_path = os.path.join(s_dir, fname)
                if os.path.exists(full_path):
                    print(f"   📂 成功找到 [{key}] 数据: {full_path}")
                    try:
                        dataframes[key] = pd.read_csv(
                            full_path, encoding='utf-8'
                        )
                    except UnicodeDecodeError:
                        try:
                            dataframes[key] = pd.read_csv(
                                full_path, encoding='gbk'
                            )
                        except Exception:
                            dataframes[key] = pd.read_csv(
                                full_path, encoding='utf-8-sig'
                            )
                    loaded = True
                    break

            if not loaded:
                xlsx_fname = fname.replace(" - Sheet1.csv", "")
                for s_dir in search_dirs:
                    full_path = os.path.join(s_dir, xlsx_fname)
                    if os.path.exists(full_path):
                        print(f"   📂 成功找到 [{key}] 数据: {full_path}")
                        dataframes[key] = pd.read_excel(full_path)
                        loaded = True
                        break

            if not loaded:
                raise FileNotFoundError(
                    f"\n❌ 找不到 {key} 的文件 '{fname}'！"
                )

        return (dataframes['herb'], dataframes['ingredient'],
                dataframes['target'], dataframes['relation'])

    def process(self):
        print("\n🔨 开始构建异构物理图（Cold-Start Edition）...")
        self.real_feats = load_precomputed_features()

        df_herb, df_ing, df_tgt, df_rel = self.load_data_from_files()

        # ── 1. 挂载节点特征 ────────────────────────────────
        self.data['herb'].x = align_features(
            self.real_feats.get('herb_ai_features'), len(df_herb)
        )
        self.data['ingredient'].x = align_features(
            self.real_feats.get('ingredient_ai_features'), len(df_ing)
        )
        if 'ingredient_coords' in self.real_feats:
            self.data['ingredient'].pos = align_features(
                self.real_feats['ingredient_coords'], len(df_ing)
            )
        self.data['target'].x = align_features(
            self.real_feats.get('target_ai_features'), len(df_tgt)
        )
        augment_graph_with_similarity_features(
            self.data,
            real_feats=self.real_feats,
            dataframes={
                'herb': df_herb,
                'ingredient': df_ing,
                'target': df_tgt,
                'relation': df_rel,
            },
        )

        # ── 2. 构建节点映射 ────────────────────────────────
        herb_map = {
            str(name).strip().lower(): idx
            for idx, name in enumerate(df_herb['chinese_name'])
        }
        ing_map = {
            str(name).strip().lower(): idx
            for idx, name in enumerate(df_ing['ingredient_name'])
        }
        tgt_map = {
            str(name).strip().lower(): idx
            for idx, name in enumerate(df_tgt['target_name'])
        }

        # ── 3. 构建物理边 + 记录herb分组元数据 ────────────
        h_i_src, h_i_dst = [], []
        i_t_src, i_t_dst = [], []

        # ✅ 新增：记录每个herb对应的(ingredient, target)对
        # 用于冷启动划分：key=herb_id, value=[(ing_id, tgt_id), ...]
        herb_to_pairs = {}

        for _, row in df_rel.iterrows():
            h_name = str(row['chinese_name']).strip().lower()
            i_name = str(row['ingredient_name']).strip().lower()
            t_name = str(row['target_name']).strip().lower()

            h_in_map = h_name in herb_map
            i_in_map = i_name in ing_map
            t_in_map = t_name in tgt_map

            if h_in_map and i_in_map:
                h_i_src.append(herb_map[h_name])
                h_i_dst.append(ing_map[i_name])

            if i_in_map and t_in_map:
                i_t_src.append(ing_map[i_name])
                i_t_dst.append(tgt_map[t_name])

            # ✅ 同时记录herb→(ing, tgt)分组关系
            if h_in_map and i_in_map and t_in_map:
                h_id = herb_map[h_name]
                i_id = ing_map[i_name]
                t_id = tgt_map[t_name]
                if h_id not in herb_to_pairs:
                    herb_to_pairs[h_id] = []
                herb_to_pairs[h_id].append([i_id, t_id])

        # 写入物理边
        if h_i_src:
            self.data['herb', 'has_ingredient', 'ingredient'].edge_index = (
                torch.tensor([h_i_src, h_i_dst], dtype=torch.long)
            )
        if i_t_src:
            self.data['ingredient', 'binds_to', 'target'].edge_index = (
                torch.tensor([i_t_src, i_t_dst], dtype=torch.long)
            )

        # ── 4. 保存图 ──────────────────────────────────────
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        torch.save(
            self.data,
            os.path.join(PROCESSED_DIR, "hetero_graph.pt")
        )

        # ── 5. 保存ID-名称映射 ─────────────────────────────
        id_maps = {
            'herb': {
                str(idx): name
                for idx, name in enumerate(df_herb['chinese_name'])
            },
            'ingredient': {
                str(idx): name
                for idx, name in enumerate(df_ing['ingredient_name'])
            },
            'target': {
                str(idx): name
                for idx, name in enumerate(df_tgt['target_name'])
            }
        }
        with open(
            os.path.join(PROCESSED_DIR, "index_name.json"),
            "w", encoding="utf-8"
        ) as f:
            json.dump(id_maps, f, ensure_ascii=False, indent=2)

        # ── 6. ✅ 新增：保存herb分组元数据（冷启动划分用）──
        # herb_to_pairs: {herb_id(int): [[ing_id, tgt_id], ...]}
        # 转为字符串key以兼容JSON
        herb_to_pairs_str = {
            str(k): v for k, v in herb_to_pairs.items()
        }
        with open(
            os.path.join(PROCESSED_DIR, "herb_to_pairs.json"),
            "w", encoding="utf-8"
        ) as f:
            json.dump(herb_to_pairs_str, f, ensure_ascii=False, indent=2)

        # 打印统计
        total_pairs = sum(len(v) for v in herb_to_pairs.values())
        print(f"\n✅ 图构建完成！")
        print(f"{'─'*50}")
        print(f"📊 图谱统计:")
        print(f"   Herb节点:       {self.data['herb'].num_nodes}")
        print(f"   Ingredient节点: {self.data['ingredient'].num_nodes}")
        print(f"   Target节点:     {self.data['target'].num_nodes}")
        print(f"   herb→ingredient边: {len(h_i_src)}")
        print(f"   ingredient→target边: {len(i_t_src)}")
        print(f"   herb分组数:     {len(herb_to_pairs)}")
        print(f"   总(ing,tgt)对数: {total_pairs}")
        print(f"{'─'*50}")
        print(f"💾 保存文件:")
        print(f"   图文件:     {PROCESSED_DIR}/hetero_graph.pt")
        print(f"   ID映射:     {PROCESSED_DIR}/index_name.json")
        print(f"   Herb分组:   {PROCESSED_DIR}/herb_to_pairs.json  ← 新增")
        print(f"{'─'*50}")
        print(f"💡 herb_to_pairs.json 将用于冷启动数据划分，")
        print(f"   确保测试集的herb在训练时完全不可见。")


if __name__ == "__main__":
    processor = HeteroGraphProcessor()
    processor.process()
