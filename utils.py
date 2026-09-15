# -*- coding: utf-8 -*-
import os
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from torch import nn
from collections import defaultdict
from torch_geometric.data import HeteroData
from typing import Optional
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score
)
from config import PROCESSED_DIR


# =========================
# 数据访问工具函数
# =========================

def get_edge_index(data, edge_type):
    """
    从不同类型的数据对象中获取边索引
    
    参数:
        data: 数据对象，可以是 HeteroData 对象或字典
        edge_type: 边类型元组 (src_type, rel_type, dst_type)
    
    返回:
        边索引张量，形状为 [2, num_edges]
    """
    if hasattr(data, 'edge_index_dict'):
        # 处理 HeteroData 对象
        return data.edge_index_dict.get(edge_type, torch.zeros((2, 0), dtype=torch.long))
    elif isinstance(data, dict):
        # 处理字典类型数据
        return data.get('edge_index_dict', {}).get(edge_type, torch.zeros((2, 0), dtype=torch.long))
    else:
        # 尝试直接访问
        try:
            return data[edge_type].edge_index
        except (KeyError, AttributeError, TypeError):
            # 如果无法访问，返回空的边索引
            return torch.zeros((2, 0), dtype=torch.long)


def get_num_nodes(data, node_type):
    """
    获取指定节点类型的节点数量
    
    参数:
        data: 数据对象，可以是 HeteroData 对象或字典
        node_type: 节点类型
    
    返回:
        节点数量
    """
    try:
        if hasattr(data, 'x_dict'):
            return data.x_dict[node_type].shape[0]
        elif isinstance(data, dict):
            if 'x_dict' in data:
                return data['x_dict'][node_type].shape[0]
            elif node_type in data:
                return data[node_type].num_nodes
        else:
            return data[node_type].num_nodes
    except (KeyError, AttributeError, TypeError):
        # 默认返回 1000
        return 1000


def get_node_types(data):
    """
    从数据对象中获取节点类型列表
    
    参数:
        data: 数据对象，可以是 HeteroData 对象或字典
    
    返回:
        节点类型列表
    """
    if hasattr(data, 'x_dict'):
        return list(data.x_dict.keys())
    elif isinstance(data, dict):
        if 'x_dict' in data:
            return list(data['x_dict'].keys())
        elif 'metadata' in data:
            return list(data['metadata'].get('x_dim', {}).keys())
    
    # 默认返回常见节点类型
    return ['herb', 'ingredient', 'target']


def get_x_dict(data):
    """
    从数据对象中获取节点特征字典
    
    参数:
        data: 数据对象，可以是 HeteroData 对象或字典
    
    返回:
        节点特征字典
    """
    if hasattr(data, 'x_dict'):
        return data.x_dict
    elif isinstance(data, dict):
        return data.get('x_dict', {})
    else:
        return {}


def get_edge_index_dict(data):
    """
    从数据对象中获取边索引字典
    
    参数:
        data: 数据对象，可以是 HeteroData 对象或字典
    
    返回:
        边索引字典
    """
    if hasattr(data, 'edge_index_dict'):
        return data.edge_index_dict
    elif isinstance(data, dict):
        return data.get('edge_index_dict', {})
    else:
        return {}


def create_safe_data_wrapper(data):
    """
    创建一个安全的数据包装器，统一数据访问接口
    
    参数:
        data: 原始数据对象
    
    返回:
        安全的数据包装器
    """
    return {
        'x_dict': get_x_dict(data),
        'edge_index_dict': get_edge_index_dict(data),
        'node_types': get_node_types(data),
        'get_edge_index': lambda edge_type: get_edge_index(data, edge_type),
        'get_num_nodes': lambda node_type: get_num_nodes(data, node_type)
    }


# =========================
# 图数据加载与准备
# =========================
def load_hetero_graph(path: str = None) -> HeteroData:
    """
    加载异构图（HeteroData），并为每个节点类型注册 num_nodes。
    """
    path = path or os.path.join(PROCESSED_DIR, "hetero_graph.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Graph file not found: {path}")

    # 添加weights_only=False以兼容PyTorch 2.6的加载行为变化
    data = torch.load(path, weights_only=False)

    if isinstance(data, HeteroData):
        for ntype in data.node_types:
            if "x" in data[ntype]:
                data[ntype].num_nodes = data[ntype].x.size(0)
            else:
                print(f"⚠️ 节点类型 {ntype} 缺少特征 'x'，无法注册 num_nodes")

    return data


def prepare_data(data: HeteroData, device: Optional[torch.device] = None) -> HeteroData:
    """准备图数据并迁移到指定设备。"""
    try:
        if device is None:
            # 首先尝试使用GPU
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        # 确保数据结构正确
        if not isinstance(data, HeteroData):
            print("⚠️ 数据类型不是HeteroData，尝试转换...")
            return data
        
        # 尝试将数据迁移到设备
        print(f"📤 正在将数据迁移到设备: {device}")
        return data.to(device)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("⚠️ GPU内存不足，尝试使用CPU...")
            device = torch.device("cpu")
            return data.to(device)
        else:
            print(f"⚠️ 数据迁移失败: {e}")
            return data
    except Exception as e:
        print(f"⚠️ 准备数据时出错: {e}")
        return data


# =========================
# 任务样本构建（原有：分类式）
# =========================
def build_task_pairs_hard_negative(
    # 注意：此函数已重命名，为保持兼容性，保留原函数名作为别名
    *args, **kwargs
):
    """build_task_pairs_hard_negative的别名，用于保持向后兼容性"""
    return build_task_pairs_hard_negative_original(*args, **kwargs)

def build_task_pairs_hard_negative_original(
    data, train=True, herb_list=None, split_ratio=0.8, negative_ratio=1.0,
    return_contrastive=False,
):
    """
    构造 Herb-Target 任务对（原始版本，保持与原代码兼容）
    
    可选返回 Herb-Herb 对比学习样本。
    
    改进点：
    1. 先分割正样本，确保训练集和测试集都有足够的正样本
    2. 为训练集和测试集分别生成负样本，避免样本重叠
    3. 确保每个集合中的正负样本比例平衡
    4. 增加样本验证，确保没有重复样本
    5. 修复变量名不一致问题
    6. 使用统一的数据访问接口，增强鲁棒性
    """
    # 使用统一的数据访问接口获取边索引
    edge_hi = get_edge_index(data, ("herb", "has_ingredient", "ingredient"))
    edge_it = get_edge_index(data, ("ingredient", "binds_to", "target"))

    # 1. 构建正样本集合
    ht_set = set()
    herb_name_to_idx = {}
    if herb_list is not None:
        # 获取草药名称到索引的映射
        from unified_cold_start import HeteroGraphColdStartPredictor
        predictor = HeteroGraphColdStartPredictor(verbose=False)
        herb_name_to_idx = predictor.name_mapping.get('herb', {}).get('name_to_index', {})
        # 转换草药名称列表为索引集合
        allowed_herb_indices = set(herb_name_to_idx.get(name, -1) for name in herb_list)
    
    for i in range(edge_hi.shape[1]):
        h, ing = edge_hi[:, i]
        h_item = h.item()
        
        # 如果指定了草药列表，只保留列表中的草药
        if herb_list is not None and h_item not in allowed_herb_indices:
            continue
            
        targets = edge_it[1, edge_it[0] == ing]
        for t in targets.tolist():
            ht_set.add((h_item, t))

    ht_pos = list(ht_set)
    print(f"总正样本数量: {len(ht_pos)}")
    
    # 2. 分割正样本为训练集和测试集
    random.shuffle(ht_pos)
    split_pos = int(len(ht_pos) * split_ratio)
    
    train_pos = ht_pos[:split_pos]
    test_pos = ht_pos[split_pos:]
    
    print(f"训练集正样本: {len(train_pos)}, 测试集正样本: {len(test_pos)}")
    
    # 使用统一的数据访问接口获取靶点数量
    num_targets = get_num_nodes(data, "target")
    all_targets = list(range(num_targets))
    
    # 3. 为训练集和测试集分别生成负样本
    def generate_negatives(pos_samples, num_neg_per_pos):
        """为给定的正样本生成负样本"""
        neg_samples = []
        
        for h, t in pos_samples:
            num_negs = int(num_neg_per_pos)
            negs_for_this = 0
            
            # 计算当前可用的负靶点数量 - 使用所有正样本集合ht_set，而不是当前批次的pos_set
            available_targets = [t_neg for t_neg in all_targets if (h, t_neg) not in ht_set]
            available_count = len(available_targets)
            
            # 如果没有可用的负靶点，尝试随机选择一个靶点作为负样本（允许低概率的假负样本）
            if available_count == 0:
                print(f"提示: 正样本 ({h}, {t}) 没有可用的负靶点，尝试生成假负样本")
                # 随机选择一个靶点作为负样本
                t_neg = random.choice(all_targets)
                neg_samples.append((h, t_neg, 0))
                negs_for_this += 1
                continue
            
            # 如果可用靶点数量小于期望的负样本数，调整期望数量
            adjusted_num_negs = min(num_negs, available_count)
            
            if adjusted_num_negs < num_negs:
                print(f"提示: 正样本 ({h}, {t}) 可用靶点只有 {available_count} 个，调整为 {adjusted_num_negs} 个负样本")
            
            # 直接从可用靶点中随机选择，不需要检查全局重复
            # 这样可以确保找到足够的负样本，同时避免过度限制
            selected_neg_targets = random.sample(available_targets, adjusted_num_negs)
            for t_neg in selected_neg_targets:
                neg_samples.append((h, t_neg, 0))
                negs_for_this += 1
            
            # 只在完全没有找到负样本时显示警告
            if negs_for_this == 0:
                print(f"警告: 正样本 ({h}, {t}) 无法找到任何负样本")
        
        return neg_samples
    
    # 4. 生成训练集和测试集的负样本
    train_neg = generate_negatives(train_pos, negative_ratio)
    test_neg = generate_negatives(test_pos, negative_ratio)
    
    # 5. 构建最终的训练集和测试集
    train_samples = [(h, t, 1) for h, t in train_pos] + train_neg
    test_samples = [(h, t, 1) for h, t in test_pos] + test_neg
    
    # 6. 打乱样本顺序
    random.shuffle(train_samples)
    random.shuffle(test_samples)
    
    # 7. 验证样本质量
    def validate_samples(samples, name):
        """验证样本质量"""
        pos_count = sum(1 for _, _, label in samples if label == 1)
        neg_count = sum(1 for _, _, label in samples if label == 0)
        
        # 检查重复样本
        sample_set = set()
        duplicates = 0
        for h, t, label in samples:
            if (h, t) in sample_set:
                duplicates += 1
            sample_set.add((h, t))
        
        print(f"{name}集: {len(samples)} 样本, {pos_count} 正样本, {neg_count} 负样本")
        print(f"{name}集正负样本比例: {neg_count/pos_count:.2f} (期望: {negative_ratio})")
        if duplicates > 0:
            print(f"{name}集重复样本数: {duplicates}")
        
        return pos_count, neg_count
    
    validate_samples(train_samples, "训练")
    validate_samples(test_samples, "测试")
    
    # 8. 返回指定数据集
    ht_final = train_samples if train else test_samples
    
    # herb-herb optional contrastive task
    hh_pos, hh_neg = [], []
    if ("herb", "similar", "herb") in data.edge_types:
        edge = data["herb", "similar", "herb"].edge_index
        hh_pos = [(h1.item(), h2.item(), 1) for h1, h2 in edge.t()]
        herbs = list(range(data["herb"].num_nodes))
        
        # 为hh生成负样本时也避免重复
        used_hh_negs = set()
        while len(hh_neg) < len(hh_pos):
            h1, h2 = random.sample(herbs, 2)
            hh_pair = (h1, h2, 0)
            reverse_pair = (h2, h1, 1)
            hh_tuple = (h1, h2)
            if (h1, h2, 1) not in hh_pos and reverse_pair not in hh_pos and hh_tuple not in used_hh_negs:
                hh_neg.append(hh_pair)
                used_hh_negs.add(hh_tuple)
    
    # 分割hh样本
    hh_all = hh_pos + hh_neg
    if hh_all:
        random.shuffle(hh_all)
        split_hh = int(len(hh_all) * split_ratio)
        hh_final = hh_all[:split_hh] if train else hh_all[split_hh:]
    else:
        hh_final = []

    if return_contrastive:
        contrastive_pairs = build_herb_contrastive_pairs(data)
        return ht_final, hh_final, contrastive_pairs
    else:
        return ht_final, hh_final


# 为保持向后兼容性，添加build_task_pairs作为build_task_pairs_hard_negative的别名
build_task_pairs = build_task_pairs_hard_negative

def build_herb_contrastive_pairs(
    data: HeteroData,
    min_shared: int = 1,
    max_pair: int = 5000
):
    """
    基于共享 target/ingredient 构建 Herb-Herb 的对比学习样本。
    正样本：共享 target 数 >= min_shared
    负样本：共享 target=0 且共享 ingredient=0
    """
    edge_hi = data["herb", "has_ingredient", "ingredient"].edge_index
    edge_it = data["ingredient", "binds_to", "target"].edge_index

    herb_to_ing = defaultdict(set)
    ing_to_target = defaultdict(set)
    for h, i in edge_hi.t().tolist():
        herb_to_ing[h].add(i)
    for i, t in edge_it.t().tolist():
        ing_to_target[i].add(t)

    herb_to_target = defaultdict(set)
    for h in herb_to_ing:
        for i in herb_to_ing[h]:
            herb_to_target[h].update(ing_to_target.get(i, set()))

    herbs = list(herb_to_ing.keys())
    seen = set()
    pos_pairs, neg_pairs = [], []

    for i in range(len(herbs)):
        for j in range(i + 1, len(herbs)):
            h1, h2 = herbs[i], herbs[j]
            if (h1, h2) in seen or (h2, h1) in seen:
                continue
            seen.add((h1, h2))

            shared_tar = herb_to_target[h1] & herb_to_target[h2]
            shared_ing = herb_to_ing[h1] & herb_to_ing[h2]

            if len(shared_tar) >= min_shared:
                pos_pairs.append((h1, h2, 1))
            elif len(shared_tar) == 0 and len(shared_ing) == 0:
                neg_pairs.append((h1, h2, -1))

    random.shuffle(pos_pairs)
    random.shuffle(neg_pairs)
    num = min(max_pair, len(pos_pairs), len(neg_pairs))
    return pos_pairs[:num] + neg_pairs[:num]


# =========================
# 新增：排序/成对损失用三元组
# =========================
def build_ht_pairs_ranked(
    data: HeteroData,
    max_pos_per_herb: Optional[int] = None,  # None表示不限制数量
    neg_per_pos: int = 5,
    popularity_log_smooth: float = 1.0,
    hard_negative: bool = True,
    seed: int = 42
):
    """
    构造 Ranking 三元组 (h, t_pos, t_neg)：
    - 正样本：来自 H-I-T 的可达集合（全部使用，无数量限制）
    - 负样本：不在正集合中；若 hard_negative=True，则按"靶点受欢迎度"加权采样
    - 受欢迎度 popularity[t] = in_degree_{ingredient->target}(t)（仅基于成分-靶点关系）

    参数
    ----
    max_pos_per_herb : 每个 herb 最多取多少个正样本（None表示使用所有可达关系）
    neg_per_pos      : 每个正样本配多少个负样本
    popularity_log_smooth : 受欢迎度采样时的 log(1+pop) 平滑项
    hard_negative    : 是否启用受欢迎度加权采样（True 推荐）
    seed             : 随机种子

    返回
    ----
    triplets : List[(h, t_pos, t_neg)]
    target_pop : np.ndarray，长度 = num_targets（受欢迎度）
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # 1) 取边（仅H-I-T三元关系，无配体数据）
    # 使用统一的数据访问接口获取边索引
    edge_hi = get_edge_index(data, ("herb", "has_ingredient", "ingredient"))
    edge_it = get_edge_index(data, ("ingredient", "binds_to", "target"))

    # 2) 获取靶点数量
    num_targets = get_num_nodes(data, "target")
    
    all_targets = list(range(num_targets))

    # 2) Herb->Ingredient，Ingredient->Targets
    herb_to_ing = defaultdict(set)
    ing_to_target = defaultdict(set)
    
    # 只有当edge_hi和edge_it有效时才处理
    if edge_hi is not None and edge_it is not None:
        for h, i in edge_hi.t().tolist():
            herb_to_ing[h].add(i)

        for i, t in edge_it.t().tolist():
            ing_to_target[i].add(t)

    # 3) 正样本集合：Herb 可达 Targets
    herb_to_pos_targets = defaultdict(set)
    for h, ings in herb_to_ing.items():
        for i in ings:
            herb_to_pos_targets[h].update(ing_to_target.get(i, set()))

    # 4) 靶点受欢迎度（仅基于ingredient->target关系）
    pop = np.zeros(num_targets, dtype=np.float64)
    # 来自 ingredient->target
    for _i, t in edge_it.t().tolist():
        pop[t] += 1

    # 平滑以避免全 0
    if hard_negative:
        weights = np.log1p(popularity_log_smooth + pop)  # log(1 + s + pop)
        # 避免全 0 权重
        if np.allclose(weights.sum(), 0.0):
            weights = np.ones_like(weights)
        weights = (weights / weights.sum()).astype(np.float64)
    else:
        weights = None

    # 5) 采样三元组（不限制正样本数量）
    triplets = []
    total_positive_relations = 0
    
    for h, pos_set in herb_to_pos_targets.items():
        if not pos_set:
            continue

        pos_list = list(pos_set)
        rng.shuffle(pos_list)
        
        # 🔥 关键修复：取消正样本数量限制
        if max_pos_per_herb is not None:
            pos_list = pos_list[:max_pos_per_herb]
        
        total_positive_relations += len(pos_list)

        # 负候选集合
        neg_candidates = np.array([t for t in all_targets if t not in pos_set], dtype=np.int64)
        if len(neg_candidates) == 0:
            # 所有 target 都是正样本（极少见），跳过该 herb
            continue

        if hard_negative and weights is not None:
            # 针对该 herb 的负采样权重（从全局权重中取子集）
            sub_w = weights[neg_candidates]
            s = sub_w.sum()
            if s <= 0:
                sub_w = np.ones_like(sub_w, dtype=np.float64) / len(sub_w)
            else:
                sub_w = (sub_w / s).astype(np.float64)

        for t_pos in pos_list:
            if hard_negative and weights is not None:
                # 加权抽样（可重复抽样，避免小集合报错）
                t_negs = np_rng.choice(
                    neg_candidates,
                    size=min(neg_per_pos, len(neg_candidates)),
                    replace=(neg_per_pos > len(neg_candidates)),
                    p=sub_w
                ).tolist()
            else:
                # 均匀随机抽样
                k = min(neg_per_pos, len(neg_candidates))
                t_negs = rng.sample(neg_candidates.tolist(), k=k)

            for t_neg in t_negs:
                triplets.append((h, t_pos, int(t_neg)))

    rng.shuffle(triplets)
    return triplets, pop


# =========================
# 评估（性能优化版本）
# =========================
def evaluate_all(model, data, ht_test, hh_test, path_finder=None, path_encoder=None, 
                 max_paths=None, max_len=None, batch_size=None, skip_metrics=None):
    """
    评估 HT（草药-靶点）与 HH（草药-草药）两个任务，返回多指标（性能优化版本）。
    
    Args:
        model: 模型实例
        data: 图数据
        ht_test: HT测试对
        hh_test: HH测试对  
        path_finder: 路径查找器
        path_encoder: 路径编码器
        max_paths: 最大路径数（使用配置参数）
        max_len: 最大路径长度（使用配置参数）
        batch_size: 批处理大小（使用配置参数）
        skip_metrics: 跳过的指标（使用配置参数）
    """
    # 导入配置参数
    from config import (
        PATH_MAX_PATHS, PATH_MAX_LEN, EVALUATION_BATCH_SIZE, 
        SKIP_PRECISION_RECALL_K, MEMORY_OPTIMIZATION
    )
    
    # 使用配置参数
    max_paths = max_paths or PATH_MAX_PATHS
    max_len = max_len or PATH_MAX_LEN  
    batch_size = batch_size or EVALUATION_BATCH_SIZE
    skip_precision_k = skip_metrics or SKIP_PRECISION_RECALL_K
    
    model.eval()
    device = next(model.parameters()).device
    results = {}

    # ---- HT ----
    if ht_test:
        print(f"🔬 评估 HT 任务: {len(ht_test)} 样本")
        
        # 批量评估优化
        if len(ht_test) > batch_size:
            print(f"⚡ 批量评估模式，批次大小: {batch_size}")
            preds = evaluate_ht_batch(model, data, ht_test, path_finder, path_encoder, 
                                    max_paths, max_len, batch_size)
        else:
            # 单次评估
            with torch.no_grad():
                pairs = [(h, t) for h, t, _ in ht_test]
                labels = torch.tensor([y for _, _, y in ht_test], dtype=torch.float32).to(device)
                raw_logits = model.predict_ht(data, pairs, path_finder, path_encoder, 
                                            max_paths=max_paths, max_len=max_len).to(device)
                preds = torch.sigmoid(raw_logits)

        # 转换为numpy并计算指标
        labels = torch.tensor([y for _, _, y in ht_test], dtype=torch.float32)
        preds_np = preds.cpu().numpy() if hasattr(preds, 'cpu') else preds.numpy()

        # 添加预测值到结果中（用于train.py中的评估）
        results['ht_predictions'] = preds_np.tolist()

        try:
            results['ht_auc'] = roc_auc_score(labels, preds_np)
            results['ht_ap'] = average_precision_score(labels, preds_np)
        except Exception as e:
            print(f"⚠️ HT 指标计算失败: {e}")
            results['ht_auc'] = 0.0
            results['ht_ap'] = 0.0

        y_true = labels.int().numpy()
        y_pred_bin = (preds_np > 0.5).astype(int)
        results['ht_f1'] = f1_score(y_true, y_pred_bin, zero_division=0)
        results['ht_precision'] = precision_score(y_true, y_pred_bin, zero_division=0)
        results['ht_recall'] = recall_score(y_true, y_pred_bin, zero_division=0)

        # Precision@K / Recall@K（可跳过以加速评估）
        if not skip_precision_k:
            print("📊 计算 Precision@K / Recall@K")
            results.update(compute_precision_recall_at_k(ht_test, preds_np))
        
        # 内存优化
        if MEMORY_OPTIMIZATION:
            del preds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---- HH ----
    if hh_test:
        print(f"🔬 评估 HH 任务: {len(hh_test)} 样本")
        
        # 批量评估优化
        if len(hh_test) > batch_size:
            print(f"⚡ HH批量评估模式，批次大小: {batch_size}")
            hh_preds = evaluate_hh_batch(model, data, hh_test, path_finder, path_encoder, 
                                       max_paths, max_len, batch_size)
        else:
            # 单次评估
            with torch.no_grad():
                pairs = [(h1, h2) for h1, h2, _ in hh_test]
                labels = torch.tensor([y for _, _, y in hh_test], dtype=torch.float32).to(device)
                raw_logits = model.predict_hh(data, pairs, path_finder, path_encoder,
                                            max_paths=max_paths, max_len=max_len).to(device)
                hh_preds = torch.sigmoid(raw_logits)

        # 转换为numpy并计算指标
        hh_labels = torch.tensor([y for _, _, y in hh_test], dtype=torch.float32)
        hh_preds_np = hh_preds.cpu().numpy() if hasattr(hh_preds, 'cpu') else hh_preds.numpy()

        try:
            results['hh_auc'] = roc_auc_score(hh_labels, hh_preds_np)
            results['hh_ap'] = average_precision_score(hh_labels, hh_preds_np)
        except Exception as e:
            print(f"⚠️ HH 指标计算失败: {e}")
            results['hh_auc'] = 0.0
            results['hh_ap'] = 0.0
        
        # 内存优化
        if MEMORY_OPTIMIZATION:
            del hh_preds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    results['ht_count'] = len(ht_test) if ht_test else 0
    results['hh_count'] = len(hh_test) if hh_test else 0
    
    print(f"✅ 评估完成 | HT: {results['ht_count']} | HH: {results['hh_count']}")
    return results


def evaluate_multi_ht(model, data, multi_ht_test, batch_size=None):
    """
    评估多成分-多靶点预测性能
    
    Args:
        model: 模型实例
        data: 图数据
        multi_ht_test: 多成分-多靶点测试对，格式为 [(herb_idx, [ingredient_indices], [target_indices], label)]
        batch_size: 批处理大小
    """
    # 导入配置参数
    from config import (
        EVALUATION_BATCH_SIZE, MEMORY_OPTIMIZATION
    )
    
    # 使用配置参数
    batch_size = batch_size or EVALUATION_BATCH_SIZE
    
    model.eval()
    device = next(model.parameters()).device
    results = {}

    if multi_ht_test:
        print(f"🔬 评估多成分-多靶点任务: {len(multi_ht_test)} 样本")
        
        # 准备测试数据
        herb_ingredient_pairs = []
        target_sets = []
        labels = []
        
        for herb_idx, ingredient_indices, target_indices, label in multi_ht_test:
            herb_ingredient_pairs.append((herb_idx, ingredient_indices))
            target_sets.append(target_indices)
            labels.append(label)
        
        # 批量评估
        if len(multi_ht_test) > batch_size:
            print(f"⚡ 批量评估模式，批次大小: {batch_size}")
            
            all_preds = []
            num_batches = (len(multi_ht_test) + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(multi_ht_test))
                
                batch_herb_ingredient_pairs = herb_ingredient_pairs[start_idx:end_idx]
                batch_target_sets = target_sets[start_idx:end_idx]
                
                with torch.no_grad():
                    preds = model.predict_multi_ht(data, batch_herb_ingredient_pairs, batch_target_sets)
                    all_preds.extend(preds.cpu().numpy().tolist())
        else:
            # 单次评估
            with torch.no_grad():
                preds = model.predict_multi_ht(data, herb_ingredient_pairs, target_sets)
                all_preds = preds.cpu().numpy().tolist()
        
        # 转换为numpy并计算指标
        labels_np = np.array(labels, dtype=np.float32)
        preds_np = np.array(all_preds, dtype=np.float32)
        
        # 添加预测值到结果中
        results['multi_ht_predictions'] = all_preds
        
        try:
            results['multi_ht_auc'] = roc_auc_score(labels_np, preds_np)
            results['multi_ht_ap'] = average_precision_score(labels_np, preds_np)
        except Exception as e:
            print(f"⚠️ 多成分-多靶点指标计算失败: {e}")
            results['multi_ht_auc'] = 0.0
            results['multi_ht_ap'] = 0.0
        
        y_true = labels_np.astype(int)
        y_pred_bin = (preds_np > 0.5).astype(int)
        results['multi_ht_f1'] = f1_score(y_true, y_pred_bin, zero_division=0)
        results['multi_ht_precision'] = precision_score(y_true, y_pred_bin, zero_division=0)
        results['multi_ht_recall'] = recall_score(y_true, y_pred_bin, zero_division=0)
        
        # 计算多成分协同效应指标
        results['multi_ingredient_score'] = compute_multi_ingredient_score(multi_ht_test, preds_np)
        
        # 计算多靶点覆盖指标
        results['multi_target_coverage'] = compute_multi_target_coverage(multi_ht_test, preds_np)
        
        # 内存优化
        if MEMORY_OPTIMIZATION:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    results['multi_ht_count'] = len(multi_ht_test) if multi_ht_test else 0
    
    print(f"✅ 多成分-多靶点评估完成 | 样本数: {results['multi_ht_count']}")
    return results


def compute_multi_ingredient_score(multi_ht_test, preds_np):
    """
    计算多成分协同效应指标
    """
    # 简单实现：基于成分数量和预测分数的相关性
    ingredient_counts = [len(ingredients) for _, ingredients, _, _ in multi_ht_test]
    
    # 计算成分数量与预测分数的相关性
    if len(ingredient_counts) > 1:
        correlation = np.corrcoef(ingredient_counts, preds_np)[0, 1]
        return float(correlation)
    else:
        return 0.0


def compute_multi_target_coverage(multi_ht_test, preds_np):
    """
    计算多靶点覆盖指标
    """
    # 简单实现：基于靶点数量和预测分数的相关性
    target_counts = [len(targets) for _, _, targets, _ in multi_ht_test]
    
    # 计算靶点数量与预测分数的相关性
    if len(target_counts) > 1:
        correlation = np.corrcoef(target_counts, preds_np)[0, 1]
        return float(correlation)
    else:
        return 0.0


# =========================
# 批量评估辅助函数
# =========================
def evaluate_ht_batch(model, data, ht_test, path_finder, path_encoder, max_paths, max_len, batch_size):
    """HT任务批量评估"""
    device = next(model.parameters()).device
    all_preds = []
    
    num_batches = (len(ht_test) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(ht_test))
        batch_data = ht_test[start_idx:end_idx]
        
        with torch.no_grad():
            pairs = [(h, t) for h, t, _ in batch_data]
            raw_logits = model.predict_ht(data, pairs, path_finder, path_encoder, 
                                        max_paths=max_paths, max_len=max_len).to(device)
            batch_preds = torch.sigmoid(raw_logits)
            all_preds.extend(batch_preds.cpu().numpy().tolist())
        
        # 内存清理
        if batch_idx % 5 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return torch.tensor(all_preds, dtype=torch.float32)


def evaluate_hh_batch(model, data, hh_test, path_finder, path_encoder, max_paths, max_len, batch_size):
    """HH任务批量评估"""
    device = next(model.parameters()).device
    all_preds = []
    
    num_batches = (len(hh_test) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(hh_test))
        batch_data = hh_test[start_idx:end_idx]
        
        with torch.no_grad():
            pairs = [(h1, h2) for h1, h2, _ in batch_data]
            raw_logits = model.predict_hh(data, pairs, path_finder, path_encoder,
                                        max_paths=max_paths, max_len=max_len).to(device)
            batch_preds = torch.sigmoid(raw_logits)
            all_preds.extend(batch_preds.cpu().numpy().tolist())
        
        # 内存清理
        if batch_idx % 5 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return torch.tensor(all_preds, dtype=torch.float32)


def compute_precision_recall_at_k(test_data, predictions, k_values=[1, 5, 10]):
    """计算Precision@K和Recall@K指标"""
    results = {}
    
    # 按herb分组
    topk_dict = defaultdict(list)
    for (h, t, y), score in zip(test_data, predictions):
        topk_dict[h].append((t, score, y))
    
    def precision_at_k(k):
        correct, total = 0, 0
        for h in topk_dict:
            sorted_targets = sorted(topk_dict[h], key=lambda x: -x[1])[:k]
            correct += sum(y for _, _, y in sorted_targets)
            total += k
        return correct / total if total > 0 else 0

    def recall_at_k(k):
        correct, total = 0, 0
        for h in topk_dict:
            sorted_targets = sorted(topk_dict[h], key=lambda x: -x[1])[:k]
            hits = sum(y for _, _, y in sorted_targets)
            correct += hits
            total += sum(y for _, _, y in topk_dict[h])
        return correct / total if total > 0 else 0

    for k in k_values:
        results[f'ht_precision@{k}'] = precision_at_k(k)
        results[f'ht_recall@{k}'] = recall_at_k(k)
    
    return results


# =========================
# 映射加载（统一到 index_name.json）
# =========================
def load_id_name_mapping(path: str = None) -> dict:
    """
    读取构图阶段统一输出的 'index_name.json'，返回结构：
    {
      "herb": {"name_to_index": {...}, "index_to_name": {"0": "xx", ...}},
      "ingredient": {...}, "ligand": {...}, "target": {...}, "prescription": {...}
    }
    """
    path = path or os.path.join(PROCESSED_DIR, "index_name.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mapping file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)

    # 兼容旧键名
    for node_type in list(m.keys()):
        d = m[node_type]
        if "name2index" in d and "name_to_index" not in d:
            d["name_to_index"] = d.pop("name2index")
        if "index2name" in d and "index_to_name" not in d:
            d["index_to_name"] = d.pop("index2name")

    return m

# 为保持向后兼容性，添加_load_name_mapping作为load_id_name_mapping的别名
_load_name_mapping = load_id_name_mapping


# =========================
# Herb 属性工具
# =========================
def build_vocab(series, sep=';'):
    """从分号分隔的多标签文本构建词表。"""
    vocab = {}
    # **重要修改**：保留原始命名方式，不进行任何清洗操作
    for text in series.dropna():
        for token in str(text).split(sep):
            # 仅进行类型转换，不进行strip清洗
            token = str(token)
            if token and token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def field_to_index_lists(series, vocab, sep=';'):
    """将多标签文本映射为索引列表（供 MultiLabelEmbedder 使用）。"""
    all_indices = []
    # **重要修改**：保留原始命名方式，不进行任何清洗操作
    for text in series.fillna(''):
        # 仅进行类型转换，不进行strip清洗
        indices = [vocab[token] for token in text.split(sep) if token in vocab]
        all_indices.append(indices)
    return all_indices


class MultiLabelEmbedder(nn.Module):
    """
    多标签平均池化嵌入：
    - 输入：List[List[int]]（每个样本一个索引列表）
    - 输出：每个样本一个向量，取已嵌入向量的平均
    """
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, index_lists):
        if len(index_lists) == 0:
            # 空输入保护
            return torch.zeros(
                (0, self.embedding.embedding_dim),
                dtype=torch.float32,
                device=self.embedding.weight.device
            )

        max_len = max(len(lst) for lst in index_lists) if index_lists else 1
        padded = [lst + [0]*(max_len - len(lst)) for lst in index_lists]
        mask = torch.tensor(
            [[1]*len(lst)+[0]*(max_len - len(lst)) for lst in index_lists],
            dtype=torch.float32,
            device=self.embedding.weight.device
        )
        indices = torch.tensor(padded, dtype=torch.long, device=self.embedding.weight.device)
        embeds = self.embedding(indices)
        summed = (embeds * mask.unsqueeze(-1)).sum(dim=1)
        lengths = mask.sum(dim=1, keepdim=True)
        return summed / lengths.clamp(min=1)
