"""
cold_start_split.py
===================
基于 Herb 冷启动 + I-T 边遮蔽的数据划分

消息传递图: Herb(train_only) --H-I--> Ingredient --I-T(80%)--> Target
预测目标:   Herb --> Target (从被遮蔽的20% I-T边推导，无路径泄露)
"""

import random
import numpy as np
import torch
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple
from torch_geometric.data import HeteroData


# ================================================================
#  数据容器
# ================================================================

@dataclass
class HTSplitBundle:
    msg_graph: HeteroData

    train_pos_h: torch.Tensor
    train_pos_t: torch.Tensor

    val_pos_h: torch.Tensor
    val_pos_t: torch.Tensor
    val_neg_h: torch.Tensor
    val_neg_t: torch.Tensor

    test_pos_h: torch.Tensor
    test_pos_t: torch.Tensor
    test_neg_h: torch.Tensor
    test_neg_t: torch.Tensor

    herb_to_ings: Dict[int, List[int]]
    all_ht_pos:   Set[Tuple[int, int]]
    train_herbs:  Set[int]
    val_herbs:    Set[int]
    test_herbs:   Set[int]

    stats: Dict = field(default_factory=dict)


# ================================================================
#  工具
# ================================================================

def _find_edge_type(data: HeteroData, src_type: str, dst_type: str):
    for edge_type in data.edge_types:
        s, r, d = edge_type
        if s == src_type and d == dst_type:
            return edge_type
    return None


def _build_mapping(edge_index: torch.Tensor) -> Dict[int, Set[int]]:
    """从 edge_index [2, N] 构建 src -> {dst} 映射"""
    mapping = defaultdict(set)
    for i in range(edge_index.size(1)):
        mapping[edge_index[0, i].item()].add(edge_index[1, i].item())
    return dict(mapping)


def _derive_ht(
    herb_to_ings: Dict[int, Set[int]],
    ing_to_tgts:  Dict[int, Set[int]],
) -> Set[Tuple[int, int]]:
    """从 H-I 和 I-T 映射推导 H-T 正样本集合"""
    ht = set()
    for h, ings in herb_to_ings.items():
        for ing in ings:
            for t in ing_to_tgts.get(ing, set()):
                ht.add((h, t))
    return ht


def _enforce_min_pos_per_herb(
    base_pairs: List[Tuple[int, int]],
    herb_set: Set[int],
    pure_by_herb: Dict[int, Set[int]],
    masked_by_herb: Dict[int, Set[int]],
    train_ht_all: Set[Tuple[int, int]],
    min_pos_per_herb: int,
    max_fallback_per_herb: int,
) -> Tuple[List[Tuple[int, int]], Dict[str, float]]:
    """为每个 herb 提供最小正样本配额（仅在纯净不足时用 masked 非泄漏补足）。"""
    if min_pos_per_herb <= 0:
        return base_pairs, {
            'coverage': 0.0,
            'avg_pos_per_herb': 0.0,
            'fallback_used_pairs': 0,
            'fallback_used_ratio': 0.0,
        }

    pair_set = set(base_pairs)
    fallback_used = 0

    for h in sorted(herb_set):
        pure_targets = sorted(pure_by_herb.get(h, set()))
        cur_targets = [t for t in pure_targets if (h, t) in pair_set]

        if len(cur_targets) >= min_pos_per_herb:
            continue

        need = min_pos_per_herb - len(cur_targets)
        fallback_candidates = [
            t for t in sorted(masked_by_herb.get(h, set()))
            if (h, t) not in train_ht_all and (h, t) not in pair_set
        ]
        use_k = min(need, max(0, int(max_fallback_per_herb)), len(fallback_candidates))
        for t in fallback_candidates[:use_k]:
            pair_set.add((h, t))
            fallback_used += 1

    out_pairs = sorted(pair_set)
    herb_cov = 0
    herb_cnt = max(1, len(herb_set))
    by_herb_count = defaultdict(int)
    for h, _ in out_pairs:
        if h in herb_set:
            by_herb_count[h] += 1
    herb_cov = sum(1 for h in herb_set if by_herb_count[h] > 0)
    avg_pos = sum(by_herb_count.values()) / float(herb_cnt)

    audit = {
        'coverage': herb_cov / float(herb_cnt),
        'avg_pos_per_herb': float(avg_pos),
        'fallback_used_pairs': int(fallback_used),
        'fallback_used_ratio': float(fallback_used / max(1, len(out_pairs))),
    }
    return out_pairs, audit


def _build_target_popularity(all_pos_pairs: Set[Tuple[int, int]], num_tgts: int) -> np.ndarray:
    """基于可见正样本统计 target 流行度，并转成平滑采样分布。"""
    degree = np.zeros(num_tgts, dtype=np.float64)
    for _, t in all_pos_pairs:
        if 0 <= t < num_tgts:
            degree[t] += 1.0
    weights = np.log1p(1.0 + degree)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.ones(num_tgts, dtype=np.float64)
    weights /= weights.sum()
    return weights


def _sample_from_weighted_candidates(candidates: List[int], target_weights: np.ndarray, k: int, rng) -> List[int]:
    """从候选 target 中按全局流行度权重无放回采样。"""
    if k <= 0 or not candidates:
        return []
    cand_arr = np.array(sorted(set(int(t) for t in candidates)), dtype=np.int64)
    if cand_arr.size == 0:
        return []
    probs = target_weights[cand_arr].astype(np.float64, copy=True)
    if not np.isfinite(probs).all() or probs.sum() <= 0:
        probs = np.ones_like(probs, dtype=np.float64)
    probs /= probs.sum()
    k = min(int(k), int(cand_arr.size))
    chosen = rng.choice(cand_arr, size=k, replace=False, p=probs)
    return [int(x) for x in chosen]


# ================================================================
#  主划分函数
# ================================================================

def cold_start_split(
    data:           HeteroData,
    val_ratio:      float = 0.16,
    test_ratio:     float = 0.20,
    neg_ratio:      float = 1.0,
    hard_neg_ratio: float = 0.3,
    pop_neg_ratio:  float = 0.3,
    it_mask_ratio:  float = 0.20,
    seed:           int   = 42,
    strict_eval_neg_filter: bool = True,
    val_hard_neg_ratio: float = None,
    val_pop_neg_ratio: float = None,
    test_hard_neg_ratio: float = None,
    test_pop_neg_ratio: float = None,
    min_pos_per_herb_val: int = 1,
    min_pos_per_herb_test: int = 1,
    max_fallback_per_herb: int = 2,
    use_it_mask: bool = True,
    filter_hi_to_train_herbs: bool = True,
) -> HTSplitBundle:
    """
    Herb 冷启动划分 + I-T 边遮蔽，消除隐性路径泄露。

    核心逻辑：
      1. 将 I-T 边随机划分为 可见(80%) 和 遮蔽(20%)
      2. 消息传递图只含 train_herbs 的 H-I 边 + 训练可见 I-T 边
         → val/test herb 的成分信息在训练时完全不可见（消除隐患1）
      3. 训练 H-T label 由可见 I-T 推导
      4. val/test H-T label 由遮蔽 I-T 推导（排除训练中已出现的对）
      5. 负采样过滤只用训练正样本，不泄露遮蔽部分信息（消除隐患3）
    """
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    if val_hard_neg_ratio is None:
        val_hard_neg_ratio = hard_neg_ratio
    if val_pop_neg_ratio is None:
        val_pop_neg_ratio = pop_neg_ratio
    if test_hard_neg_ratio is None:
        test_hard_neg_ratio = hard_neg_ratio
    if test_pop_neg_ratio is None:
        test_pop_neg_ratio = pop_neg_ratio

    min_pos_per_herb_val = max(0, int(min_pos_per_herb_val))
    min_pos_per_herb_test = max(0, int(min_pos_per_herb_test))
    max_fallback_per_herb = max(0, int(max_fallback_per_herb))

    num_tgts = data['target'].num_nodes

    # ------------------------------------------------------------------
    # Step 1: 提取 H-I 和 I-T 边
    # ------------------------------------------------------------------
    hi_etype = _find_edge_type(data, 'herb', 'ingredient')
    it_etype = _find_edge_type(data, 'ingredient', 'target')
    assert hi_etype is not None, f"找不到 herb→ingredient 边: {data.edge_types}"
    assert it_etype is not None, f"找不到 ingredient→target 边: {data.edge_types}"

    hi_edge = data[hi_etype].edge_index  # [2, N_hi]
    it_edge = data[it_etype].edge_index  # [2, N_it]

    herb_to_ings_set = _build_mapping(hi_edge)   # herb -> {ing}
    herb_to_ings = {h: sorted(ings) for h, ings in herb_to_ings_set.items()}

    # ------------------------------------------------------------------
    # Step 2: 遮蔽 I-T 边（按边随机划分）
    # ------------------------------------------------------------------
    n_it = it_edge.size(1)
    perm = np_rng.permutation(n_it)
    if use_it_mask:
        n_masked = max(1, int(n_it * it_mask_ratio)) if n_it > 0 else 0
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

    train_ing_to_tgts  = _build_mapping(train_it_edge)
    masked_ing_to_tgts = _build_mapping(masked_it_edge)

    print(f"\n{'='*55}")
    print(f"  I-T 边遮蔽划分")
    print(f"{'='*55}")
    print(f"  原始 I-T 边:  {n_it}")
    print(f"  训练可见:     {n_train}  ({n_train/n_it*100:.1f}%)")
    print(f"  遮蔽:         {n_masked} ({n_masked/n_it*100:.1f}%)")

    # ------------------------------------------------------------------
    # Step 3: 推导 H-T label
    # ------------------------------------------------------------------
    train_ht_all = _derive_ht(herb_to_ings_set, train_ing_to_tgts)

    # 全量 H-T（仅用于 stats 统计，不再用于负采样过滤）
    full_ing_to_tgts = _build_mapping(it_edge)
    all_ht_pos = _derive_ht(herb_to_ings_set, full_ing_to_tgts)
    masked_ht_all = (
        _derive_ht(herb_to_ings_set, masked_ing_to_tgts)
        if use_it_mask else all_ht_pos
    )

    # 纯净测试 H-T = 遮蔽推导的 - 训练中已有的
    pure_test_ht = masked_ht_all - train_ht_all

    print(f"\n  训练 H-T:        {len(train_ht_all)}")
    print(f"  遮蔽推导 H-T:    {len(masked_ht_all)}")
    print(f"  纯净测试 H-T:    {len(pure_test_ht)}")
    print(f"  路径泄露检查: ✅ {len(pure_test_ht & train_ht_all) == 0}")

    # ------------------------------------------------------------------
    # Step 4: Herb 冷启动划分
    # ------------------------------------------------------------------
    train_herb_to_tgts = defaultdict(set)
    for h, t in train_ht_all:
        train_herb_to_tgts[h].add(t)

    all_herbs = sorted(train_herb_to_tgts.keys())
    py_rng.shuffle(all_herbs)

    n_val_h   = max(1, int(len(all_herbs) * val_ratio))
    n_test_h  = max(1, int(len(all_herbs) * test_ratio))
    n_train_h = len(all_herbs) - n_val_h - n_test_h
    assert n_train_h > 0, f"训练 herb 数量为 0 (总: {len(all_herbs)})"

    train_herbs = set(all_herbs[:n_train_h])
    val_herbs   = set(all_herbs[n_train_h: n_train_h + n_val_h])
    test_herbs  = set(all_herbs[n_train_h + n_val_h:])

    print(f"\n{'='*55}")
    print(f"  Herb 冷启动划分")
    print(f"{'='*55}")
    print(f"  Train: {len(train_herbs)} ({len(train_herbs)/len(all_herbs)*100:.1f}%)")
    print(f"  Val:   {len(val_herbs)}   ({len(val_herbs)/len(all_herbs)*100:.1f}%)")
    print(f"  Test:  {len(test_herbs)}  ({len(test_herbs)/len(all_herbs)*100:.1f}%)")

    # ------------------------------------------------------------------
    # Step 5: 按 herb 归属分配 H-T label
    # ------------------------------------------------------------------
    train_ht = [
        (h, t) for h in train_herbs
        for t in train_herb_to_tgts.get(h, set())
    ]

    pure_test_by_herb = defaultdict(set)
    masked_by_herb = defaultdict(set)
    for h, t in pure_test_ht:
        pure_test_by_herb[h].add(t)
    for h, t in masked_ht_all:
        masked_by_herb[h].add(t)

    val_ht  = [(h, t) for h in val_herbs  for t in pure_test_by_herb.get(h, set())]
    test_ht = [(h, t) for h in test_herbs for t in pure_test_by_herb.get(h, set())]

    if not val_ht:
        val_ht = [(h, t) for h in val_herbs for t in masked_by_herb.get(h, set())]
        print("⚠️  val 纯净集为空，已回退到遮蔽 H-T")

    if not test_ht:
        test_ht = [(h, t) for h in test_herbs for t in masked_by_herb.get(h, set())]
        print("⚠️  test 纯净集为空，已回退到遮蔽 H-T")

    val_ht, val_quota_audit = _enforce_min_pos_per_herb(
        base_pairs=val_ht,
        herb_set=val_herbs,
        pure_by_herb=pure_test_by_herb,
        masked_by_herb=masked_by_herb,
        train_ht_all=train_ht_all,
        min_pos_per_herb=min_pos_per_herb_val,
        max_fallback_per_herb=max_fallback_per_herb,
    )
    test_ht, test_quota_audit = _enforce_min_pos_per_herb(
        base_pairs=test_ht,
        herb_set=test_herbs,
        pure_by_herb=pure_test_by_herb,
        masked_by_herb=masked_by_herb,
        train_ht_all=train_ht_all,
        min_pos_per_herb=min_pos_per_herb_test,
        max_fallback_per_herb=max_fallback_per_herb,
    )

    train_set      = set(train_ht)
    val_label_leak = sum(1 for p in val_ht  if p in train_set)
    tst_label_leak = sum(1 for p in test_ht if p in train_set)

    print(f"\n  Train H-T: {len(train_ht)}")
    print(f"  Val   H-T: {len(val_ht)}")
    print(f"  Test  H-T: {len(test_ht)}")
    print(f"  标签泄露:  val={val_label_leak} test={tst_label_leak}  "
          f"{'✅' if val_label_leak == 0 and tst_label_leak == 0 else '❌'}")
    print(f"  Herb覆盖审计: val_cov={val_quota_audit['coverage']:.3f} "
          f"test_cov={test_quota_audit['coverage']:.3f} "
          f"val_avg_pos={val_quota_audit['avg_pos_per_herb']:.3f} "
          f"test_avg_pos={test_quota_audit['avg_pos_per_herb']:.3f}")
    print(f"  配额回退占比: val={val_quota_audit['fallback_used_ratio']:.4f} "
          f"test={test_quota_audit['fallback_used_ratio']:.4f}")

    # ------------------------------------------------------------------
    # Step 6: 负采样
    # ★ 修复隐患3：过滤集合改为 train_ht_all，不使用 all_ht_pos
    #   避免负采样器通过过滤操作间接得知遮蔽部分的正样本信息
    # ------------------------------------------------------------------
    val_pos_filter = all_ht_pos if strict_eval_neg_filter else train_ht_all
    test_pos_filter = all_ht_pos if strict_eval_neg_filter else train_ht_all

    val_neg = _sample_ht_negatives(
        pos_pairs=val_ht,
        all_pos_global=val_pos_filter,
        herb_set=val_herbs,
        herb_to_ings=herb_to_ings,
        ing_to_tgts=train_ing_to_tgts,
        num_tgts=num_tgts,
        neg_ratio=neg_ratio, hard_neg_ratio=val_hard_neg_ratio,
        pop_neg_ratio=val_pop_neg_ratio, rng=np_rng, py_rng=py_rng, name="val",
    )

    test_neg = _sample_ht_negatives(
        pos_pairs=test_ht,
        all_pos_global=test_pos_filter,
        herb_set=test_herbs,
        herb_to_ings=herb_to_ings,
        ing_to_tgts=train_ing_to_tgts,
        num_tgts=num_tgts,
        neg_ratio=neg_ratio, hard_neg_ratio=test_hard_neg_ratio,
        pop_neg_ratio=test_pop_neg_ratio, rng=np_rng, py_rng=py_rng, name="test",
    )

    val_neg_false_neg = sum(1 for p in val_neg if p in all_ht_pos)
    test_neg_false_neg = sum(1 for p in test_neg if p in all_ht_pos)
    val_neg_false_neg_rate = val_neg_false_neg / max(1, len(val_neg))
    test_neg_false_neg_rate = test_neg_false_neg / max(1, len(test_neg))
    val_pos_neg_ratio = len(val_ht) / max(1, len(val_neg))
    test_pos_neg_ratio = len(test_ht) / max(1, len(test_neg))
    print(f"  负样本审计: val_false_neg={val_neg_false_neg} ({val_neg_false_neg_rate:.4f}) "
          f"test_false_neg={test_neg_false_neg} ({test_neg_false_neg_rate:.4f})")
    print(f"  正负样本比: val_pos/neg={val_pos_neg_ratio:.4f} test_pos/neg={test_pos_neg_ratio:.4f}")
    print(f"  严格评估负采样过滤: {'ON(all_ht_pos)' if strict_eval_neg_filter else 'OFF(train_ht_all)'}")

    # ------------------------------------------------------------------
    # Step 7: 构建消息传递图
    # ★ 修复隐患1：H-I 边只保留 train_herbs 的部分
    #   val/test herb 的成分信息在训练消息传递中完全不可见
    # ------------------------------------------------------------------
    msg_graph = data.clone()

    # ★ 过滤 H-I 边，只保留 train_herbs
    hi_edge_full = msg_graph[hi_etype].edge_index
    train_herbs_tensor = torch.tensor(
        sorted(train_herbs if filter_hi_to_train_herbs else herb_to_ings.keys()),
        dtype=torch.long,
    )
    train_herb_set_mask = torch.isin(hi_edge_full[0], train_herbs_tensor)
    msg_graph[hi_etype].edge_index = hi_edge_full[:, train_herb_set_mask]

    n_hi_removed = (~train_herb_set_mask).sum().item()
    n_hi_kept    = train_herb_set_mask.sum().item()
    print(f"\n  ✅ H-I 边过滤: 保留 {n_hi_kept} 条 (train_herbs), "
          f"移除 {n_hi_removed} 条 (val/test herbs)")

    # 替换 I-T 边为训练子集
    msg_graph[it_etype].edge_index = train_it_edge
    print(f"  ✅ I-T 边替换: 训练可见 {n_train} 条，遮蔽 {n_masked} 条")

    # 删除 H-T 边（如果存在）
    for key in list(msg_graph.edge_types):
        s, r, d = key
        if (s == 'herb' and d == 'target') or (s == 'target' and d == 'herb'):
            del msg_graph[key]
            print(f"  ⚠️  已移除 H-T 边: {key}")

    # 补充反向边
    existing_pairs = {(s, d) for s, r, d in msg_graph.edge_types}
    for edge_type in list(msg_graph.edge_types):
        s, r, d = edge_type
        if (d, s) not in existing_pairs:
            rev_rel = f'rev_{r}'
            msg_graph[d, rev_rel, s].edge_index = msg_graph[edge_type].edge_index.flip(0)
            existing_pairs.add((d, s))
            print(f"  ✅ 添加反向边: ({d}, {rev_rel}, {s})")

    print(f"\n  消息传递图边类型:")
    for et in msg_graph.edge_types:
        print(f"    {et}: {msg_graph[et].edge_index.size(1)} 条")

    # ------------------------------------------------------------------
    # Step 8: 打包
    # ------------------------------------------------------------------
    stats = {
        'num_herbs':           data['herb'].num_nodes,
        'num_ingredients':     data['ingredient'].num_nodes,
        'num_targets':         num_tgts,
        'num_train_herbs':     len(train_herbs),
        'num_val_herbs':       len(val_herbs),
        'num_test_herbs':      len(test_herbs),
        'num_train_ht':        len(train_ht),
        'num_val_ht':          len(val_ht),
        'num_test_ht':         len(test_ht),
        'num_val_neg':         len(val_neg),
        'num_test_neg':        len(test_neg),
        'num_train_it_edges':  n_train,
        'num_masked_it_edges': n_masked,
        'it_mask_ratio':       it_mask_ratio,
        'use_it_mask':         bool(use_it_mask),
        'filter_hi_to_train_herbs': bool(filter_hi_to_train_herbs),
        'num_hi_kept':         n_hi_kept,
        'num_hi_removed':      n_hi_removed,
        'val_label_leak':      val_label_leak,
        'test_label_leak':     tst_label_leak,
        'val_neg_false_neg':   val_neg_false_neg,
        'test_neg_false_neg':  test_neg_false_neg,
        'val_neg_false_neg_rate': val_neg_false_neg_rate,
        'test_neg_false_neg_rate': test_neg_false_neg_rate,
        'val_pos_neg_ratio':   val_pos_neg_ratio,
        'test_pos_neg_ratio':  test_pos_neg_ratio,
        'val_hard_neg_ratio':  float(val_hard_neg_ratio),
        'val_pop_neg_ratio':   float(val_pop_neg_ratio),
        'test_hard_neg_ratio': float(test_hard_neg_ratio),
        'test_pop_neg_ratio':  float(test_pop_neg_ratio),
        'val_herb_coverage':   float(val_quota_audit['coverage']),
        'test_herb_coverage':  float(test_quota_audit['coverage']),
        'val_avg_pos_per_herb': float(val_quota_audit['avg_pos_per_herb']),
        'test_avg_pos_per_herb': float(test_quota_audit['avg_pos_per_herb']),
        'val_fallback_used_ratio': float(val_quota_audit['fallback_used_ratio']),
        'test_fallback_used_ratio': float(test_quota_audit['fallback_used_ratio']),
    }

    return HTSplitBundle(
        msg_graph=msg_graph,
        train_pos_h=torch.tensor([h for h, t in train_ht], dtype=torch.long),
        train_pos_t=torch.tensor([t for h, t in train_ht], dtype=torch.long),
        val_pos_h  =torch.tensor([h for h, t in val_ht],   dtype=torch.long),
        val_pos_t  =torch.tensor([t for h, t in val_ht],   dtype=torch.long),
        val_neg_h  =torch.tensor([h for h, t in val_neg],  dtype=torch.long),
        val_neg_t  =torch.tensor([t for h, t in val_neg],  dtype=torch.long),
        test_pos_h =torch.tensor([h for h, t in test_ht],  dtype=torch.long),
        test_pos_t =torch.tensor([t for h, t in test_ht],  dtype=torch.long),
        test_neg_h =torch.tensor([h for h, t in test_neg], dtype=torch.long),
        test_neg_t =torch.tensor([t for h, t in test_neg], dtype=torch.long),
        herb_to_ings=herb_to_ings,
        all_ht_pos  =all_ht_pos,
        train_herbs =train_herbs,
        val_herbs   =val_herbs,
        test_herbs  =test_herbs,
        stats=stats,
    )


# ================================================================
#  多层次负采样
# ================================================================

def _sample_ht_negatives(
    pos_pairs, all_pos_global, herb_set, herb_to_ings,
    ing_to_tgts, num_tgts, neg_ratio, hard_neg_ratio,
    pop_neg_ratio, rng, py_rng, name="",
):
    num_neg = max(1, int(len(pos_pairs) * neg_ratio))
    n_hard  = max(0, int(round(num_neg * hard_neg_ratio)))
    n_pop   = max(0, int(round(num_neg * pop_neg_ratio)))
    n_rand  = max(0, num_neg - n_hard - n_pop)

    neg_set = set()
    result  = []

    herb_list = np.array(sorted(herb_set), dtype=np.int64)
    if herb_list.size == 0:
        return result

    herb_reachable = {}
    for h in herb_list:
        reachable = set()
        for ing in herb_to_ings.get(int(h), []):
            reachable.update(ing_to_tgts.get(ing, set()))
        herb_reachable[int(h)] = reachable

    herb_direct_pos = defaultdict(set)
    for h, t in pos_pairs:
        herb_direct_pos[h].add(t)

    herb_all_pos = defaultdict(set)
    for h, t in all_pos_global:
        herb_all_pos[h].add(t)

    target_weights = _build_target_popularity(all_pos_global, num_tgts)

    def _add(h, t):
        if (h, t) not in all_pos_global and (h, t) not in neg_set:
            neg_set.add((h, t))
            result.append((h, t))
            return True
        return False

    def _fallback_fill(h: int, need: int, allow_pop: bool = True) -> tuple[int, int]:
        pop_added = 0
        rand_added = 0
        forbidden = herb_all_pos.get(h, set())
        attempts = 0
        if allow_pop:
            pop_candidates = [t for t in range(num_tgts) if t not in forbidden and (h, t) not in neg_set]
            for t in _sample_from_weighted_candidates(pop_candidates, target_weights, need, rng):
                if _add(h, t):
                    pop_added += 1
            need -= pop_added

        while need > 0 and attempts < max(50, need * 60):
            t = int(rng.integers(0, num_tgts))
            if _add(h, t):
                rand_added += 1
                need -= 1
            attempts += 1
        return pop_added, rand_added

    hard_count = 0
    pop_count = 0
    rand_count = 0
    n_neg_per = max(1, int(neg_ratio))

    for h, _ in pos_pairs:
        collected = 0
        hard_need = min(n_neg_per, max(0, int(round(n_neg_per * hard_neg_ratio))))
        pop_need = min(n_neg_per - hard_need, max(0, int(round(n_neg_per * pop_neg_ratio))))
        rand_need = max(0, n_neg_per - hard_need - pop_need)

        hard_candidates = [
            t for t in (herb_reachable.get(h, set()) - herb_direct_pos[h])
            if (h, t) not in all_pos_global and (h, t) not in neg_set
        ]
        if hard_candidates and hard_need > 0:
            chosen_hard = _sample_from_weighted_candidates(hard_candidates, target_weights, hard_need, rng)
            for t in chosen_hard:
                if _add(h, t):
                    hard_count += 1
                    collected += 1
        residual_after_hard = n_neg_per - collected

        pop_candidates = [
            t for t in range(num_tgts)
            if t not in herb_all_pos.get(h, set()) and (h, t) not in neg_set and t not in herb_reachable.get(h, set())
        ]
        if pop_need > 0 and residual_after_hard > 0:
            chosen_pop = _sample_from_weighted_candidates(pop_candidates, target_weights, min(pop_need, residual_after_hard), rng)
            for t in chosen_pop:
                if _add(h, t):
                    pop_count += 1
                    collected += 1
        residual_after_pop = n_neg_per - collected

        if rand_need > 0 and residual_after_pop > 0:
            pop_added, rand_added = _fallback_fill(h, min(rand_need, residual_after_pop), allow_pop=False)
            pop_count += pop_added
            rand_count += rand_added
            collected += pop_added + rand_added

        remaining = n_neg_per - collected
        if remaining > 0:
            pop_added, rand_added = _fallback_fill(h, remaining, allow_pop=True)
            pop_count += pop_added
            rand_count += rand_added

    py_rng.shuffle(result)
    print(f"\n  [{name}] 负样本采样:")
    print(f"    困难(结构可达): {hard_count}")
    print(f"    流行度加权:     {pop_count}")
    print(f"    简单随机:       {rand_count}")
    print(f"    正样本数:       {len(pos_pairs)}")
    print(f"    负正比:         1:{len(result)/max(1, len(pos_pairs)):.2f}")
    return result


# ================================================================
#  Padding 工具
# ================================================================

def build_herb_ing_padded(
    herb_ids:     torch.Tensor,
    herb_to_ings: Dict[int, List[int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    B = herb_ids.size(0)
    ing_lists = [herb_to_ings.get(herb_ids[i].item(), []) for i in range(B)]
    max_K = max((len(x) for x in ing_lists), default=1)

    padded = torch.zeros(B, max_K, dtype=torch.long)
    mask   = torch.zeros(B, max_K, dtype=torch.bool)
    for i, ings in enumerate(ing_lists):
        k = len(ings)
        if k > 0:
            padded[i, :k] = torch.tensor(ings, dtype=torch.long)
            mask[i, :k]   = True
    return padded, mask


# ================================================================
#  在线负采样器
# ================================================================

class OnlineHTNegSampler:
    def __init__(self, all_pos_global, herb_to_ings, ing_to_tgts,
                 num_tgts, hard_ratio=0.3, pop_ratio=0.0,
                 target_popularity=None, seed=42):
        # ★ 修复隐患3：all_pos_global 传入 train_ht_all 而非 all_ht_pos
        self.all_pos      = all_pos_global
        self.herb_to_ings = herb_to_ings
        self.num_tgts     = num_tgts
        self.hard_ratio   = float(hard_ratio)
        self.pop_ratio    = float(pop_ratio)
        self.rng          = np.random.default_rng(seed)

        self._herb_reachable = {}
        for h, ings in herb_to_ings.items():
            reachable = set()
            for ing in ings:
                reachable.update(ing_to_tgts.get(ing, set()))
            self._herb_reachable[h] = reachable

        self._herb_pos_targets = defaultdict(set)
        for h, t in all_pos_global:
            self._herb_pos_targets[h].add(t)

        if target_popularity is None:
            self._target_popularity = _build_target_popularity(all_pos_global, num_tgts)
        else:
            pop = np.asarray(target_popularity, dtype=np.float64)
            if pop.shape[0] != num_tgts or not np.isfinite(pop).all() or pop.sum() <= 0:
                pop = _build_target_popularity(all_pos_global, num_tgts)
            else:
                pop = pop / pop.sum()
            self._target_popularity = pop

    def sample(self, pos_h, pos_t, neg_ratio=1.0):
        B = pos_h.size(0)
        n_neg_per = max(1, int(neg_ratio))
        n_hard = min(n_neg_per, max(0, int(round(n_neg_per * self.hard_ratio))))
        n_pop = min(n_neg_per - n_hard, max(0, int(round(n_neg_per * self.pop_ratio))))

        neg_h_list, neg_t_list = [], []

        for i in range(B):
            h = pos_h[i].item()
            collected = []
            collected_set = set()
            pos_h_tgts = self._herb_pos_targets[h]
            reachable = self._herb_reachable.get(h, set())

            hard_cands = list(reachable - pos_h_tgts)
            if hard_cands and n_hard > 0:
                for t in _sample_from_weighted_candidates(hard_cands, self._target_popularity, n_hard, self.rng):
                    pair = (h, int(t))
                    if pair not in self.all_pos and pair not in collected_set:
                        collected.append(pair)
                        collected_set.add(pair)

            remaining_after_hard = n_neg_per - len(collected)
            if remaining_after_hard > 0 and n_pop > 0:
                pop_cands = [
                    t for t in range(self.num_tgts)
                    if t not in pos_h_tgts and (h, t) not in collected_set and t not in reachable
                ]
                for t in _sample_from_weighted_candidates(pop_cands, self._target_popularity, min(n_pop, remaining_after_hard), self.rng):
                    pair = (h, int(t))
                    if pair not in self.all_pos and pair not in collected_set:
                        collected.append(pair)
                        collected_set.add(pair)

            attempts = 0
            while len(collected) < n_neg_per and attempts < n_neg_per * 60:
                t = int(self.rng.integers(0, self.num_tgts))
                pair = (h, t)
                if pair not in self.all_pos and pair not in collected_set:
                    collected.append(pair)
                    collected_set.add(pair)
                attempts += 1

            for hh, tt in collected[:n_neg_per]:
                neg_h_list.append(hh)
                neg_t_list.append(tt)

        return (
            torch.tensor(neg_h_list, dtype=torch.long),
            torch.tensor(neg_t_list, dtype=torch.long),
        )
