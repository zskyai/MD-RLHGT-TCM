"""
explain.py
==========
成分重要性解释工具

对于预测 "Herb_h → Target_t":
  → 输出每个成分的注意力权重 α(h, i, t)
  → 排名显示哪些成分对该预测贡献最大
  → 支持批量分析 + 可视化
"""

import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from torch_geometric.data import HeteroData

from cold_start_split import build_herb_ing_padded, HTSplitBundle
from model import IngredientAwareHTModel


# ================================================================
#  单条预测解释
# ================================================================

@torch.no_grad()
def explain_single(
    model:        IngredientAwareHTModel,
    msg_graph:    HeteroData,
    herb_id:      int,
    target_id:    int,
    herb_to_ings: Dict[int, List[int]],
    ing_names:    Optional[Dict[int, str]] = None,
    target_names: Optional[Dict[int, str]] = None,
    herb_names:   Optional[Dict[int, str]] = None,
    top_k:        int = 10,
    device:       torch.device = torch.device('cpu'),
) -> dict:
    """
    解释单条 Herb → Target 预测.

    Returns:
        dict with keys:
          'score':        预测分数 (logit)
          'probability':  概率 (sigmoid)
          'ingredients':  list of {id, name, attention, compatibility_rank}
    """
    model.eval()

    h_tensor = torch.tensor([herb_id], dtype=torch.long)
    t_tensor = torch.tensor([target_id], dtype=torch.long)

    padded, mask = build_herb_ing_padded(h_tensor, herb_to_ings)
    padded = padded.to(device)
    mask   = mask.to(device)
    h_tensor = h_tensor.to(device)
    t_tensor = t_tensor.to(device)

    scores, attn_weights = model(
        msg_graph, h_tensor, t_tensor, padded, mask,
        return_attention=True,
    )

    score = scores[0].item()
    prob  = torch.sigmoid(scores[0]).item()
    attn  = attn_weights[0].cpu()     # [max_K]
    valid_mask = mask[0].cpu()         # [max_K]

    # 提取有效成分的注意力
    ing_list = herb_to_ings.get(herb_id, [])
    results = []
    for idx, ing_id in enumerate(ing_list):
        if idx < attn.size(0) and valid_mask[idx]:
            name = (ing_names or {}).get(ing_id, f"Ingredient_{ing_id}")
            results.append({
                'ingredient_id':   ing_id,
                'ingredient_name': name,
                'attention':       attn[idx].item(),
            })

    # 按注意力排序
    results.sort(key=lambda x: x['attention'], reverse=True)

    # 归一化为百分比
    total_attn = sum(r['attention'] for r in results)
    for r in results:
        r['contribution_pct'] = (r['attention'] / total_attn * 100
                                  if total_attn > 0 else 0)

    herb_name   = (herb_names or {}).get(herb_id, f"Herb_{herb_id}")
    target_name = (target_names or {}).get(target_id, f"Target_{target_id}")

    return {
        'herb_id':     herb_id,
        'herb_name':   herb_name,
        'target_id':   target_id,
        'target_name': target_name,
        'score':       score,
        'probability': prob,
        'ingredients': results[:top_k],
    }


def print_explanation(result: dict):
    """打印单条解释"""
    print(f"\n{'='*65}")
    print(f"  预测: {result['herb_name']} → {result['target_name']}")
    print(f"  得分: {result['score']:.4f}   概率: {result['probability']:.4f}")
    print(f"{'='*65}")
    print(f"  {'排名':<5} {'成分':<30} {'注意力':<10} {'贡献':<8} 可视化")
    print(f"  {'-'*62}")

    for rank, ing in enumerate(result['ingredients'], 1):
        bar = '█' * int(ing['contribution_pct'] / 2)
        print(f"  {rank:<5} {ing['ingredient_name']:<30} "
              f"{ing['attention']:<10.4f} {ing['contribution_pct']:>5.1f}%  {bar}")

    if not result['ingredients']:
        print("  (该草药无已知成分信息)")


# ================================================================
#  批量解释: 某个 Herb 的 Top-N 靶点
# ================================================================

@torch.no_grad()
def explain_herb_top_targets(
    model:         IngredientAwareHTModel,
    msg_graph:     HeteroData,
    herb_id:       int,
    herb_to_ings:  Dict[int, List[int]],
    num_targets:   int,
    top_n:         int = 20,
    batch_size:    int = 512,
    device:        torch.device = torch.device('cpu'),
    herb_names:    Optional[Dict[int, str]] = None,
    target_names:  Optional[Dict[int, str]] = None,
    ing_names:     Optional[Dict[int, str]] = None,
) -> pd.DataFrame:
    """
    对指定 herb, 扫描所有 target, 返回预测得分最高的 Top-N.
    每个 Top-N 靶点附带成分重要性解释.
    """
    model.eval()

    all_targets = torch.arange(num_targets, dtype=torch.long)
    all_herbs   = torch.full((num_targets,), herb_id, dtype=torch.long)

    # 分 batch 打分
    all_scores = []
    for start in range(0, num_targets, batch_size):
        end = min(start + batch_size, num_targets)
        bh = all_herbs[start:end].to(device)
        bt = all_targets[start:end].to(device)
        padded, mask = build_herb_ing_padded(bh, herb_to_ings)
        s = model(msg_graph, bh, bt, padded.to(device), mask.to(device))
        all_scores.append(s.cpu())

    all_scores = torch.cat(all_scores)       # [num_targets]
    probs      = torch.sigmoid(all_scores)

    # Top-N
    topk_vals, topk_idx = probs.topk(top_n)

    herb_name = (herb_names or {}).get(herb_id, f"Herb_{herb_id}")
    rows = []

    for rank, (tgt_idx, prob_val) in enumerate(
        zip(topk_idx.tolist(), topk_vals.tolist()), 1
    ):
        # 获取该靶点的成分解释
        result = explain_single(
            model, msg_graph, herb_id, tgt_idx,
            herb_to_ings, ing_names, target_names, herb_names,
            top_k=3, device=device,
        )

        tgt_name = (target_names or {}).get(tgt_idx, f"Target_{tgt_idx}")
        top_ings = "; ".join(
            f"{ing['ingredient_name']}({ing['contribution_pct']:.1f}%)"
            for ing in result['ingredients'][:3]
        )

        rows.append({
            'rank':           rank,
            'herb':           herb_name,
            'target':         tgt_name,
            'target_id':      tgt_idx,
            'probability':    prob_val,
            'top_ingredients': top_ings,
        })

    df = pd.DataFrame(rows)
    return df


# ================================================================
#  全局成分重要性统计
# ================================================================

@torch.no_grad()
def global_ingredient_importance(
    model:        IngredientAwareHTModel,
    msg_graph:    HeteroData,
    pos_h:        torch.Tensor,
    pos_t:        torch.Tensor,
    herb_to_ings: Dict[int, List[int]],
    batch_size:   int = 512,
    device:       torch.device = torch.device('cpu'),
    ing_names:    Optional[Dict[int, str]] = None,
) -> pd.DataFrame:
    """
    在所有正样本 H-T 对上, 统计每个成分的平均注意力权重.
    → 识别"全局最重要的成分"
    """
    model.eval()

    ing_attn_sum   = defaultdict(float)
    ing_attn_count = defaultdict(int)

    N = pos_h.size(0)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        bh = pos_h[start:end]
        bt = pos_t[start:end].to(device)

        padded, mask = build_herb_ing_padded(bh, herb_to_ings)
        bh_dev = bh.to(device)
        padded = padded.to(device)
        mask   = mask.to(device)

        _, attn_weights = model(
            msg_graph, bh_dev, bt, padded, mask,
            return_attention=True,
        )
        # attn_weights: [B, max_K]

        for i in range(bh.size(0)):
            h = bh[i].item()
            ing_list = herb_to_ings.get(h, [])
            for j, ing_id in enumerate(ing_list):
                if j < attn_weights.size(1) and mask[i, j].item():
                    w = attn_weights[i, j].item()
                    ing_attn_sum[ing_id]   += w
                    ing_attn_count[ing_id] += 1

    rows = []
    for ing_id in sorted(ing_attn_sum.keys()):
        name = (ing_names or {}).get(ing_id, f"Ingredient_{ing_id}")
        avg  = ing_attn_sum[ing_id] / max(1, ing_attn_count[ing_id])
        rows.append({
            'ingredient_id':   ing_id,
            'ingredient_name': name,
            'avg_attention':   avg,
            'appear_count':    ing_attn_count[ing_id],
            'total_attention': ing_attn_sum[ing_id],
        })

    df = pd.DataFrame(rows)
    df = df.sort_values('avg_attention', ascending=False).reset_index(drop=True)
    return df


# ================================================================
#  Case Study: 冷启动 Herb 的预测解释
# ================================================================

def case_study_cold_start(
    model:        IngredientAwareHTModel,
    msg_graph:    HeteroData,
    bundle:       HTSplitBundle,
    n_herbs:      int = 5,
    top_targets:  int = 10,
    device:       torch.device = torch.device('cpu'),
    herb_names:   Optional[Dict[int, str]] = None,
    target_names: Optional[Dict[int, str]] = None,
    ing_names:    Optional[Dict[int, str]] = None,
):
    """
    对测试集中的冷启动 herb, 展示预测结果和成分解释.
    """
    test_herbs = sorted(bundle.test_herbs)[:n_herbs]
    num_targets = msg_graph['target'].num_nodes

    print(f"\n{'#'*70}")
    print(f"#  冷启动 Case Study  ({len(test_herbs)} herbs)")
    print(f"{'#'*70}")

    for herb_id in test_herbs:
        herb_name = (herb_names or {}).get(herb_id, f"Herb_{herb_id}")
        print(f"\n{'='*65}")
        print(f"  冷启动草药: {herb_name} (ID={herb_id})")
        ings = bundle.herb_to_ings.get(herb_id, [])
        print(f"  已知成分数: {len(ings)}")

        if ing_names:
            print(f"  成分列表: {', '.join(ing_names.get(i, str(i)) for i in ings[:8])}"
                  f"{'...' if len(ings) > 8 else ''}")

        # 获取真实靶点
        true_targets = set()
        for i in range(bundle.test_pos_h.size(0)):
            if bundle.test_pos_h[i].item() == herb_id:
                true_targets.add(bundle.test_pos_t[i].item())

        print(f"  真实靶点数: {len(true_targets)}")

        # Top-N 预测
        df = explain_herb_top_targets(
            model, msg_graph, herb_id,
            bundle.herb_to_ings, num_targets,
            top_n=top_targets, device=device,
            herb_names=herb_names,
            target_names=target_names,
            ing_names=ing_names,
        )

        # 标记命中
        df['hit'] = df['target_id'].apply(lambda x: '✅' if x in true_targets else '  ')

        print(f"\n  Top-{top_targets} 预测靶点:")
        print(f"  {'Rank':<5} {'靶点':<20} {'概率':<10} {'命中':<5} 关键成分")
        print(f"  {'-'*70}")
        for _, row in df.iterrows():
            print(f"  {row['rank']:<5} {row['target']:<20} "
                  f"{row['probability']:<10.4f} {row['hit']:<5} "
                  f"{row['top_ingredients']}")

        hit_count = df['target_id'].apply(lambda x: x in true_targets).sum()
        print(f"\n  命中率: {hit_count}/{top_targets} = {hit_count/top_targets*100:.1f}%")