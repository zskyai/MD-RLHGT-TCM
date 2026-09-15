"""
model.py
========
完整模型:  HGT 编码器  +  Ingredient-Aware H-T 解码器

消息传递:  H-I 和 I-T 边  (图结构不变)
预测:      H-T 交互  (通过成分分解注意力)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    GraphConv,
    HGTConv,
    HeteroConv,
    Linear,
    SAGEConv,
)
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Optional


class SpatialEncoder(nn.Module):
    """参考第八版本：距离矩阵 + 质心距离统计 的3D空间编码器（带padding mask）。"""

    def __init__(self, max_atoms: int = 64, hidden_dim: int = 128, dist_hidden_dim: int = 64, centroid_hidden_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.max_atoms = int(max_atoms)
        self.hidden_dim = int(hidden_dim)
        self.dist_hidden_dim = int(dist_hidden_dim)
        self.centroid_hidden_dim = int(centroid_hidden_dim)

        self.dist_mlp = nn.Sequential(
            nn.Linear(self.max_atoms, self.dist_hidden_dim),
            nn.LayerNorm(self.dist_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.centroid_mlp = nn.Sequential(
            nn.Linear(1, self.centroid_hidden_dim),
            nn.LayerNorm(self.centroid_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.structure_proj = nn.Sequential(
            nn.Linear(self.max_atoms * (self.dist_hidden_dim + self.centroid_hidden_dim), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if coords is None:
            raise ValueError("SpatialEncoder requires ingredient coordinates")
        coords = coords.to(dtype=torch.float32)
        if coords.dim() != 3:
            raise ValueError(f"Expected coords shape [N, A, 3], got {tuple(coords.shape)}")
        if coords.size(1) != self.max_atoms:
            raise ValueError(f"Expected atom dimension {self.max_atoms}, got {coords.size(1)}")

        atom_mask = coords.abs().sum(dim=-1) > 0  # [N, A]
        mask_f = atom_mask.float()
        pair_mask = atom_mask.unsqueeze(1) & atom_mask.unsqueeze(2)  # [N, A, A]

        # pairwise distance matrix
        dists = torch.cdist(coords, coords)
        dists = dists * pair_mask.float()
        dist_encoded = self.dist_mlp(dists)
        dist_encoded = dist_encoded * mask_f.unsqueeze(-1)

        # distances to centroid (masked)
        valid_counts = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        centroid = (coords * mask_f.unsqueeze(-1)).sum(dim=1, keepdim=True) / valid_counts.unsqueeze(-1)
        centroid_dists = torch.norm(coords - centroid, dim=-1) * mask_f
        centroid_encoded = self.centroid_mlp(centroid_dists.unsqueeze(-1))
        centroid_encoded = centroid_encoded * mask_f.unsqueeze(-1)

        encoded = torch.cat([dist_encoded, centroid_encoded], dim=-1)
        encoded = encoded.reshape(encoded.size(0), -1)
        spatial = self.structure_proj(encoded)

        all_invalid = ~atom_mask.any(dim=1)
        if all_invalid.any():
            spatial[all_invalid] = 0.0
        return spatial


class FusionGate(nn.Module):
    """参考第八版本：3D-语义自适应门控融合，初始偏向语义特征。"""

    def __init__(self, hidden_dim: int, dropout: float = 0.1, semantic_bias: float = 1.5):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        final_linear = self.gate[4]
        nn.init.zeros_(final_linear.weight)
        nn.init.constant_(final_linear.bias, float(semantic_bias))

    def forward(self, semantic: torch.Tensor, spatial: torch.Tensor) -> torch.Tensor:
        alpha = self.gate(torch.cat([semantic, spatial], dim=-1))
        return alpha * semantic + (1.0 - alpha) * spatial


# ================================================================
#  HGT 编码器
# ================================================================

class HGTEncoder(nn.Module):
    """
    异构图变换器编码器 (Heterogeneous Graph Transformer).

    在 H-I + I-T 边上做消息传递, 输出每个节点的 embedding.
    """

    def __init__(
        self,
        node_types:   List[str],
        edge_types:   List[Tuple[str, str, str]],
        in_dim_dict:  Dict[str, int],
        hidden_dim:   int  = 128,
        num_layers:   int  = 3,
        num_heads:    int  = 4,
        dropout:      float = 0.1,
        ingredient_spatial_dim: int = 0,
    ):
        super().__init__()
        self.node_types = node_types
        self.hidden_dim = hidden_dim

        # 投影到统一维度
        self.input_projs = nn.ModuleDict()
        for ntype in node_types:
            self.input_projs[ntype] = Linear(in_dim_dict[ntype], hidden_dim)

        # HGT 层
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                HGTConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    metadata=(node_types, edge_types),
                    heads=num_heads,
                )
            )

        self.norms = nn.ModuleDict()
        for ntype in node_types:
            self.norms[ntype] = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """
        Args:
            data: HeteroData (仅含 H-I 和 I-T 边)

        Returns:
            dict: {node_type: embedding [N, hidden_dim]}
        """
        # 投影
        x_dict = {}
        for ntype in self.node_types:
            x_dict[ntype] = self.input_projs[ntype](data[ntype].x)
            x_dict[ntype] = self.dropout(F.gelu(x_dict[ntype]))

        # HGT 消息传递
        for layer in self.layers:
            x_dict_new = layer(x_dict, data.edge_index_dict)
            # 残差 + LayerNorm
            for ntype in self.node_types:
                if ntype in x_dict_new and x_dict_new[ntype] is not None:
                    x_dict[ntype] = self.norms[ntype](
                        x_dict[ntype] + self.dropout(x_dict_new[ntype])
                    )

        return x_dict


def _build_relation_conv(
    backbone: str,
    out_dim: int,
    heads: int,
    dropout: float,
):
    name = str(backbone).lower()
    if name == "sage":
        return SAGEConv((-1, -1), out_dim)
    if name == "gat":
        return GATConv(
            (-1, -1),
            out_dim,
            heads=heads,
            concat=False,
            dropout=dropout,
            add_self_loops=False,
        )
    if name in {"graph", "graphconv", "gcn"}:
        return GraphConv((-1, -1), out_dim, aggr="mean")
    raise ValueError(f"Unsupported encoder backbone: {backbone}")


class RelationalBackboneEncoder(nn.Module):
    """
    在相同异构图协议下切换不同消息传递底座，
    便于做 backbone comparison，而不改动下游解码器。
    """

    def __init__(
        self,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        in_dim_dict: Dict[str, int],
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        backbone: str = "sage",
    ):
        super().__init__()
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)
        self.hidden_dim = int(hidden_dim)
        self.backbone = str(backbone).lower()

        self.input_projs = nn.ModuleDict()
        self.norms = nn.ModuleDict()
        for ntype in self.node_types:
            self.input_projs[ntype] = Linear(in_dim_dict[ntype], hidden_dim)
            self.norms[ntype] = nn.LayerNorm(hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(int(num_layers)):
            convs = {}
            for edge_type in self.edge_types:
                convs[edge_type] = _build_relation_conv(
                    backbone=self.backbone,
                    out_dim=hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                )
            self.layers.append(HeteroConv(convs, aggr="sum"))

        self.dropout = nn.Dropout(dropout)

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        x_dict = {}
        for ntype in self.node_types:
            x = self.input_projs[ntype](data[ntype].x)
            x_dict[ntype] = self.dropout(F.gelu(x))

        for layer in self.layers:
            x_dict_new = layer(x_dict, data.edge_index_dict)
            for ntype in self.node_types:
                updated = x_dict_new.get(ntype, None)
                if updated is None:
                    continue
                x_dict[ntype] = self.norms[ntype](
                    x_dict[ntype] + self.dropout(updated)
                )

        return x_dict


# ================================================================
#  成分感知 H-T 解码器  (Batched)
# ================================================================

class IngredientAwareHTDecoder(nn.Module):
    """
    成分感知的 Herb-Target 交互解码器.

    核心公式:
        score(h, t) = Σ_{i ∈ I(h)}  α(h,i,t) · φ(i, t)  +  λ · ψ(h, t)

    其中:
        I(h)       : herb h 包含的成分集合
        α(h,i,t)   : 三方注意力 — 成分 i 在 h→t 中的重要性
        φ(i,t)     : 成分 i 与 target t 的结合兼容性
        ψ(h,t)     : 全局 herb-target 直接兼容性 (偏置项)
        λ           : 可学习的平衡系数
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads:  int   = 4,
        dropout:    float = 0.1,
        attention_activation: str = 'softmax',
        attention_normalization: str = 'length_mean',
        freq_bias_beta: float = 0.0,
        use_ingredient_path: bool = True,
        use_tri_attention: bool = True,
        use_global_path: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads  = num_heads
        self.attention_activation = str(attention_activation).lower()
        self.attention_normalization = str(attention_normalization).lower()
        self.freq_bias_beta = float(freq_bias_beta)
        self.use_ingredient_path = bool(use_ingredient_path)
        self.use_tri_attention = bool(use_tri_attention)
        self.use_global_path = bool(use_global_path)
        self.register_buffer('ingredient_log_freq', torch.zeros(1), persistent=False)
        self.register_buffer('target_log_freq', torch.zeros(1), persistent=False)

        # ─── φ: 成分-靶点兼容性 ───
        self.compat_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # ─── α: 三方注意力  (herb, ingredient, target) ───
        self.attn_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_heads),
        )

        # ─── 类型分布校准（降低ingredient/target隐藏偏置） ───
        self.ingredient_calibration = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.target_calibration = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # ─── ψ: 全局 H-T 偏置 ───
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.lambda_global = nn.Parameter(torch.tensor(0.3))

    def set_frequency_bias(self, ingredient_log_freq: torch.Tensor, target_log_freq: torch.Tensor):
        if ingredient_log_freq is None or target_log_freq is None:
            return
        device = self.lambda_global.device
        self.ingredient_log_freq = ingredient_log_freq.detach().to(device=device, dtype=torch.float32)
        self.target_log_freq = target_log_freq.detach().to(device=device, dtype=torch.float32)

    def forward(
        self,
        herb_emb:       torch.Tensor,   # [B, d]
        target_emb:     torch.Tensor,   # [B, d]
        ingredient_emb: torch.Tensor,   # [N_all_ing, d]
        herb_ing_padded: torch.Tensor,  # [B, max_K]
        herb_ing_mask:   torch.Tensor,  # [B, max_K]
        target_ids:      Optional[torch.Tensor] = None,
        return_attention: bool = False,
        return_aux: bool = False,
    ):
        """
        Args:
            herb_emb:        [B, d]     batch 中 herb 的 embedding
            target_emb:      [B, d]     batch 中 target 的 embedding
            ingredient_emb:  [N, d]     所有 ingredient 的 embedding
            herb_ing_padded: [B, max_K] 每个 herb 的成分索引 (padded)
            herb_ing_mask:   [B, max_K] True = 有效成分位置
            return_attention: 是否返回注意力权重 (可解释性用)

        Returns:
            scores: [B]
            attn_weights (optional): [B, max_K]
        """
        B, max_K = herb_ing_padded.shape
        d = self.hidden_dim
        device = herb_emb.device

        # 1. 取出每个 herb 的成分 embedding  [B, max_K, d]
        safe_idx = herb_ing_padded.clamp(min=0)  # padding位用0号ingredient代替(后面mask掉)
        ing_embs = ingredient_emb[safe_idx]       # [B, max_K, d]
        ing_embs = self.ingredient_calibration(ing_embs)

        # 2. 扩展 herb / target  → [B, max_K, d]
        h_exp = herb_emb.unsqueeze(1).expand(-1, max_K, -1)
        t_exp = target_emb.unsqueeze(1).expand(-1, max_K, -1)
        t_exp_calibrated = self.target_calibration(t_exp)

        # ─── φ(i, t): 成分-靶点兼容性 [B, max_K] ───
        compat_input = torch.cat([ing_embs, t_exp_calibrated], dim=-1)      # [B, max_K, 2d]
        compat_scores = self.compat_mlp(compat_input).squeeze(-1) # [B, max_K]

        # ─── α(h, i, t): 三方注意力 [B, max_K] ───
        attn_input  = torch.cat([h_exp, ing_embs, t_exp_calibrated], dim=-1)  # [B, max_K, 3d]
        attn_logits = self.attn_mlp(attn_input)                                 # [B, max_K, heads]

        if self.freq_bias_beta > 0 and self.ingredient_log_freq.numel() > 1 and self.target_log_freq.numel() > 1 and target_ids is not None:
            safe_ing_idx = herb_ing_padded.clamp(min=0)
            ing_freq_bias = self.ingredient_log_freq[safe_ing_idx].unsqueeze(-1)
            safe_tgt_idx = target_ids.clamp(min=0)
            tgt_freq_bias = self.target_log_freq[safe_tgt_idx].view(B, 1, 1)
            attn_logits = attn_logits - self.freq_bias_beta * (ing_freq_bias + tgt_freq_bias)

        # mask 无效位
        inv_mask = ~herb_ing_mask                                               # [B, max_K]

        if self.attention_activation == 'sigmoid':
            attn_weights = torch.sigmoid(attn_logits)
            attn_weights = attn_weights * herb_ing_mask.unsqueeze(-1).float()
        else:
            attn_logits = attn_logits.masked_fill(
                inv_mask.unsqueeze(-1), float('-inf')
            )
            attn_weights = F.softmax(attn_logits, dim=1)

        attn_weights = attn_weights.mean(dim=-1)                                 # [B, max_K]
        if not self.use_tri_attention:
            valid_counts = herb_ing_mask.sum(dim=1, keepdim=True).clamp_min(1).float()
            attn_weights = herb_ing_mask.float() / valid_counts

        # 处理全无效行
        all_invalid = ~herb_ing_mask.any(dim=1)  # [B]
        if all_invalid.any():
            attn_weights[all_invalid] = 0.0

        # ─── 加权聚合:  Σ α · φ  →  [B] ───
        ingredient_score = (attn_weights * compat_scores).sum(dim=1)
        ingredient_context = (attn_weights.unsqueeze(-1) * ing_embs).sum(dim=1)
        if self.attention_activation == 'sigmoid':
            if self.attention_normalization == 'alpha_sum':
                denom = attn_weights.sum(dim=1).clamp_min(1e-6)
                ingredient_score = ingredient_score / denom
                ingredient_context = ingredient_context / denom.unsqueeze(-1)
            else:
                valid_counts = herb_ing_mask.sum(dim=1).clamp_min(1).float()
                ingredient_score = ingredient_score / valid_counts
                ingredient_context = ingredient_context / valid_counts.unsqueeze(-1)
        if not self.use_ingredient_path:
            ingredient_score = herb_emb.new_zeros(B)
            ingredient_context = herb_emb.new_zeros(B, d)

        # ─── ψ(h, t): 全局偏置 [B] ───
        global_input = torch.cat([herb_emb, target_emb], dim=-1)
        global_score = self.global_mlp(global_input).squeeze(-1)
        if not self.use_global_path:
            global_score = herb_emb.new_zeros(B)

        # ─── 最终分数 ───
        if self.use_ingredient_path and self.use_global_path:
            scores = ingredient_score + self.lambda_global * global_score
        elif self.use_ingredient_path:
            scores = ingredient_score
        elif self.use_global_path:
            scores = global_score
        else:
            scores = herb_emb.new_zeros(B)

        aux = {
            'herb_emb': herb_emb,
            'target_emb': target_emb,
            'ingredient_context': ingredient_context,
            'ingredient_score': ingredient_score.detach(),
            'global_score': global_score.detach(),
            'compat_scores': compat_scores.detach(),
            'attn_logits': attn_logits.detach(),
            'lambda_global': self.lambda_global.detach(),
        }

        if return_attention and return_aux:
            return scores, attn_weights.detach(), aux
        if return_attention:
            return scores, attn_weights.detach()
        if return_aux:
            return scores, aux
        return scores


# ================================================================
#  完整模型
# ================================================================

class IngredientAwareHTModel(nn.Module):
    """
    端到端模型 = HGT 编码器 + 成分感知 H-T 解码器

    图结构:   Herb --H-I--> Ingredient --I-T--> Target  (消息传递)
    预测目标: Herb --> Target             (成分分解注意力)
    """

    def __init__(
        self,
        node_types:  List[str],
        edge_types:  List[Tuple[str, str, str]],
        in_dim_dict: Dict[str, int],
        hidden_dim:  int   = 128,
        num_layers:  int   = 3,
        num_heads:   int   = 4,
        dropout:     float = 0.1,
        attention_activation: str = 'softmax',
        attention_normalization: str = 'length_mean',
        freq_bias_beta: float = 0.0,
        use_spatial_encoder: bool = True,
        spatial_dim: int = 64,
        spatial_dropout: float = 0.1,
        spatial_max_atoms: int = 64,
        spatial_dist_hidden_dim: int = 64,
        spatial_centroid_hidden_dim: int = 16,
        spatial_semantic_bias: float = 1.5,
        encoder_backbone: str = 'hgt',
        decoder_use_ingredient_path: bool = True,
        decoder_use_tri_attention: bool = True,
        decoder_use_global_path: bool = True,
    ):
        super().__init__()
        self.use_spatial_encoder = bool(use_spatial_encoder)
        self.spatial_dim = int(spatial_dim) if self.use_spatial_encoder else 0
        self.spatial_encoder = (
            SpatialEncoder(
                max_atoms=spatial_max_atoms,
                hidden_dim=self.spatial_dim,
                dist_hidden_dim=spatial_dist_hidden_dim,
                centroid_hidden_dim=spatial_centroid_hidden_dim,
                dropout=spatial_dropout,
            )
            if self.use_spatial_encoder else None
        )
        self.fusion_gate = (
            FusionGate(
                hidden_dim=self.spatial_dim,
                dropout=spatial_dropout,
                semantic_bias=spatial_semantic_bias,
            )
            if self.use_spatial_encoder else None
        )

        fused_in_dim_dict = dict(in_dim_dict)
        if self.use_spatial_encoder:
            ingredient_dim = int(in_dim_dict['ingredient'])
            if ingredient_dim != self.spatial_dim:
                self.ingredient_semantic_proj = nn.Sequential(
                    nn.Linear(ingredient_dim, self.spatial_dim),
                    nn.LayerNorm(self.spatial_dim),
                    nn.GELU(),
                    nn.Dropout(spatial_dropout),
                )
            else:
                self.ingredient_semantic_proj = nn.Identity()
            fused_in_dim_dict['ingredient'] = self.spatial_dim
        else:
            self.ingredient_semantic_proj = nn.Identity()

        self.encoder_backbone = str(encoder_backbone).lower()
        if self.encoder_backbone == 'hgt':
            self.encoder = HGTEncoder(
                node_types=node_types,
                edge_types=edge_types,
                in_dim_dict=fused_in_dim_dict,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
            )
        else:
            self.encoder = RelationalBackboneEncoder(
                node_types=node_types,
                edge_types=edge_types,
                in_dim_dict=fused_in_dim_dict,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                backbone=self.encoder_backbone,
            )

        self.decoder = IngredientAwareHTDecoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            attention_activation=attention_activation,
            attention_normalization=attention_normalization,
            freq_bias_beta=freq_bias_beta,
            use_ingredient_path=decoder_use_ingredient_path,
            use_tri_attention=decoder_use_tri_attention,
            use_global_path=decoder_use_global_path,
        )
        self.herb_context_projector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.target_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.hidden_dim = hidden_dim

    def encode(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        if self.use_spatial_encoder:
            semantic_ing = self.ingredient_semantic_proj(data['ingredient'].x)
            coords = getattr(data['ingredient'], 'pos', None)
            if coords is not None:
                spatial_ing = self.spatial_encoder(coords.to(data['ingredient'].x.device))
            else:
                spatial_ing = torch.zeros_like(semantic_ing)
            fused_ing = self.fusion_gate(semantic_ing, spatial_ing)

            fused_data = data.clone()
            fused_data['ingredient'].x = fused_ing
            return self.encoder(fused_data)
        return self.encoder(data)

    def decode(
        self,
        x_dict:          Dict[str, torch.Tensor],
        herb_ids:        torch.Tensor,    # [B]
        target_ids:      torch.Tensor,    # [B]
        herb_ing_padded: torch.Tensor,    # [B, max_K]
        herb_ing_mask:   torch.Tensor,    # [B, max_K]
        return_attention: bool = False,
        return_aux: bool = False,
    ):
        herb_emb   = x_dict['herb'][herb_ids]          # [B, d]
        target_emb = x_dict['target'][target_ids]       # [B, d]
        ing_emb    = x_dict['ingredient']               # [N_ing, d]

        out = self.decoder(
            herb_emb, target_emb, ing_emb,
            herb_ing_padded, herb_ing_mask,
            target_ids=target_ids,
            return_attention=return_attention,
            return_aux=return_aux,
        )
        return out

    def forward(
        self,
        data:            HeteroData,
        herb_ids:        torch.Tensor,
        target_ids:      torch.Tensor,
        herb_ing_padded: torch.Tensor,
        herb_ing_mask:   torch.Tensor,
        return_attention: bool = False,
        return_aux: bool = False,
    ):
        x_dict = self.encode(data)
        return self.decode(
            x_dict, herb_ids, target_ids,
            herb_ing_padded, herb_ing_mask,
            return_attention=return_attention,
            return_aux=return_aux,
        )


class VanillaHGTBaselineModel(nn.Module):
    """
    独立的 vanilla HGT baseline:
    仅保留 HGT 编码器和 Herb-Target 直接匹配打分。
    """

    def __init__(
        self,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        in_dim_dict: Dict[str, int],
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = HGTEncoder(
            node_types=node_types,
            edge_types=edge_types,
            in_dim_dict=in_dim_dict,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.herb_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.target_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.scorer = nn.Bilinear(hidden_dim, hidden_dim, 1)

    def forward(
        self,
        data: HeteroData,
        herb_ids: torch.Tensor,
        target_ids: torch.Tensor,
        herb_ing_padded: Optional[torch.Tensor] = None,
        herb_ing_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        return_aux: bool = False,
    ):
        x_dict = self.encoder(data)
        herb_emb = self.herb_head(x_dict['herb'][herb_ids])
        target_emb = self.target_head(x_dict['target'][target_ids])
        scores = self.scorer(herb_emb, target_emb).squeeze(-1)

        if return_attention and return_aux:
            return scores, None, {}
        if return_attention:
            return scores, None
        if return_aux:
            return scores, {}
        return scores

