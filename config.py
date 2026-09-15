# 📂 文件：config.py
# 项目路径与超参数配置模块（科研项目 - 性能优化版）

import os
import platform
import multiprocessing as mp

# === 服务器环境检测与优化 ===
def detect_server_environment():
    system = platform.system()
    is_server = (os.environ.get('SSH_CLIENT') or 
                 os.environ.get('SSH_TTY') or 
                 os.environ.get('TERM') == 'xterm' or
                 os.environ.get('HOSTNAME', '').lower() in ['server', 'linux', 'ubuntu'])
    
    return {
        'is_server': is_server,
        'system': system,
        'cpu_count': mp.cpu_count(),
        'has_cuda': False
    }

ENV = detect_server_environment()

# === 项目根路径 ===
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# === 文件路径配置 ===
RAW_DIR = os.path.join(BASE_DIR, 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'checkpoints')
VIS_DIR = os.path.join(BASE_DIR, 'vis')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# === 模型结构配置 (🔥 核心升级) ===
# 增加维度以提升模型容量
HGT_HIDDEN_DIM = 128            
HGT_NUM_LAYERS = 2
HGT_NUM_HEADS = 4
PATH_EMBED_DIM = 64
PATH_HIDDEN_DIM = 128

# === 训练配置 ===
EPOCHS = 100
BATCH_SIZE = 512                # 减小Batch Size，增加梯度更新频率
LR = 5e-4                       # 稍微调高学习率，配合Warmup
WEIGHT_DECAY = 1e-5             # 减小正则化，允许模型拟合更复杂的Ranking关系
EARLY_STOPPING_PATIENCE = 20
WARMUP_EPOCHS = 10

# === 损失函数配置 ===
# Ranking Loss
RANKING_MARGIN = 0.5            # 减小Margin，让模型关注更细微的分数差异

# Focal Loss
PRIMARY_LOSS = 'focal'
FOCAL_LOSS_GAMMA = 2.0
FOCAL_LOSS_ALPHA = 0.65         # 稍微降低正样本权重，防止模型“无脑”预测为正

# 其他损失
USE_WEIGHTED_BCE = True
BCE_POS_WEIGHT = 5.0
USE_DICE_LOSS = True
DICE_LOSS_SMOOTH = 1e-5
CONTRASTIVE_LAMBDA = 0.3
USE_CONTRASTIVE = False

# === 排序任务负样本优化 (🔥 核心升级) ===
# 这是提升 HR@10 的关键：大量负样本逼迫模型把正样本排在前面
RANKING_NEG_PER_POS = 30        
USE_HARD_NEGATIVE = True
POPULARITY_LOG_SMOOTH = 1.0
NEGATIVE_SAMPLING_STRATEGY = 'popularity_weighted'
USE_STRUCTURED_NEGATIVE = True
NEGATIVE_DIVERSITY_PENALTY = 0.1

# === 图构建与相似度阈值 ===
SIMILARITY_THRESHOLD_ING_ING = 0.75
SIMILARITY_THRESHOLD_TGT_TGT = 0.75
TOPK_SIM_ING = 10
REMOVE_ISOLATED_NODES = True
FILTER_ZERO_INOUT_DEGREE = True

# === 虚拟边配置 ===
USE_VIRTUAL_EDGES = True
VIRTUAL_EDGE_WEIGHT = 0.5
VIRTUAL_EDGE_TYPES = ['similar_to']
VIRTUAL_EDGE_PREFIX = 'virtual_'

# === 评估配置 ===
EVALUATION_PRESET = 'balanced'
EVALUATION_BATCH_SIZE = 256
EVALUATION_MAX_SAMPLES = None
OPTIMIZE_THRESHOLD_PER_FOLD = True
THRESHOLD_OPTIMIZATION_METRIC = 'balanced'
THRESHOLD_SEARCH_SPACE = [0.001, 0.999]
THRESHOLD_SEARCH_STEPS = 0.005

# === 药学专业指标配置 ===
ENABLE_PHARMA_METRICS = True
PHARMA_METRICS_WEIGHTS = {
    'sensitivity': 0.15, 'specificity': 0.15, 'selectivity': 0.10,
    'enrichment_factor': 0.15, 'drug_similarity': 0.10, 'target_similarity': 0.10,
    'polypharmacology': 0.15, 'off_target_risk': 0.10
}

# === 功能开关 ===
SKIP_PRECISION_RECALL_K = False
MEMORY_OPTIMIZATION = True
CLEAR_CACHE_AFTER_FOLD = True
USE_CROSS_VALIDATION = True
CROSS_VALIDATION_FOLDS = 5
USE_STRATIFIED_SPLIT = True
ENABLE_PROBABILITY_CALIBRATION = True
CALIBRATION_METHOD = 'isotonic'
CLASS_WEIGHT_ENABLED = True
NEGATIVE_SAMPLING_RATIO = 5.0

# === 监控与稳定性 ===
USE_GRADIENT_CLIPPING = True
GRADIENT_CLIP_VALUE = 1.0
USE_DROPOUT = True
DROPOUT_RATE = 0.2              # 降低Dropout，防止欠拟合
USE_BATCH_NORMALIZATION = True
ENABLE_PERFORMANCE_MONITORING = True
ENABLE_CV_EARLY_STOP = True
CV_EARLY_STOP_THRESHOLD = None
CV_TIME_LIMIT_MINUTES = None

# === 特征提取 ===
ENABLE_GCN_FEATURE_ENHANCEMENT = True
GCN_HIDDEN_DIM = 32

# === 预训练模型配置 ===
PRETRAINED_MODELS_DIR = os.path.join(BASE_DIR, 'pretrained_models')
DOWNLOAD_PRETRAINED_MODELS = False
USE_MODEL_CACHING = True

PRETRAINED_MODELS = {
    'chem_gpt': {'name': 'ChemGPT-19M', 'local_path': os.path.join(PRETRAINED_MODELS_DIR, 'ChemGPT-19M')},
    'uni_mol': {'name': 'Uni-Mol', 'local_path': os.path.join(PRETRAINED_MODELS_DIR, 'Uni-Mol')}
}

# 确保目录存在
for d in [PROCESSED_DIR, MODEL_DIR, VIS_DIR, LOG_DIR, PRETRAINED_MODELS_DIR]:
    os.makedirs(d, exist_ok=True)