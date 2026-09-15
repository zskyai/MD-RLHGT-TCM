# -*- coding: utf-8 -*-
import os
import time

# 🔥 核心优化 1：彻底禁用 Tokenizers 并行，防止死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import pandas as pd
import numpy as np
import warnings
import gc
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from concurrent.futures import ProcessPoolExecutor
from rdkit import Chem
from rdkit.Chem import AllChem, SaltRemover
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# 忽略非关键警告
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

# ==========================================
# 1. 基础配置
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed')
RAW_DIR = os.path.join(BASE_DIR, 'raw')
os.makedirs(PROCESSED_DIR, exist_ok=True)

FILE_MAPPING = {
    'herb': ["herb.xlsx - Sheet1.csv", "herb.csv", "herb.xlsx"],
    'ingredient': ["ingredients.xlsx - Sheet1.csv", "ingredients.csv", "ingredients.xlsx"],
    'target': ["target.xlsx - Sheet1.csv", "target.csv", "target.xlsx"]
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = max(1, os.cpu_count() - 2)

# =========================================================
# 🔥 绝对路径配置 (已集成 ChemGPT)
# =========================================================
PROJECT_ROOT = "/home/zhangxukun/HIT"
CHEM_PATH = os.path.join(PROJECT_ROOT, "chemberta_model")
# ✅ 新增 ChemGPT 路径
CHEMGPT_PATH = os.path.join(PROJECT_ROOT, "ChemGPT-19M") 
PROT_PATH = os.path.join(PROJECT_ROOT, "protbert_model")
QWEN_PATH = os.path.join(PROJECT_ROOT, "Qwen3-Embedding-4B")

# --- 性能超参数 ---
BATCH_SIZE_MOL = 256  
BATCH_SIZE_PROT = 16  
BATCH_SIZE_LLM = 4    

MAX_ATOMS = 64
CONFORMER_COUNT = 10

print(f"🚀 运行设备: {DEVICE}")
print(f"⚡ 优化模式: 智能长度排序 + 双塔分子编码(BERT+GPT) + 最优能级3D筛选")

# ==========================================
# 2. 数据集类与工具函数
# ==========================================
class TextDataset(Dataset):
    def __init__(self, text_list):
        self.texts = [str(t) for t in text_list]
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx]

def worker_clean_smiles(smiles):
    try:
        if not isinstance(smiles, str) or len(smiles) < 1: return "C"
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return "C"
        remover = SaltRemover.SaltRemover()
        mol = remover.StripMol(mol)
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    except:
        return "C"

def worker_morgan(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return np.zeros(1024, dtype=np.float32)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        arr = np.zeros((1,), dtype=np.float32)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except:
        return np.zeros(1024, dtype=np.float32)

def worker_3d_coords(smiles):
    """3D 构象生成与能量优化"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: raise ValueError
        mol = Chem.AddHs(mol)
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=CONFORMER_COUNT, params=AllChem.ETKDG())
        if not cids: cids = AllChem.EmbedMultipleConfs(mol, numConfs=1, useRandomCoords=True)
        if not cids: raise ValueError
        
        min_energy = float('inf')
        best_id = cids[0]
        for cid in cids:
            try:
                if AllChem.MMFFOptimizeMolecule(mol, confId=cid) == 0:
                    props = AllChem.MMFFGetMoleculeProperties(mol)
                    ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
                    energy = ff.CalcEnergy()
                    if energy < min_energy:
                        min_energy = energy
                        best_id = cid
            except: continue

        conf = mol.GetConformer(best_id)
        pos = []
        num_atoms = min(mol.GetNumAtoms(), MAX_ATOMS)
        for i in range(num_atoms):
            p = conf.GetAtomPosition(i)
            pos.append([p.x, p.y, p.z])
        if num_atoms < MAX_ATOMS:
            pos.extend([[0.0, 0.0, 0.0]] * (MAX_ATOMS - num_atoms))
        return np.array(pos, dtype=np.float32)
    except:
        return np.zeros((MAX_ATOMS, 3), dtype=np.float32)

# ==========================================
# 4. 智能推理引擎
# ==========================================
def run_inference_smart(model, tokenizer, data_list, batch_size, desc, model_type="encoder", max_len=512):
    # 1. 长度排序
    indexed_data = []
    for i, text in enumerate(data_list):
        t_str = str(text)
        indexed_data.append((i, len(t_str), t_str))
    indexed_data.sort(key=lambda x: x[1], reverse=True)
    sorted_texts = [x[2] for x in indexed_data]
    original_indices = [x[0] for x in indexed_data]
    
    # 2. 推理
    dataset = TextDataset(sorted_texts)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    sorted_embeddings = []
    model.eval()
    
    print(f"🚀 [{desc}] 启动智能推理 (Smart Batching)...")
    with torch.no_grad():
        for batch_text in tqdm(loader, desc=desc, unit="batch"):
            try:
                inputs = tokenizer(batch_text, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(DEVICE)
                with torch.cuda.amp.autocast():
                    if model_type == "encoder":
                        outputs = model(**inputs)
                        mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                        emb = torch.sum(outputs.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
                    else: # Decoder (GPT)
                        outputs = model(**inputs, output_hidden_states=True)
                        last_hidden = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs.hidden_states[-1]
                        mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
                        emb = torch.sum(last_hidden * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
                sorted_embeddings.append(emb.cpu())
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("💥 OOM! 自动降级 Batch Size 重试...")
                    torch.cuda.empty_cache()
                    dim = model.config.hidden_size
                    sorted_embeddings.append(torch.zeros(len(batch_text), dim))
                else:
                    print(f"⚠️ 错误: {e}")
                    
    # 3. 恢复顺序
    all_sorted = torch.cat(sorted_embeddings, dim=0)
    final_embeddings = torch.zeros_like(all_sorted)
    final_embeddings[original_indices] = all_sorted
    return final_embeddings

# ==========================================
# 5. 业务逻辑
# ==========================================
def load_df(key):
    candidates = FILE_MAPPING.get(key, [])
    for fname in candidates:
        paths = [os.path.join(RAW_DIR, fname), os.path.join(BASE_DIR, fname)]
        for p in paths:
            if os.path.exists(p):
                print(f"📖 读取: {os.path.basename(p)}")
                try: return pd.read_csv(p) if p.endswith('.csv') else pd.read_excel(p)
                except: return pd.read_csv(p, encoding='gbk')
    return None

def process_ingredients():
    df = load_df('ingredient')
    if df is None: return None, None
    raw_smiles = df['smiles'].fillna('').astype(str).tolist() if 'smiles' in df.columns else []
    if not raw_smiles: return None, None

    # 1. 清洗
    print(f"⚡ [CPU] 并行清洗 {len(raw_smiles)} 个 SMILES...")
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        clean_smi = list(tqdm(executor.map(worker_clean_smiles, raw_smiles), total=len(raw_smiles)))

    # 2. ChemBERTa (Encoder)
    print(f"📦 [GPU] 加载 ChemBERTa: {CHEM_PATH}")
    tok_bert = AutoTokenizer.from_pretrained(CHEM_PATH, local_files_only=True)
    mod_bert = AutoModel.from_pretrained(CHEM_PATH, local_files_only=True).to(DEVICE)
    bert_emb = run_inference_smart(mod_bert, tok_bert, clean_smi, BATCH_SIZE_MOL, "ChemBERTa", "encoder", 256)
    del mod_bert, tok_bert
    torch.cuda.empty_cache()

    # 3. ChemGPT (Decoder) - 核心新增
    print(f"📦 [GPU] 加载 ChemGPT-19M: {CHEMGPT_PATH}")
    try:
        tok_gpt = AutoTokenizer.from_pretrained(CHEMGPT_PATH, local_files_only=True)
        if tok_gpt.pad_token is None:
            if tok_gpt.eos_token is not None:
                tok_gpt.pad_token = tok_gpt.eos_token
            else:
                tok_gpt.add_special_tokens({'pad_token': '[PAD]'})

        mod_gpt = AutoModel.from_pretrained(CHEMGPT_PATH, local_files_only=True).to(DEVICE)
        if getattr(mod_gpt.config, 'pad_token_id', None) is None and tok_gpt.pad_token_id is not None:
            mod_gpt.config.pad_token_id = tok_gpt.pad_token_id

        input_emb = mod_gpt.get_input_embeddings()
        if input_emb is not None and input_emb.num_embeddings < len(tok_gpt):
            mod_gpt.resize_token_embeddings(len(tok_gpt))

        gpt_emb = run_inference_smart(mod_gpt, tok_gpt, clean_smi, BATCH_SIZE_MOL, "ChemGPT", "decoder", 256)
        del mod_gpt, tok_gpt
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ ChemGPT 失败: {e}，使用零向量替代。")
        gpt_emb = torch.zeros(len(clean_smi), 256)

    # 4. 指纹
    print(f"⚡ [CPU] 计算指纹...")
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        fp_list = list(tqdm(executor.map(worker_morgan, clean_smi), total=len(clean_smi)))
    fp_emb = torch.tensor(np.array(fp_list), dtype=torch.float32)

    # 5. 融合
    print(f"🔗 [Fusion] 融合特征: BERT + GPT + FP")
    combined_features = torch.cat([bert_emb, gpt_emb, fp_emb], dim=1)

    # 6. 3D
    print(f"⚡ [CPU] 生成 3D 构象...")
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        coords_list = list(tqdm(executor.map(worker_3d_coords, clean_smi), total=len(clean_smi)))
    coords_emb = torch.tensor(np.array(coords_list), dtype=torch.float32)

    # 返回融合特征、3D坐标，以及各子模态（供 E 组消融使用）
    return combined_features, coords_emb, bert_emb, gpt_emb, fp_emb

def process_targets():
    df = load_df('target')
    if df is None: return None
    raw_seqs = df['sequence'].fillna('').astype(str).tolist() if 'sequence' in df.columns else []
    clean_seqs = [" ".join(list(s.upper().replace(' ','').strip())) for s in raw_seqs]
    
    print(f"📦 [GPU] 加载 ProtBERT: {PROT_PATH}")
    tok = AutoTokenizer.from_pretrained(PROT_PATH, do_lower_case=False, local_files_only=True)
    mod = AutoModel.from_pretrained(PROT_PATH, local_files_only=True).to(DEVICE)
    ai_emb = run_inference_smart(mod, tok, clean_seqs, BATCH_SIZE_PROT, "ProtBERT", "encoder", 1024)
    del mod, tok
    torch.cuda.empty_cache()
    return ai_emb

def process_herbs():
    df = load_df('herb')
    if df is None: return None
    texts = []
    for _, row in df.iterrows():
        parts = [f"中药名：{str(row.get('chinese_name','')).strip()}"]
        for col in ['property_flavor', 'channel_tropism', 'effect', 'indication']:
            if pd.notna(row.get(col)): parts.append(f"{col}：{str(row[col]).strip()}")
        texts.append("。".join(parts))

    print(f"📦 加载 Qwen: {QWEN_PATH}")

    # 🔥 彻底清理 GPU 显存
    print("⚡ [清理] 释放所有 GPU 显存...")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    tok = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True, padding_side='left', local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    # 尝试 GPU 加载（如果显存不足则回退到 CPU）
    try:
        print(f"⚡ [GPU] 尝试加载 Qwen 到 GPU...")
        mod = AutoModelForCausalLM.from_pretrained(
            QWEN_PATH,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            local_files_only=True
        ).to(DEVICE)
        device_used = DEVICE
        print(f"✅ Qwen 成功加载到 GPU")
    except torch.OutOfMemoryError:
        print(f"⚠️  GPU 显存不足，回退到 CPU 运行...")
        torch.cuda.empty_cache()
        gc.collect()
        CPU_DEVICE = torch.device("cpu")
        mod = AutoModelForCausalLM.from_pretrained(
            QWEN_PATH,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            local_files_only=True
        ).to(CPU_DEVICE)
        device_used = CPU_DEVICE

        # 🔥 修改 run_inference_smart 使用正确的设备
        original_run_inference = run_inference_smart

        def run_inference_on_device(model, tokenizer, data_list, batch_size, desc, model_type="encoder", max_len=512):
            indexed_data = []
            for i, text in enumerate(data_list):
                t_str = str(text)
                indexed_data.append((i, len(t_str), t_str))
            indexed_data.sort(key=lambda x: x[1], reverse=True)
            sorted_texts = [x[2] for x in indexed_data]
            original_indices = [x[0] for x in indexed_data]

            dataset = TextDataset(sorted_texts)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
            sorted_embeddings = []
            model.eval()

            print(f"🚀 [{desc}] CPU 推理模式...")
            with torch.no_grad():
                for batch_text in tqdm(loader, desc=desc, unit="batch"):
                    try:
                        inputs = tokenizer(batch_text, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device_used)
                        outputs = model(**inputs, output_hidden_states=True)
                        last_hidden = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs.hidden_states[-1]
                        mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
                        emb = torch.sum(last_hidden * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
                        sorted_embeddings.append(emb.cpu())
                    except Exception as e:
                        print(f"⚠️ 错误: {e}")
                        sorted_embeddings.append(torch.zeros(len(batch_text), 4096))

            all_sorted = torch.cat(sorted_embeddings, dim=0)
            final_embeddings = torch.zeros_like(all_sorted)
            final_embeddings[original_indices] = all_sorted
            return final_embeddings

        ai_emb = run_inference_on_device(mod, tok, texts, BATCH_SIZE_LLM, "Qwen-Herb", "decoder", 512)
        del mod, tok
        return ai_emb

    ai_emb = run_inference_smart(mod, tok, texts, BATCH_SIZE_LLM, "Qwen-Herb", "decoder", 512)
    del mod, tok
    torch.cuda.empty_cache()
    gc.collect()
    return ai_emb

def main():
    start_time = time.time()
    save_dict = {}
    
    # ── 成分特征：保存融合向量 + 各子模态 ─────────────────────
    ing_result = process_ingredients()
    if ing_result is not None:
        ing_feat, ing_coords, bert_emb, gpt_emb, fp_emb = ing_result
        save_dict["ingredient_ai_features"]   = ing_feat
        save_dict["ingredient_coords"]         = ing_coords
        # 单独保存各模态，供 ablation.py E 组消融的 _patch_graph_features 使用
        save_dict["ingredient_bert_features"] = bert_emb
        save_dict["ingredient_gpt_features"]  = gpt_emb
        save_dict["ingredient_fp_features"]   = fp_emb

    tgt_feat = process_targets()
    if tgt_feat is not None: save_dict["target_ai_features"] = tgt_feat

    herb_feat = process_herbs()
    if herb_feat is not None: save_dict["herb_ai_features"] = herb_feat

    output_path = os.path.join(PROCESSED_DIR, "real_features_ultra.pt")
    torch.save(save_dict, output_path)

    # ── 保存特征维度元数据 feature_dims.json ──────────────────
    # ablation.py 中的 _get_ing_split() 优先读此文件做精确子向量切分
    import json as _json
    feature_dims = {}
    if ing_result is not None:
        feature_dims["ingredient"] = {
            "bert":  int(bert_emb.shape[1]),
            "gpt":   int(gpt_emb.shape[1]),
            "fp":    int(fp_emb.shape[1]),
            "total": int(ing_feat.shape[1]),
        }
    if tgt_feat is not None:
        feature_dims["target"] = {
            "protbert": int(tgt_feat.shape[1]),
            "total":    int(tgt_feat.shape[1]),
        }
    if herb_feat is not None:
        feature_dims["herb"] = {
            "qwen":  int(herb_feat.shape[1]),
            "total": int(herb_feat.shape[1]),
        }
    dims_path = os.path.join(PROCESSED_DIR, "feature_dims.json")
    with open(dims_path, "w") as f:
        _json.dump(feature_dims, f, indent=2)
    print(f"\n📐 特征维度元数据: {dims_path}")
    print(f"   {_json.dumps(feature_dims)}")

    print(f"\n🎉 完美完成！耗时: {(time.time()-start_time)/60:.1f}m. 结果: {output_path}")

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()