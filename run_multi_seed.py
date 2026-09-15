# -*- coding: utf-8 -*-
"""
多随机种子实验脚本
运行主实验（Vanilla, Static, Proposed）各5个种子

使用方法：
    python run_multi_seed.py

会运行：
    - Vanilla HGT (hgt_baseline) x 5 seeds
    - Static Similarity HGT (static_similarity_hgt) x 5 seeds
    - M²-RLHGT (proposed_model) x 5 seeds

结果保存到：
    checkpoints/strict_main_multi_seed/
"""
import os
import sys
import json
import time

# 添加当前目录到路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# 设置工作目录
os.chdir(SCRIPT_DIR)
sys.stdout.reconfigure(encoding='utf-8')

# 定义5个随机种子
SEEDS = [42, 123, 456, 2024, 9999]

# 主实验变体（只运行这3个）
MAIN_VARIANTS = [
    "hgt_baseline",      # Vanilla HGT
    "static_similarity_hgt",  # Static Similarity HGT
    "proposed_model",     # M²-RLHGT
]


def run_multi_seed():
    """运行多种子主实验"""
    from experiment_suite import (
        run_variant_seed,
        ensure_dirs,
        VARIANT_INDEX,
        STRICT_SPLIT_CFG,
        STRICT_PROTOCOL_VERSION,
        STRICT_SPLIT_ID,
        STRICT_THRESHOLD_POLICY,
        _write_json,
        _seed_dir,
        _summary_path,
        _error_path,
    )
    import traceback
    import numpy as np

    # 输出目录
    output_root = os.path.join(SCRIPT_DIR, "checkpoints", "strict_main_multi_seed")
    paths = ensure_dirs(output_root)

    print("=" * 72)
    print("  多随机种子主实验")
    print("  变体: Vanilla HGT, Static HGT, M²-RLHGT")
    print(f"  种子: {SEEDS}")
    print("=" * 72)

    # 收集所有结果
    all_results = []

    # 对每个变体运行5个种子
    for variant_key in MAIN_VARIANTS:
        variant = VARIANT_INDEX.get(variant_key)
        if not variant:
            print(f"  [警告] 未找到变体: {variant_key}")
            continue

        variant_name = variant.get("display_name", variant_key)
        print(f"\n{'=' * 72}")
        print(f"  变体: {variant_name} ({variant_key})")
        print(f"{'=' * 72}")

        for seed in SEEDS:
            print(f"\n  [种子 {seed}]")
            start_time = time.time()

            try:
                # 重置和配置train模块
                import train as train_module
                train_module.reset_cfg()

                overrides = dict(variant.get("overrides", {}))
                run_dir = _seed_dir(paths, variant_key, seed)
                os.makedirs(run_dir, exist_ok=True)

                overrides.update({
                    "save_dir": run_dir,
                    "seed": int(seed),
                    "curve_alias": variant_key,
                    "experiment_name": variant_key,
                    "experiment_group": variant["group"],
                    "experiment_title": variant["title"],
                    "rl_save_path": os.path.join(run_dir, "rl_agent.pt"),
                })
                overrides.update(STRICT_SPLIT_CFG)
                train_module.merge_cfg(overrides)

                # 运行训练
                print(f"    开始训练...")
                summary = train_module.main()

                elapsed = time.time() - start_time

                # 整理结果
                summary.update({
                    "display_name": variant["display_name"],
                    "status": "ok",
                    "source": "internal",
                    "order": variant["order"],
                    "seed": int(seed),
                    "strict_protocol_version": STRICT_PROTOCOL_VERSION,
                    "split_id": STRICT_SPLIT_ID,
                    "threshold_policy": STRICT_THRESHOLD_POLICY,
                })
                _write_json(_summary_path(paths, variant_key, seed), summary)

                auc = float(summary.get("test_AUC", 0))
                f1 = float(summary.get("test_F1", 0))
                print(f"    [完成] AUC={auc:.4f} F1={f1:.4f} ({elapsed:.1f}s)")

                all_results.append({
                    "variant": variant_key,
                    "seed": seed,
                    "status": "success",
                    "test_AUC": auc,
                    "test_AUPRC": float(summary.get("test_AUPRC", 0)),
                    "test_F1": f1,
                    "test_Precision": float(summary.get("test_Precision", 0)),
                    "test_Recall": float(summary.get("test_Recall", 0)),
                    "test_ACC": float(summary.get("test_ACC", 0)),
                    "elapsed_seconds": elapsed,
                })

            except Exception as exc:
                elapsed = time.time() - start_time
                error_info = traceback.format_exc()
                print(f"    [错误] {str(exc)}")

                _write_json(_error_path(paths, variant_key, seed), {
                    "variant": variant_key,
                    "seed": seed,
                    "error": str(exc),
                    "traceback": error_info,
                })

                all_results.append({
                    "variant": variant_key,
                    "seed": seed,
                    "status": "error",
                    "error": str(exc),
                    "elapsed_seconds": elapsed,
                })

    # 保存结果到JSON
    results_path = os.path.join(output_root, "multi_seed_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n\n{'=' * 72}")
    print("  结果汇总")
    print("=" * 72)

    # 按变体汇总
    summary = {}
    for result in all_results:
        if result["status"] == "success":
            vk = result["variant"]
            if vk not in summary:
                summary[vk] = {
                    "name": VARIANT_INDEX.get(vk, {}).get("display_name", vk),
                    "seeds": [],
                    "auc_values": [],
                    "f1_values": [],
                }
            summary[vk]["seeds"].append(result["seed"])
            summary[vk]["auc_values"].append(result["test_AUC"])
            summary[vk]["f1_values"].append(result["test_F1"])

    print(f"\n{'变体':<25} {'Seeds':<12} {'AUC':<15} {'F1':<15}")
    print("-" * 70)

    for vk, data in summary.items():
        auc_mean = np.mean(data["auc_values"])
        auc_std = np.std(data["auc_values"])
        f1_mean = np.mean(data["f1_values"])
        f1_std = np.std(data["f1_values"])
        seeds_str = ",".join(str(s) for s in data["seeds"])
        print(f'{data["name"]:<25} {seeds_str:<12} {auc_mean:.4f}±{auc_std:.4f}  {f1_mean:.4f}±{f1_std:.4f}')

    print(f"\n\n结果已保存: {results_path}")

    return all_results


if __name__ == "__main__":
    run_multi_seed()
