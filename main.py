# -*- coding: utf-8 -*-
"""
M²-RLHGT 主入口（Cold-Start Edition）
"""
import os
import sys
import torch
import argparse

from config import PROCESSED_DIR, VIS_DIR, MODEL_DIR


def parse_args():
    parser = argparse.ArgumentParser(
        description="M²-RLHGT: TCM靶点预测（冷启动评估）"
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='all',
        choices=['preprocess', 'train', 'evaluate', 'all'],
        help=(
            'preprocess: 仅预处理数据\n'
            'train:      仅训练消融实验\n'
            'evaluate:   仅评估（需先训练）\n'
            'all:        全流程（默认）'
        )
    )
    parser.add_argument(
        '--gpu', type=int, default=0,
        help='GPU编号（默认0，-1表示CPU）'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='?????????42?'
    )
    parser.add_argument(
        '--seeds', nargs='*', type=int, default=None,
        help='??????????: --seeds 42 43 44 45 46'
    )
    return parser.parse_args()


def setup_device(gpu_id: int):
    if gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        print(f"🖥️  使用GPU: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device("cpu")
        print("🖥️  使用CPU")
    return device


def run_preprocess():
    """预处理：构建图 + 生成herb_to_pairs.json"""
    print("\n" + "="*60)
    print("  Step 1: 数据预处理（Cold-Start Edition）")
    print("="*60)

    from processed import HeteroGraphProcessor
    processor = HeteroGraphProcessor()
    processor.process()

    # 验证herb_to_pairs.json是否生成
    pairs_path = os.path.join(PROCESSED_DIR, "herb_to_pairs.json")
    if os.path.exists(pairs_path):
        import json
        with open(pairs_path, 'r') as f:
            pairs = json.load(f)
        print(f"\n✅ herb_to_pairs.json 已生成")
        print(f"   herb分组数: {len(pairs)}")
        total = sum(len(v) for v in pairs.values())
        print(f"   总(ing,tgt)对: {total}")
    else:
        print("❌ herb_to_pairs.json 未生成，请检查processed.py")
        sys.exit(1)


def run_train(seed: int = 42, seeds=None):
    """训练所有消融变体"""
    print("\n" + "="*60)
    print("  Step 2: 消融训练（Cold-Start Edition）")
    print("  数据划分: 按herb分组，测试herb训练时完全不可见")
    print("="*60)

    from experiment_suite import run_ablation_suite, ensure_dirs, generate_report
    results = run_ablation_suite(seed=seed, seeds=seeds)
    generate_report(ensure_dirs(os.path.join(MODEL_DIR, "ablation")))
    return results


def run_evaluate(seed: int = 42):
    """评估所有消融变体"""
    print("\n" + "="*60)
    print("  Step 3: 消融评估（Cold-Start Edition）")
    print("="*60)

    from experiment_suite import ensure_dirs, generate_report
    results = generate_report(ensure_dirs(os.path.join(MODEL_DIR, "ablation")))
    return results


def check_prerequisites(mode: str):
    """检查运行前提条件"""
    errors = []

    if mode in ('train', 'all'):
        graph_path = os.path.join(PROCESSED_DIR, "hetero_graph.pt")
        pairs_path = os.path.join(PROCESSED_DIR, "herb_to_pairs.json")

        if not os.path.exists(graph_path):
            errors.append(
                f"❌ 缺少图文件: {graph_path}\n"
                f"   请先运行: python main.py --mode preprocess"
            )
        if not os.path.exists(pairs_path):
            errors.append(
                f"❌ 缺少herb分组文件: {pairs_path}\n"
                f"   请先运行: python main.py --mode preprocess"
            )

    if mode == 'evaluate':
        graph_path = os.path.join(PROCESSED_DIR, "hetero_graph.pt")
        if not os.path.exists(graph_path):
            errors.append(
                f"❌ 缺少图文件: {graph_path}\n"
                f"   请先运行: python main.py --mode preprocess"
            )

        ablation_dir = os.path.join(MODEL_DIR, "ablation")

        if not os.path.exists(ablation_dir):
            errors.append(
                f"❌ 未找到任何训练好的模型(best_*.pth)\n"
                f"   请先运行: python main.py --mode train"
            )

    if errors:
        for e in errors:
            print(e)
        sys.exit(1)


def main():
    args   = parse_args()
    device = setup_device(args.gpu)

    print(f"\n{'🚀'*20}")
    print(f"  M²-RLHGT TCM靶点预测系统（Cold-Start Edition）")
    print(f"  Mode: {args.mode} | Seed: {args.seed} | Seeds: {args.seeds}")
    print(f"{'🚀'*20}")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)

    if args.mode == 'preprocess':
        run_preprocess()

    elif args.mode == 'train':
        check_prerequisites('train')
        run_train(seed=args.seed, seeds=args.seeds)

    elif args.mode == 'evaluate':
        check_prerequisites('evaluate')
        run_evaluate(seed=args.seed)

    elif args.mode == 'all':
        # 全流程
        run_preprocess()
        check_prerequisites('train')
        run_train(seed=args.seed, seeds=args.seeds)
        run_evaluate(seed=args.seed)

    print(f"\n{'✅'*20}")
    print(f"  完成！结果保存在: {VIS_DIR}")
    print(f"{'✅'*20}\n")


if __name__ == "__main__":
    main()
