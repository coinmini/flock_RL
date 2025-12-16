#!/usr/bin/env python3
"""
本地测试脚本 - 直接用本地模型和测试数据进行验证，不依赖 API
"""

import sys
import click
import numpy as np
from validator.modules.rl import RLValidationModule, RLConfig, RLInputData


@click.command()
@click.option('--local-model-path', required=True, type=str, help='本地 ONNX 模型文件路径')
@click.option('--local-data-path', required=True, type=str, help='本地测试数据文件路径 (.npz)')
@click.option('--batch-size', default=512, type=int, help='批处理大小')
@click.option('--seed', default=42, type=int, help='随机种子')
@click.option('--max-params', default=None, type=int, help='模型参数上限（可选）')
def main(local_model_path: str, local_data_path: str, batch_size: int, seed: int, max_params: int):
    """
    本地测试 RL 模型验证

    示例:
        python local_test.py --local-model-path /path/to/model.onnx --local-data-path /path/to/test.npz
    """
    print("=" * 60)
    print("本地测试模式")
    print("=" * 60)
    print(f"模型路径: {local_model_path}")
    print(f"数据路径: {local_data_path}")
    print(f"批处理大小: {batch_size}")
    print(f"随机种子: {seed}")
    print("=" * 60)

    # 创建配置
    config = RLConfig(per_device_eval_batch_size=batch_size, seed=seed)

    # 创建验证模块
    module = RLValidationModule(config)

    # 加载本地模型
    print("\n[1/3] 加载模型...")
    try:
        model = module._load_model(
            repo_id="",  # 不使用
            filename="",  # 不使用
            revision="",  # 不使用
            max_params=max_params,
            local_model_path=local_model_path
        )
    except FileNotFoundError as e:
        print(f"错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"模型加载失败: {e}")
        sys.exit(1)

    # 加载本地测试数据
    print("\n[2/3] 加载测试数据...")
    try:
        with np.load(local_data_path) as test_data:
            test_X = test_data['X']
            test_Info = test_data['Info']
        print(f"测试数据: X_test {test_X.shape}, Info_test {test_Info.shape}")
    except FileNotFoundError:
        print(f"错误: 测试数据文件不存在: {local_data_path}")
        sys.exit(1)
    except KeyError as e:
        print(f"错误: 测试数据格式不正确，缺少键: {e}")
        sys.exit(1)

    # 运行验证
    print("\n[3/3] 运行验证...")
    from validator.modules.rl.env import EnvLite

    env = EnvLite(test_X, test_Info, batch_size=batch_size, seed=seed)
    N = env.N
    all_rewards = []

    for start_idx in range(0, N, batch_size):
        end_idx = min(start_idx + batch_size, N)
        batch_indices = np.arange(start_idx, end_idx)
        env.idx = batch_indices
        env.X_b = env.X_all[batch_indices]
        env.Info_b = env.Info_all[batch_indices]
        env.qty_b = env.qty_all[batch_indices]
        env.duration_b = env.duration_all[batch_indices]
        env.fill_b = env.fill_all[batch_indices, :]
        env.rebate_b = env.rebate_all[batch_indices, :]
        env.punish_b = env.punish_all[batch_indices, :]
        env.vol_b = env.vol_all[batch_indices, :]

        input_name = model.get_inputs()[0].name
        outputs = model.run(None, {input_name: env.X_b})
        action = outputs[0]
        reward = env.step(action)
        all_rewards.append(reward)

        # 打印进度
        progress = (end_idx / N) * 100
        print(f"\r进度: {progress:.1f}% ({end_idx}/{N})", end="", flush=True)

    print()  # 换行

    # 计算结果
    all_rewards = np.concatenate(all_rewards)
    average_reward = float(np.mean(all_rewards))

    print("\n" + "=" * 60)
    print("验证结果")
    print("=" * 60)
    print(f"平均奖励 (average_reward): {average_reward:.6f}")
    print(f"最小奖励: {np.min(all_rewards):.6f}")
    print(f"最大奖励: {np.max(all_rewards):.6f}")
    print(f"奖励标准差: {np.std(all_rewards):.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
