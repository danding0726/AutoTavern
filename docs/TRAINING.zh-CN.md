# 训练策略网络

使用 TavernGym 的数值观测训练 PyTorch 策略网络，无需语言模型或游戏客户端。
本实现提供单学习席位对战七个内置规则机器人的 PPO，不包含自博弈。

## 安装与运行

从仓库根目录执行。需要 Python 3.10+、仓库指定的 Rust 工具链；训练额外需要 PyTorch。
如已有适合显卡的 PyTorch，直接复用，避免重装 CUDA 依赖。

```bash
conda activate qwen35-vllm
pip install -e ".[train,test]"
python -m taverngym.train --out runs/ppo --device cuda
```

默认 32 局并行，每局控制席位 0，4 个 Rust 工作线程，每次更新采集 256 步，
执行 4 轮 PPO minibatch 更新，共执行 1,000 次更新，即 8,192,000 次玩家操作。
没有 GPU 时使用 `--device cpu`。可先做短程验证：

```bash
python -m taverngym.train --out runs/smoke --device cpu \
    --n-envs 4 --rollout-steps 64 --updates 2 --epochs 1 --minibatch-size 128
```

如果直接使用源码目录，并且已有当前源码编译出的本地 `python/taverngym/_core*.so`，
可以用 `PYTHONPATH=python python -m taverngym.train ...` 运行；pytest 也会优先读取源码目录。
正常的 editable 安装会准备本地扩展。

## 模型与算法

- `features`：3,227 个数值特征，经过 `sign(x) * log(1 + abs(x))` 压缩，防止后期巨大属性主导输入。
- `cards`：124 个卡牌索引，经 16 维 Embedding 后按位置展平，空槽索引 0。
- 拼接后经 512、256 隐藏单元的 Tanh MLP，输出 245 个动作 logits 和一个价值估计。
- 参数量约 293 万。合法动作掩码同时用于采样、PPO 概率计算和熵计算。
- 奖励为原始最终名次奖励：第一名 +1、第八名 -1，线性插值；没有额外人工奖励。
- 默认 `gamma=0.999`、`gae_lambda=0.99`、学习率 `3e-4`、PPO clip 0.2、
  熵系数 0.01、价值损失系数 0.5、梯度范数上限 0.5。
- 默认每回合最多 60 次操作，防止初期策略无限调整站位。这是训练约束，可通过
  `--action-cap` 修改，评估时应使用相同值。

训练器限定 `controlled=(0,)`。此模式下，每次环境推进后，学习席位都已可行动，
或原局结束并自动开始新局，因此可以收集 `[rollout_steps, n_envs]` 的轨迹。
GAE 在 `done` 处切断 bootstrap 和递推，避免将新局价值混入旧局。
rollout 在未结束的局面截断时，使用下一局面价值 bootstrap。

环境没有独立的 time-limit truncated 标志。本实现将环境报告的所有 done 视为终止；
遇到非法动作、不可行动槽位或无有效名次的停止局面，会报错，不将其当成正常训练数据。
它不支持直接将 controlled 改为多席位自博弈；那需要另写按席位决策收集轨迹的采样器。

## 输出与继续训练

输出目录包含：

| 文件 | 内容 |
| --- | --- |
| `latest.pt` | 每次更新后保存的模型、Adam 状态、更新次数、操作总数、环境配置和引擎指纹 |
| `metrics.jsonl` | PPO 损失、熵、近似 KL、clip 比例、采样速度、完成局数，以及最近最多 100 局的名次指标 |
| `config.json` | 最近一次启动的命令参数 |

继续训练，`--updates` 表示**额外**执行的更新次数：

```bash
python -m taverngym.train --out runs/ppo --resume runs/ppo/latest.pt \
    --device cuda --updates 1000
```

恢复模型和优化器后会从新游戏开始，不恢复未结束的局面、rollout 或随机数状态，
因此不是逐步一致的断点回放。继续训练时 `--action-cap` 必须与检查点一致。
学习率使用本次命令的 `--lr`，其余 PPO 参数也以本次命令为准。
输出目录已有训练结果时，必须显式指定 `--resume` 或更换目录。
检查点会校验编译引擎指纹和观测维度，拒绝跨不兼容引擎加载。

## 独立评估

默认使用从 1,000,000 开始的连续游戏种子，与训练入口的默认种子分开。
先运行随机合法动作基线，再在相同游戏种子上评估策略：

```bash
python -m taverngym.evaluate --random --games 100 --seed 1000000 \
    --out runs/ppo/random-eval.json
python -m taverngym.evaluate --checkpoint runs/ppo/latest.pt \
    --games 100 --seed 1000000 --out runs/ppo/policy-eval.json
```

策略默认选概率最高的合法动作；添加 `--sample` 可按概率采样。
随机基线是在合法动作中均匀采样。两种策略的操作不同，消耗游戏随机数的方式也不同，
所以相同初始种子不意味着它们会经历完全相同的后续商店和战斗。
评估输出平均名次及标准误、前四率、吃鸡率、平均奖励、每局操作数和名次分布。
平均名次越低越好。自定义训练操作上限后，评估也要传入相同的 `--action-cap`。

短程运行只能验证采样、反向传播、保存和加载是否正常，不能证明策略已经学会游戏。
正式实验应增加采样量，在更多独立种子上比较多个训练检查点，并保留独立的最终测试种子。
内置机器人只是启发式对手，战胜它们不代表达到人类玩家水平。

## 测试

```bash
python -m pytest -q tests/test_training.py
python -m pytest -q
```

新增测试覆盖合法动作掩码、GAE 的终止与截断计算、真实批量对局中的奖励和参数更新、
检查点恢复、引擎不兼容检查、继续训练、评估可复现性，以及异常采样状态的拒绝。
未安装 PyTorch 时，训练测试会跳过，基础模拟器不依赖 PyTorch。
