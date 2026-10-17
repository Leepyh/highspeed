"""
MADDPG配置文件 - 多智能体应急车道控制
分段控制策略：将道路分成多段，每段独立控制应急车道开放程度
"""

import torch

# ==================== 元胞自动机模型参数 ====================
CA_CONFIG = {
    # 道路参数（固定）
    'L': 1000,             # 道路长度（元胞数）- 5公里
    'n_lanes': 3,          # 车道数量（0=超车道, 1=行车道, 2=应急车道）
    
    # 车辆数量范围（每轮随机采样，避免过拟合）
    # 说明：原值 (200, 350) 实测密度仅 0.10~0.175，社会车辆均速 4.96~5.00，
    #       远高于拥堵阈值（3.0/2.0），导致"拥堵时开放应急车道"这一核心场景
    #       在训练中从不触发。经 tools/sweep_density.py 实测，均速在
    #       n_cars≈400 处跌破 60% 阈值、n_cars≈650 处跌破 40% 阈值。
    #       现取 (450, 600)：关闭时均速约 2.5→1.8（轻度~严重拥堵），
    #       开放时约 4.7→3.3，两种拥堵状态都会出现且开/关差异显著。
    'n_cars_range': (450, 600),
    
    # 速度参数（固定）
    'v_max': 5,                      # 最大速度
    'v_initial_range': (2, 5),       # 初始速度范围（每辆车独立随机）
    
    # 应急车辆参数
    'D': 40,                         # 应急车辆影响范围（元胞数）
    'v_emergency': 5,                # 应急车辆速度
    'P_emergency_range': (0.3, 0.5), # 应急车出现概率范围（30%-50%随机）
    'max_emergency': 3,              # 最大应急车辆数
    # 应急车让行触发距离：应急车后方该距离内的社会车辆需退出应急车道。
    # 经 tools/check_emergency_yield.py 实测，D=60 时应急车平均速度最高
    # (3.61)、最长连续停滞从 9 步降到 3 步；再增大改善有限且会挤占社会车辆。
    'emergency_yield_distance': 60,
    # 应急车速度下界（验收测试用）：低于该值视为被社会车辆堵住
    'emergency_min_acceptable_speed': 2.0,
    
    # 行为参数（固定）
    'P1': 0.6,                       # 无应急车时换道概率
    'P2': 0.9,                       # 有应急车时换道概率
    
    # 安全距离参数（固定）
    'dev': 2,                        # 安全距离
    
    # 仿真参数（固定）
    'warm_up_steps': 100,            # 预热步数（固定100，让系统更稳定）
    'max_steps': 500,                # 每轮最大仿真步数
}

# ==================== 多智能体环境参数 ====================
ENV_CONFIG = {
    # 分段控制参数
    'n_segments': 5,       # 将道路分成5段（每段200元胞）
    'n_agents': 5,         # 5个智能体，每个控制一段
    
    # 状态空间参数
    'state_history': 5,    # 状态历史长度
    'local_observation_range': 2,  # 每个智能体能观察相邻段的数量（2=观察左右各2段，共5段）
    
    # 动作空间（离散）
    # 每个智能体的动作：0或1（该段应急车道的开/关）
    # 0 = 关闭，1 = 开放
    'n_actions': 2,        # 每个智能体的动作数量（离散）
    
    # 决策频率
    'decision_interval': 3,  # 每3步做一次决策（更频繁）
    
    # 奖励函数权重（与物理后果挂钩版本）
    #
    # 设计原则：奖励必须由"动作产生的实际后果"决定，而不是由动作本身决定。
    # 修复前 emergency_lane_capacity_bonus / emergency_vehicle_open_reward 等项
    # 只要 action==1 就无条件加分，等价于给常量加分，智能体可以不做任何
    # 有效控制就刷高奖励。
    #
    # 现在保留三类项：
    #   A. 后果项：直接读仿真状态（速度），动作通过改变状态间接影响奖励
    #   B. 通行效率项：全局/局部速度，体现"开放缓解拥堵"的收益
    #   C. 轻微资源占用项：提醒应急车道是稀缺资源，权重很小
    'reward_weights': {
        # ---- C. 资源占用（小权重，仅作提示）----
        # 不需要开放且关闭：保留应急车道的轻微正收益
        'correct_close_reward': 1.0,
        # 不需要开放却开放：轻微浪费惩罚（拥堵时 local_need_open 为真，不触发）
        'emergency_lane_usage_penalty': -1.0,

        # ---- A. 应急车通行后果（次要权重项）----
        # 有应急车时，本段应急车速度越接近 v_max 奖励越高。
        # 权重从 15 降到 4：因为"关闭"时应急车恒为 v_max=5，该项在关闭状态下
        # 近似一个常数（约 +15/步），会压倒真正的任务奖励，使最优动作在密度
        # 区间内非单调翻转，训练结果退化为"恒关闭"。
        # 降权后它只作为"别把应急车堵死"的约束，而非主导信号。
        'emergency_speed_weight': 4.0,

        # 以下三个键在旧版中按"动作"给分，现置 0 停用（保留键名以兼容旧配置读取）
        'emergency_vehicle_open_reward': 0.0,
        'emergency_vehicle_close_penalty': 0.0,
        'emergency_lane_capacity_bonus': 0.0,
        'congestion_open_reward': 0.0,

        # ---- B. 通行效率（主导权衡项）----
        # 权重经实测标定：使"最优动作"在 n_cars 450→600 上单调翻转
        # （低密度 CLOSE、高密度 OPEN），且对 P_emergency 稳健。
        # 标定证据见 tools/check_policy_state_dependence.py 与模型诊断报告.md。
        #
        # 标定结果（Δ = 开放奖励 − 关闭奖励）：
        #   n_cars  450: -3920  CLOSE        n_cars  560: +2273  OPEN
        #   n_cars  480: -2149  CLOSE        n_cars  600: +3972  OPEN
        #   n_cars  520: +4353  OPEN
        # P_em = 0.3 / 0.4 / 0.5 三档判定完全一致。
        'global_avg_speed': 20.0,              # 全局平均车速权重
        'congestion_relief_weight': 20.0,      # 拥堵缓解额外权重（仅在确实拥堵时生效）
        'speed_improvement_weight': 40.0,      # 车速改善权重（直接度量疏解效果）
        'global_congestion_penalty': -2.0,     # 全局拥堵惩罚
        'local_speed': 2.0,                    # 本段车速权重
        'local_density_balance': 1.0,          # 本段车道均衡权重（预留）

        # ---- 平滑性与协同（仍作用在动作上，属于控制代价而非任务奖励）----
        'local_action_smoothness': -0.05,      # 动作切换代价
        'coordination_consistency': -0.1,      # 相邻段动作不一致代价
    },
    
    # 拥堵阈值（调整：从2.0/1.0提高到3.0/2.0，更合理）
    'congestion_threshold': 3.0,         # 轻度拥堵阈值（速度<3.0，即<60%最大速度）
    'severe_congestion_threshold': 2.0,  # 严重拥堵阈值（速度<2.0，即<40%最大速度）
}

# ==================== MADDPG智能体参数 ====================
AGENT_CONFIG = {
    # Actor网络结构（每个智能体一个）
    'actor_hidden_dims': [128, 64],      # Actor隐藏层
    
    # Critic网络结构（集中式，所有智能体共享）
    'critic_hidden_dims': [256, 128, 64], # Critic隐藏层（更大，处理全局信息）
    
    # 学习率
    # actor_lr 从 1e-4 降到 3e-5：实测原设置下策略熵在第 400 轮就坍缩到 0
    # （5 个智能体全部恒选"关闭"），actor 收敛过快是主因之一。
    'actor_lr': 3e-5,        # Actor学习率
    'critic_lr': 1e-3,       # Critic学习率（通常比Actor高）
    
    # 折扣因子
    'gamma': 0.95,           # 折扣因子（降低，更关注近期奖励）
    
    # 软更新参数
    'tau': 0.005,            # 目标网络软更新系数

    # ---- 探索与熵正则（2026-10-01 第二次修复）----
    #
    # 历史（两轮都失败，见 会话记录与交接文档.md §7.1 与 §11.3b）：
    #   固定 coef=0.02 -> 第 400 轮熵坍缩到 0，恒选"关闭"，Avg 3569 跌到 3246；
    #   固定 coef=5.0  -> 熵全程 0.68/0.693 ≈ 均匀随机，argmax 策略奖励平在
    #                     3346 无提升，第 1000 轮后 argmax 退化成"近乎恒关闭"。
    #
    # 结论：**熵系数不能是一个贯穿全程的常数**。前期需要大系数保住探索，
    # 后期必须压小，否则 actor 的对数几率永远贴近 0、概率停在 0.5 附近，
    # argmax 只是在读噪声 —— 实测 27.4% 的决策点 |p-0.5|<0.05，
    # 这正是"600 多轮后不理想"、相邻检查点排名乱跳的直接原因。
    #
    # 改为**线性退火**：entropy_coef(ep) 从起点线性降到终点，
    # 走完 entropy_anneal_episodes 轮后保持终值。
    'entropy_coef': 1.0,             # 退火起点（前期探索）
    'entropy_coef_end': 0.05,        # 退火终点（后期允许策略变确定）
    'entropy_anneal_episodes': 800,  # 用多少轮从起点线性降到终点
    'entropy_anneal': True,          # False = 固定为 entropy_coef（复现旧行为）

    # ε 探索：改为**线性衰减**，探索时长可精确控制。
    # （指数 0.998 在第 1000 轮仍有 13.5%，第 2000 轮已压到下限，前后不均匀）
    'epsilon_schedule': 'linear',    # 'linear' | 'exp'
    'epsilon_decay': 0.998,          # 仅 epsilon_schedule='exp' 时使用
    'epsilon_decay_episodes': 800,   # 仅 'linear'：多少轮从 1.0 线性降到下限
    'epsilon_min': 0.05,     # ε 下限（保留 5% 随机探索，不归零）

    # 探索噪声（连续动作版参数，离散动作下不使用，保留以兼容旧配置读取）
    'noise_std': 0.2,        # 高斯噪声标准差
    'noise_decay': 0.995,    # 噪声衰减
    'noise_min': 0.05,       # 最小噪声
    
    # 经验回放
    'buffer_capacity': 500000,  # 经验回放缓冲区容量（更大，多智能体需要更多样本）
    'batch_size': 128,          # 批次大小
    
    # 目标策略平滑（离散动作版）
    #
    # 【本次修复的坑】原 'target_noise' / 'target_noise_clip' 是连续动作版 TD3
    # 的高斯噪声参数，离散动作下**从未被使用**：learn() 里目标 Q 直接把 target
    # actor 输出的**概率向量（软动作）**喂给 target critic。但 critic 只在
    # **独热(one-hot)动作**上被训练过 —— 等于目标 Q 一直在分布外输入点上求值，
    # TD 目标本身带系统偏差。这是训练后期不稳定的重要来源之一。
    #
    # 现改为：目标动作取 target actor 的 argmax（独热，回到分布内），并以
    # target_flip_p 的概率翻转某一智能体的目标动作，作为"目标策略平滑"的
    # 离散对应物，抑制 critic 过度自信。
    'target_action_mode': 'greedy',  # 'greedy'（推荐）| 'soft'（旧行为）
    'target_flip_p': 0.10,           # 离散目标平滑：翻转概率，0 = 关闭

    # Actor 更新方式（2026-10-01 修复）
    #
    # 【本次修复的坑】旧实现把 5 个 Actor 的**软概率向量**拼成联合动作喂给 critic：
    #     q = critic_i(obs, [pi_0, ..., pi_{n-1}])
    # 但 critic 只在**独热动作**上被训练过（critic 更新用的是 F.one_hot）。
    # 用"只在独热输入上有效"的 critic 去评价软动作，梯度为零 —— actor 实际上
    # 学不到任何东西，策略概率长期停在 0.5 附近，argmax 只是在读噪声。
    # 单元测试 analysis/test_actor_update.py 复现并验证了这一点：
    #     旧更新: p(open) 0.5177 -> 0.5177（零梯度）
    #     新更新: p(open) 0.5177 -> 0.9913
    #
    # 'counterfactual'（推荐）：其余智能体保持本批次实际执行的独热动作，
    #   只把第 i 个动作分别设为 0/1，得到 Q(s,a_i=0)、Q(s,a_i=1)，
    #   再按当前策略概率加权： L = -(p0*Q0 + p1*Q1) - beta*H(p)
    # 'legacy'：旧行为，仅用于 A/B 对照，不建议使用。
    'actor_update_mode': 'counterfactual',

    # ---- Twin critic（TD3 的 clipped double-Q，2026-10-01 新增）----
    #
    # 依据：Q 过估计是"训练先升后降"最经典的成因 —— critic 把某个动作的价值估高，
    # actor 追过去，实际回报跟不上，策略就退化。TD3 用两套 critic 取 min 压制它。
    # 开源对照（见 analysis/训练策略对比与改进.md）：
    #   Ronchy2000/Multi-agent-RL (MATD3) : 双 critic + min，policy_freq=3，目标平滑 0.1/0.5
    #   yongqingzs/MARL-code-pytorch      : 双 critic + min，policy_freq=2，目标平滑 0.2/0.5
    #   YuHan-.../Highway-Ramp-Metering   : 双 critic + min，POLICY_FREQ=150
    #   sfujim/TD3（原作者）              : target_Q = min(target_Q1, target_Q2)
    # 本项目原本只有单 critic。
    #
    # ⚠️ TD3 的三件套是 **twin critic + 目标策略平滑 + 延迟策略更新**，
    #    "只加 twin critic"不算 TD3。本项目后两件已有：
    #      延迟策略更新： policy_freq = 2（原有）
    #      目标策略平滑： target_action_mode='greedy' + target_flip_p=0.10（本次已有）
    #    所以再加上第三件就是完整的。
    #
    # 兼容性：默认 False = 行为与旧版完全一致且不改变检查点格式；
    #         开启后检查点多存一个可选键 critic2_state_dicts，
    #         旧检查点在新代码下仍可加载（用第一套 critic 热启动第二套）。
    # ============================================================================
    # ⚠️ 2026-10-02 实测结论：**twin critic 是负收益，默认已改回 False**
    #
    # 实测（5 个种子 x 700 轮，逐检查点确定性 argmax 筛查 + 160 个配对场景）：
    #   runA_ep200（无 twin）   3473.81   <- 当前最优
    #   旧最优 ep664            3439.56
    #   v5 最佳(seed3 ep100)    3392.69   <- 有 twin
    #   v5 次佳(seed3 ep500)    3393.14   <- 有 twin
    #
    #   配对检验（160 场景）：旧最优 显著优于 v5 最佳 +46.9，95% CI [+22.7, +71.0]
    #                        旧最优 显著优于 v5 次佳 +46.4，95% CI [+17.2, +75.6]
    #   => **加 twin critic 反而显著更差**（约 47 分）。
    #
    # 另外这次 5 个种子的结果高度分化：seed3 的 7 个检查点全部健康（最高 3325），
    # seed4 的 7 个全部退化到 ~2920。**随机种子的影响远大于算法改动**，
    # 所以单个 5 种子样本仍不足以给出终审结论；但"最好的一版反而更差"已足够
    # 支持把默认值改回 False。
    #
    # 代码保留（`--twin-critics` 可随时启用），想做严格的 A/B 就用
    # `run_multiseed.py --no-twin-critics` 再跑 5 个种子对比分布。
    # ============================================================================
    'twin_critics': False,

    'target_noise': 0.2,      # [已弃用] 连续动作版高斯噪声，离散动作不使用
    'target_noise_clip': 0.5, # [已弃用] 同上，仅为兼容旧配置读取保留
    
    # 训练频率
    # train_every: 每多少个决策步做一次梯度更新。
    #   决策间隔为 3 个 CA 步，状态变化平缓，隔步更新几乎不损失样本效率，
    #   但可把梯度更新的开销按比例摊薄（实测梯度更新占单步耗时约 76%）。
    #   1 = 每步都更新（原行为），2 = 隔步更新。
    'policy_freq': 2,         # Actor更新频率（每2次Critic更新更新1次Actor）
    'train_every': 2,         # 每2个决策步做一次梯度更新
    
    # 梯度裁剪
    'max_grad_norm': 1.0,     # 最大梯度范数
}

# ==================== 训练参数 ====================
TRAIN_CONFIG = {
    # 训练轮数
    # 从 3000 提到 4000：上一版 best 停在第 621 轮，说明探索期结束后仍有
    # 提升空间；权重标定后地形单调，需要更多回合让策略充分收敛。
    'n_episodes': 4000,       # 总训练轮数
    'max_steps_per_episode': 500,  # 每轮最大步数

    # 保存和评估频率
    'save_freq': 100,         # 每100轮保存一次模型（轻量，不含回放缓冲）
    'eval_freq': 100,         # 每100轮评估一次

    # ---- best 模型的挑选方式（2026-10-01 修复）----
    # 【坑】原实现按"**单回合最高奖励**"存 best：
    #     if episode_reward > best_reward: save('maddpg_best.pth')
    #   单回合奖励标准差约 300+（SEM≈47），最大值几乎总是"运气最好的一轮"，
    #   而不是"最好的策略"。上一版恰好挑中了真最优，但那是运气。
    # 现在改为按**滑动平均**挑选：窗口 best_window 轮的平均值刷新才存
    # best_avg；同时仍另存一个按单回合最高分的 best_peak 供对照。
    'best_window': 100,       # 滑动平均窗口（轮）
    'best_min_episodes': 50,  # 至少训练这么多轮后才开始存 best（均值才稳定）
    'eval_probe_freq': 50,    # 每多少轮写一次 argmax 探针日志（监控退化）

    # ---- 学习前的预热轮数（2026-10-01 新增，默认 0 = 关闭）----
    # 参考实现的做法：
    #   TD3 (sfujim/TD3)       : start_timesteps = 25e3 步纯随机采集，之后才开始 learn
    #   MADDPG (openai/maddpg) : buffer 不足 batch_size × max_episode_len 时直接 return
    # 本项目原来只要求 buffer 里够一个 batch（约 1 轮）就开始学，critic 早期只能
    # 拟合随机初始化的 target，噪声较大。设为 20~30 可复现参考实现的"先攒数据再学"。
    # 默认 0 以保持与本次实验一致的既有行为；想试参考做法就改成 30。
    'learn_start_episodes': 0,

    # 断点续训：每 N 轮保存一次**完整训练状态**（含回放缓冲），
    # 用于中途停止后继续训练。中断时最多丢 50 轮（约 5 分钟）。
    # 文件较大（每 1 万条转移约 30MB，缓冲上限 50 万条约 1.5GB），
    # 故不宜过于频繁；若磁盘紧张可调大该值。
    'resume_freq': 50,
    'resume_file': './models_v2/maddpg_resume.pth',

    # 早停参数
    'early_stop_patience': 500,   # 早停耐心值
    'improvement_threshold': 0.01, # 提升阈值

    # 设备
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    # 随机种子
    'seed': 42,

    # 日志与模型目录
    #
    # 【重要】2026-10-01 起默认输出目录从 models_matd3 / logs_matd3 改为
    # models_v2 / logs_v2，原因是：从零开始的训练只要在某一轮超过
    # 3851.00 就会覆盖 models_matd3/maddpg_best.pth（现有最优模型）。
    # 换目录可以保证旧模型、旧日志一根毫毛都不动。
    # 如需写回旧目录，用命令行显式指定：--model-dir ./models_matd3 --log-dir ./logs_matd3
    'log_dir': './logs_v2',      # 日志保存目录
    'log_file': './logs_v2/training_output.log',  # 训练日志文件（供 visualize_training.py 解析）
    'model_dir': './models_v2',  # 模型保存目录
}

# ==================== 评估参数 ====================
EVAL_CONFIG = {
    # 评估轮数
    'n_eval_episodes': 50,
    
    # 对比策略
    'compare_with': {
        'uniform_open': True,     # 均匀开放（所有段开放程度相同）
        'uniform_close': True,    # 均匀关闭
        'random': True,           # 随机策略
        'greedy': True,           # 贪心策略（哪段堵开哪段）
    },
    
    # 可视化
    'save_results': True,
}

# ==================== 辅助函数 ====================
def get_complete_config():
    """获取完整配置字典"""
    return {
        'ca': CA_CONFIG,
        'env': ENV_CONFIG,
        'agent': AGENT_CONFIG,
        'train': TRAIN_CONFIG,
        'eval': EVAL_CONFIG,
    }

def print_config():
    """打印当前配置"""
    config = get_complete_config()
    for section, params in config.items():
        print(f"\n{'='*50}")
        print(f"{section.upper()} CONFIG")
        print(f"{'='*50}")
        for key, value in params.items():
            print(f"  {key}: {value}")

if __name__ == '__main__':
    print_config()
