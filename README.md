# 高速公路应急车道元胞自动机仿真模型

## 模型概述

本模型基于元胞自动机（Cellular Automaton, CA）理论，模拟高速公路上应急车辆对社会车辆交通流的影响。模型考虑了三车道场景（超车道、行车道、应急车道），并实现了应急车道的动态开放机制。

> **本 README 描述的是修复后的模型。**
> 修复前存在一个致命缺陷：`env_maddpg` 挂载的应急车道准入回调在 `high.py` 中
> 从未被调用（死代码），且 CA 硬性禁止社会车辆进入应急车道，
> 导致**控制动作对交通流没有任何影响**。
> 详见 `模型诊断报告.md`。

---

## 零、2026-10-01 训练策略修复（**先读这一节**）

上一轮修复解决了"动作没效果""奖励地形非单调""熵坍缩"三个问题，
但训练**到第 600 轮之后就不再改善、第 1000 轮之后明显退化**
（argmax 策略奖励从 ~3350 掉到 ~2920，与 Always Close 持平）。

本次定位到根因并修复，**奖励函数一个字都没改**。完整对照见
`analysis/训练策略对比与改进.md`（含 openai/maddpg、oxwhirl/pymarl、sfujim/TD3 的源码引用）。

### 根因：Actor 更新时把"软概率向量"喂给了 Critic

```python
# 修复前：5 个 Actor 的软概率拼成联合动作
joint = torch.stack([self.actors[j](obs[:, j, :]) for j in range(n)], dim=1)
q = self.critics[i](obs_flat, joint.reshape(bs, -1)).mean()
```

但 critic 的**训练**用的是 `F.one_hot(action_tensor)` —— **独热动作**。
critic 从未见过软概率向量，用它去评价软向量等于在未拟合的输入点上取梯度，
**actor 实际上学不到东西**：策略概率长期贴在 0.5 附近
（实测 27.4% 的决策点 |p−0.5| < 0.05），argmax 只是在读噪声。

对照 openai/maddpg 原作者实现，它写的是 `act_input_n[p_index] = act_pd.sample()`，
**始终在真实动作上求值**。

### 本次改动清单

| # | 改动 | 文件 |
|---|---|---|
| 1 | Actor 更新改为**反事实期望 Q**：其余智能体保持本批次实际执行的独热动作，只把第 i 个动作分别设为 0/1，得 Q(s,a_i=0)、Q(s,a_i=1)，按策略概率加权 | `maddpg_v2.py` |
| 2 | 目标 Q 改为**独热 argmax + 离散版目标策略平滑**（`target_flip_p=0.1`），不再把软向量喂给 target critic | `maddpg_v2.py`、`config_maddpg.py` |
| 3 | **熵系数线性退火** 1.0→0.05（800 轮），不再固定 5.0 | `config_maddpg.py` |
| 4 | **ε 改线性衰减** 1.0→0.05（800 轮），替代指数 0.998 | `config_maddpg.py` |
| 5 | **best 按滑动平均挑选**（`maddpg_best_avg.pth`），单回合最高分另存 `maddpg_best_peak.pth` | `maddpg_v2.py` |
| 6 | 训练日志新增 `ArgmaxOpen / Entropy / EntCoef`，**策略退化当场可见** | `maddpg_v2.py` |
| 7 | 输出目录默认 `models_v2` / `logs_v2`，且**不再写 `maddpg_best.pth`** —— 旧最优模型零风险 | `config_maddpg.py` |

### 同时修掉的 4 个评估/监控坑

| 坑 | 修复 |
|---|---|
| `select_action(evaluate=True)` **只关 ε，仍按概率采样**，没有确定性评估路径（历史"MADDPG 排名第 3"的由来） | 新增 `select_action(..., deterministic=True)`；`evaluate.py` 默认走它 |
| `evaluate.py` 各策略**各自随机场景**，不可比 | 默认**配对**：同一 seed 列表，策略调用期间冻结全局随机流 |
| 指标是**回合末单帧快照** | 改为**整回合逐决策步平均** |
| `tools/check_policy_entropy.py` 只看采样熵，**看不到 argmax 退化** | 三层判据：熵 → argmax 开放率 → **与恒关闭/恒开放做配对对照** |

### 验证

```cmd
:: 单元测试：复现旧更新的失明 + 验证新更新有效
.venv\Scripts\python.exe analysis\test_actor_update.py
::   旧更新: p(open) 0.5177 -> 0.5177（零梯度）   新更新: p(open) 0.5177 -> 0.9913

:: 健康度检查（三层），对已知退化检查点会 FAIL
.venv\Scripts\python.exe tools\check_policy_entropy.py --all --model-dir models_v2 --quick-episodes 6
```

### ⚠️ 实测结论：**这次改动没有真正修好"退化"，请如实看待**

我按上面的分析跑了三轮对照训练（runA 反事实+熵退火 / runB 旧actor+熵退火 / runC 反事实+常数熵0.8），
逐检查点做确定性 argmax 筛查。**结论如下（详细数据见 `analysis/训练策略对比与改进.md` §7）**：

| 结果 | 说明 |
|---|---|
| ✅ **反事实 actor 更新值得保留** | runB（旧更新）在 ep200 之后**永久塌到 2920–2950 再没恢复**；runA 撑到 ep300 才回落。合并 64 个配对场景，旧更新比旧最优**显著差 459**（p<0.001） |
| ⚠️ **熵系数退火是错的假设** | 我专门跑的 runC（常数 0.8）在**每个检查点**上都比 runA 更差，且 ep200–300 时熵系数还**更高** → "熵越低越差"只是相关，不是因果 |
| ❌ **没有做出统计上明确更优的模型** | 最好的是 `models_v3_a/a_maddpg_episode_200.pth`，两个独立种子区间都优于旧最优（+28 / +108），合并 **+78.4，95% CI [−30.9, +187.7]，不显著** |
| 🔑 **真正的收获是"怎么挑检查点"** | 四轮训练的所有检查点散落在 **2920–3382**（±400 波动），**远大于**调参带来的差异。按旧口径会挑到 runA 的 ep800（2923），按新的 argmax 筛查挑出 ep200（**3382**）—— **差距 +459，比任何调参收益都大** |

**下一步该做的**：把 argmax 逐检查点筛查固定成流程；多随机种子训练（单条曲线不足以支撑结论）；
评估场景数从 64 提到 150+；若要继续改算法，优先 twin critic + min(Q1,Q2)，
**不要再调熵系数**（两个方向都已实测失败）。

**备份**：本次修改前的原文件在 `_backup_20261001/`（含 SHA-256 清单 `MANIFEST.txt`），
可逐字节还原。奖励函数（`ENV_CONFIG`，含 `reward_weights`）经 AST 解析比对，**与备份完全一致**。

---

## 零之二、2026-10-01 第二轮修改（twin critic + 多种子 + 筛查流水线）

第一轮的实测结论是"**没有做出统计上显著更优的模型**，而且**检查点之间的波动（±400）
远大于调参收益**"。第二轮据此落实了上面那四条建议，奖励函数**依旧一字未动**：

| # | 改动 | 文件 |
|---|---|---|
| 1 | **Twin critic（TD3 clipped double-Q）**：每智能体两套 critic，目标 Q 取 `min(Q1,Q2)`，抑制 Q 过估计 | `maddpg_v2.py`、`config_maddpg.py`（`twin_critics: True`） |
| 2 | **多随机种子训练启动器** | `run_multiseed.py`（新增） |
| 3 | **检查点筛查与挑选流水线** —— 最有价值的一条：旧口径会挑到 2923 的那一版，筛查挑出 3382，**差 +459** | `tools/screen_run.py`（新增） |
| 4 | **评估场景数默认 50/60 → 160** | `evaluate.py`、`analysis/eval_paired.py`、`analysis/compare_models.py` |

> TD3 的三件套是 **twin critic + 目标策略平滑 + 延迟策略更新**；
> 本项目后两件已有（`policy_freq=2`、`target_flip_p=0.10`），所以**不是"只上双 critic"**。

**兼容性**（`analysis/test_twin_critic.py`，4/4 PASS）：
旧检查点在新代码下照常加载（用第一套 critic 热启动第二套）；
**纯评估结果逐位不变**（实测 `max|Δp| = 0.00e+00`，argmax 动作完全相同）。

**训练与评估的完整命令见 [`训练与评估流程.md`](训练与评估流程.md)。**

---

## 零之三、2026-10-02 v5 训练结果：**twin critic 是负收益（已改回默认关闭）**

跑了 5 个种子 × 700 轮（全部 rc=0，35 个检查点），逐检查点确定性 argmax 筛查 +
160 个配对场景头对头：

| 模型 | 160 配对场景 | 相对旧最优 |
|---|---|---|
| **runA_ep200（无 twin，当前最优）** | **3473.81** | +34.3（不显著） |
| 旧最优 ep664 | 3439.56 | 基准 |
| v5 次佳 seed3 ep500（有 twin） | 3393.14 | **−46.4（显著更差）** |
| v5 最佳 seed3 ep100（有 twin） | 3392.69 | **−46.9（显著更差）** |
| AlwaysOpen / AlwaysClose 锚点 | 3255.21 / 2983.52 | 显著更差 |

**结论**：`config_maddpg.AGENT_CONFIG['twin_critics']` 已改回 **`False`**（代码保留，
`--twin-critics` 可随时启用）。合并全部 **224 个配对场景**，`runA_ep200` 比旧最优
**+46.9，95% CI [−11.1, +104.8]** —— 三个独立区间方向一致为正，但**仍不显著**。

**最重要的发现是随机种子的影响远大于算法**：
同配置下 **seed3 的 7 个检查点全部健康（最高 3325）、seed4 的 7 个全部退化到 ≈2920**
（`AlwaysClose` 水平），相差 400 分；而所有算法改动带来的差异都在 ±50 量级。
35 个检查点里 **21 个（60%）是退化的** —— 这正是 `tools/screen_run.py` 筛查流水线的价值所在。

详见 [`analysis/v5训练结果与结论.md`](analysis/v5训练结果与结论.md)。

---

## 一、道路与车辆参数

以下为 `config_maddpg.py` 中 `CA_CONFIG` 的实际取值。

### 道路参数

| 参数 | 值 | 说明 |
|-----|-----|------|
| L | 1000 | 道路长度（元胞数），环形边界 |
| n_lanes | 3 | 车道数（lane 0 超车道、lane 1 行车道、lane 2 应急车道） |

### 车辆参数

| 参数 | 值 | 说明 |
|-----|-----|------|
| n_cars_range | (450, 600) | 社会车辆数**每轮随机采样** |
| v_max | 5 | 车辆最大速度（元胞/时间步） |
| v_initial_range | (2, 5) | 社会车辆初始速度范围 |

> **关于车辆数**：原配置为 (200, 350)，实测密度仅 0.10~0.175，
> 社会车辆均速 4.96~5.00，**远高于拥堵阈值，导致"拥堵"场景从不出现**。
> 经 `tools/sweep_density.py` 实测，均速在 n_cars≈400 处跌破 60% 阈值、
> n_cars≈650 处跌破 40% 阈值，故调整为 (450, 600)。
> 容量约束：`n_cars + max_emergency` 不得超过 L（`high.py` 会显式校验并报错）。

### 应急车参数

| 参数 | 值 | 说明 |
|-----|-----|------|
| D | 40 | 警笛影响范围（元胞数） |
| emergency_yield_distance | 60 | 让行触发距离（≥ D），见第三节 |
| P_emergency_range | (0.3, 0.5) | 应急车出现概率范围 |
| max_emergency | 3 | 最多应急车数量 |
| emergency_min_acceptable_speed | 2.0 | 验收用：应急车速度可接受下界 |

### 行为参数

| 参数 | 值 | 说明 |
|-----|-----|------|
| P1 | 0.6 | 无应急车时社会车辆换道概率 |
| P2 | 0.9 | 有应急车时被标记车辆（g=1）换道概率 |
| dev | 2 | 避让车辆换道时允许的最小安全距离 |
| warm_up_steps | 100 | 每轮预热的仿真步数 |

---

## 二、车辆状态定义

每辆车具有以下状态：
- **lane**: 所在车道（0, 1, 2）
- **pos**: 位置（0 到 L-1）
- **vel**: 当前速度
- **g**: 是否应该避让标记（bool）
- **b**: 刹车状态（bool）
- **a**: 加速状态（bool）

---

## 三、应急车道准入机制（修复重点）

### 3.1 准入判定链

社会车辆能否进入应急车道，由**两级判定**决定：

```
_can_change_lane(vid, dir)
   └─ target_lane == 2 且本车非应急车
        └─ self._can_enter_emergency_lane(vid)      ← 实例属性，env 可覆盖
              ├─ [env 覆盖层] 分段闸门：本段关闭 -> False
              └─ _emergency_lane_default(vid)       ← CA 自身的物理规则
                    └─ 应急车在后方 yield_distance 内 -> False
```

- `_can_enter_emergency_lane` 在 `__init__` 中绑定为实例属性，
  `env_maddpg._install_emergency_lane_gate()` 用同名属性覆盖它。
- `high.py` 单独运行时使用默认实现（只看应急车距离，无分段概念）。
- 闸门在 `__init__` / `reset()` / `step()` 三处都会重新挂载，
  避免 `reset()` 重建 CA 对象后出现"不受控空窗"。

### 3.2 让行机制

| 场景 | 行为 |
|-----|------|
| 应急车在应急车道后方 `yield_distance` 内 | 禁止社会车辆进入该车道 |
| 社会车辆已在应急车道、应急车即将到达 | 安全时以 P2 概率左移让行（每步重试） |
| 应急车 | 恒可通行，不受闸门限制；速度 `min(v_max, gap)` |

实测（`tools/check_emergency_yield.py`，n_cars=520，P_emergency=0.8）：

| 策略 | 应急车均速 | 最长连续停滞 |
|---|---|---|
| 全关闭 | 5.000 | 0 步 |
| 全开放 | 3.234 | 3 步 |

应急车在开放时会被共道社会车辆**短暂**减速，但不会长期卡死。

---

## 四、无应急车情况的规则

### 1. 加速规则

```
v(t+1) = min(v(t) + 1, v_max)
```
**条件**：前方安全距离内没有车，或前车距离 > v(t) + 1

### 2. 刹车规则

当加速后会撞到前车时，立即刹车：

```
v(t+1) = min(v(t+1), d(t), v_front)
```

### 3. 变道规则

车辆必须同时满足以下条件才能变道：

| 条件 | 说明 |
|-----|------|
| ① 匀速状态 | 不是加速状态（a = False）且不是刹车状态（b = False） |
| ② 受阻 | d(t) < v(t)，即前车距离小于当前速度 |
| ③ 能变道 | 目标车道安全（含应急车道准入判定） |
| ④ 想变道 | rand() < P1 |

**安全变道条件**：
- 目标车道前车距离 ≥ 本车当前速度
- 目标车道后车距离 ≥ 目标车道后车当前速度

**变道方向选择**：优先向左（超车），其次向右。

---

## 五、有应急车情况的规则

### 应急车特性
- 固定在最右侧车道（lane 2），**不变道**
- 目标速度 v_max = 5，实际速度受前车距离约束：`min(v_max, gap)`
- 前方 D = 40 距离内的车辆会被标记避让

### 1. 普通社会车辆（非应急车前车，g=0）
与无应急车情况相同。

### 2. 应急车前车（需要避让）

#### 第一步：尝试直接变道
必须满足以下4个条件：

| 条件 | 说明 |
|-----|------|
| ① 匀速状态 | a = False 且 b = False |
| ② 受阻 | d(t) < v(t) |
| ③ steer = 1 | 满足 steer 规则 |
| ④ rand() < P2 | 换道概率 |

**steer 规则**（同时满足）：
- 1 ≤ d_current ≤ v_current
- v_current ≤ d_target_front
- 2 ≤ d_target_back
- d_target_back ≤ v_target_back

#### 第二步：若不能直接变道，标记左侧车道相邻车 g=1

### 3. 被标记车辆（g=1）
- 可以加速
- 使用 steer 规则检查安全性
- 换道概率 P2 = 0.9

---

## 六、多智能体控制（MADDPG）

### 6.1 环境建模

| 项 | 值 |
|-----|-----|
| n_segments / n_agents | 5（每段 200 元胞） |
| 动作空间 | Discrete(2)，0=该段应急车道关闭，1=开放 |
| 观测维度 | obs_dim = 45 = (1+2×2) 段 × 8 + 5 历史动作 |
| 决策间隔 | decision_interval = 3（每 3 个 CA 步决策一次） |
| 每轮决策步数 | max_steps / decision_interval = 500/3 ≈ 167 |

每段观测 8 维：3 车道密度 + 3 车道速度 + 应急车距离 + 应急车速度。

### 6.2 奖励函数

**设计原则：奖励由动作产生的实际后果决定，而非由动作本身决定。**

修复前含多项"只要 action==1 就无条件加分"的常量项（如
`emergency_lane_capacity_bonus`、`emergency_vehicle_open_reward`），
等价于给常量加分，智能体不做任何有效控制也能刷高奖励。现改为：

| 分项 | 权重 | 含义 |
|-----|-----|------|
| `efficiency` | 20.0 × 全局车速 + 2.0 × 本段车速 | 通行效率（**主导权衡项**） |
| `efficiency`（拥堵时追加） | +20.0 × 全局车速 − 2.0 | 拥堵缓解收益 |
| `speed_gain` | 40.0 × 车速改善量 | 直接度量"开放是否真的疏解了拥堵" |
| `emergency_speed` | 4.0 × 本段应急车归一化速度 | 保障应急通行（约束项） |
| `resource` | +1.0 / −1.0 | 不需要开放时，关闭/开放的小额提示 |
| `control` | −0.05 / −0.1 | 动作切换与相邻段不一致的控制代价 |

> **权重为何这样定**：`emergency_speed` 的权重必须压得比通行效率低。
> 因为"关闭"时应急车恒为 v_max=5，该项在关闭状态下近似一个常数，
> 权重过高（早期为 15.0）会压倒真正的任务奖励，使最优动作在密度区间内
> **非单调翻转**，训练结果退化为"恒关闭"、应急车道永远用不上。
> 降到 4.0 后它只承担"别把应急车堵死"的约束作用。

其中拥堵判定使用 `_local_congestion_speed()`（只统计社会车道），
**不包含空着的应急车道**——否则关闭时应急车道恒为空、被赋值为"畅通"，
会把本段平均速度抬高约 1/3，使"需要开放"永不成立，反而惩罚开放。

**该奖励塑造出状态依赖的决策问题**（`tools/check_policy_state_dependence.py` 实测）：

| 车辆数 | 关闭奖励 | 开放奖励 | Δ(开−关) | 最优动作 |
|---|---|---|---|---|
| 450 | 17589 | 13670 | −3920 | CLOSE |
| 480 | 16318 | 14168 | −2149 | CLOSE |
| 520 | 14109 | 18462 | **+4353** | **OPEN** |
| 560 | 15466 | 17739 | **+2273** | **OPEN** |
| 600 | 13002 | 16975 | **+3972** | **OPEN** |

该判定对应急车概率**完全稳健**：P_em = 0.3 / 0.4 / 0.5 三档在每个密度下结论一致。

低密度时开放虽提升社会车速度，但应急车被拖慢的代价更大，故应关闭；
高密度严重拥堵时，开放带来的疏解收益超过应急车代价，故应开放。
**智能体必须根据状态决策，这正是 MADDPG 的价值所在。**

### 6.3 探索机制与策略熵（关键）

奖励地形正确并不足以学出正确策略，还必须保证**策略不会过早锁死**。

**症状**：早期版本训练到第 400 轮时，5 个智能体的策略熵全部降为 0
（`ln2 ≈ 0.693` 是上限），即退化为确定性策略"恒关闭"，
随后 `Avg` 从峰值 3569 一路跌到 3246。

**成因与对策**：

| 问题 | 原因 | 对策 |
|---|---|---|
| 熵坍缩 | 熵系数 0.02，熵项仅 0.014，被 Q（量级约 400）完全压倒 | `entropy_coef` → **5.0**，与 Q 同量级 |
| 探索过早停止 | ε 每回合 ×0.995，780 轮即到下限 0.02 | `epsilon_decay` → **0.998**，约 2000 轮到下限 |
| 策略过早确定 | Actor 学习率 1e-4 偏高 | `actor_lr` → **3e-5** |

熵系数的定标是实测的（200 步梯度更新后的策略熵）：

| entropy_coef | 0.02 | 0.5 | 2.0 | **5.0** | 10.0 |
|---|---|---|---|---|---|
| 更新后熵 | 0.613 | 0.599 | 0.644 | **0.683** | 0.691 |

（上限 0.693）取 5.0 时策略保持接近最大随机性，同时不压制策略改进。

> 注意：熵坍缩是**渐进**的——在只跑一两百轮时看不出系数差异（ε≈1 时
> 随机探索本身就维持了高观测熵），它发生在训练数百轮、策略开始锁定
> 状态分布之后。因此验证必须跑到至少 500 轮。

---

## 七、环境准备与运行方法

### 7.1 首次配置环境

项目自带一键配置脚本（创建 `.venv`、装 torch CUDA 12.8 与全部依赖并校验）：

```powershell
powershell -ExecutionPolicy Bypass -File setup_env.ps1
```

若没有 NVIDIA 显卡，用 CPU 版以减少下载体积：

```powershell
powershell -ExecutionPolicy Bypass -File setup_env.ps1 -CpuOnly
```

手动安装等价于：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install numpy matplotlib gymnasium pandas seaborn
```

> **RTX 50 系用户注意**：Blackwell 架构（sm_120）必须使用 CUDA 12.8
> 及以上的 torch 轮子，PyPI 默认的 torch 是 CPU 版。

### 7.2 训练

```powershell
.\.venv\Scripts\Activate.ps1

# 冒烟测试：20 轮，先确认流程能跑通
python run_training.py --episodes 20 --seed 42

# 正式训练
python run_training.py --seed 42
```

训练日志同时写入 `logs_matd3/training_output.log`，可在另一个终端查看：

```powershell
Get-Content .\logs_matd3\training_output.log -Wait
```

### 7.3 评估与出图

```powershell
python evaluate.py 50                                          # 公平评估
python plot_strategy_comparison.py --input evaluation_results.json   # 真实数据出图
python visualize_training.py                                   # 训练曲线
python training_results_table.py                               # 数据表格与 HTML 报告
python high.py                                                 # CA 示例
```

### 7.4 回归测试

```bash
python tools/run_tests.py              # 全部用例
python tools/run_tests.py --skip-slow  # 跳过耗时的密度扫描
```

覆盖的缺陷：动作-仿真解耦、拥堵场景缺失、奖励与后果脱钩、
应急车被堵死、无状态依赖性。

### 诊断工具（均为只读）

| 脚本 | 用途 |
|-----|------|
| `tools/diagnose_action_effect.py` | 验证开/关是否真的改变交通流 |
| `tools/sweep_density.py` | 密度-速度曲线，标定 n_cars_range |
| `tools/check_reward_action_coupling.py` | 验证奖励与动作后果耦合 |
| `tools/check_emergency_yield.py` | 验证应急车不被堵死 |
| `tools/check_policy_state_dependence.py` | 验证最优动作随状态翻转 |

---

## 八、依赖说明

| 依赖 | 用途 | 是否必需 |
|-----|-----|---------|
| numpy | 全部计算 | 必需 |
| matplotlib | 绘图 | 绘图时必需 |
| torch | MADDPG 训练/评估 | 仅训练与 MADDPG 评估需要 |
| gymnasium | 环境基类 | 训练与 `env_maddpg` 需要 |
| pandas / seaborn | 结果表格 | 仅 `training_results_table.py` 需要 |

一键安装见 7.1 的 `setup_env.ps1`。

> `evaluate.py` 已改为**延迟导入** torch：没有 torch 时仍可评估 5 个基线策略，
> 只有需要 MADDPG 模型时才要求安装 torch。
>
> `tools/` 下的全部诊断脚本仅依赖 numpy，可在未装 torch 的环境下运行
> （gymnasium 缺失时会注入内置最小桩，并在输出中明确标注这一限制）。

---

## 九、图表说明

### 第一组图：有应急车场景（时间步 0-200）

**图1 - 速度曲线和换道次数**：左图为社会车辆与应急车辆平均速度；
右图为每步换道次数。

**图2 - 车道密度和平均速度**：左图为各车道车辆数，右图为各车道平均速度。

### 第二组图：无应急车场景（时间步 0-150）

**图3 / 图4** 同上，作为基准对比组。

### 策略对比图

`strategy_comparison_real.png` —— 由 `plot_strategy_comparison.py`
读取真实 `evaluation_results.json` 生成，**推荐使用**。

> 历史说明：项目早期存在 `strategy_comparison.py`，其数据来自源码中写死的常量，
> 与模型实测结果不一致（连策略排名都不同），并产出了
> `strategy_comparison.png` 与 `strategy_radar.png`。
> 这三个文件已随旧模型一并删除，避免在材料中误用。
