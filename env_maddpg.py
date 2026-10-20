"""
MADDPG环境 - 多智能体分段应急车道控制

将道路分成n_segments段，每段由一个智能体控制
每个智能体的动作是离散值：0或1（该段应急车道的开/关）
"""

import numpy as np
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from high import EmergencyVehicleCA
from config_maddpg import CA_CONFIG, ENV_CONFIG


class MultiAgentEmergencyLaneEnv(gym.Env):
    """
    多智能体应急车道控制环境
    
    状态空间（每个智能体）：
        - 本段及相邻段的密度和速度
        - 应急车位置信息
        - 历史状态
    
    动作空间（每个智能体）：
        - 连续值 0.0~1.0（该段应急车道的开放程度）
    
    奖励：
        - 全局奖励（所有智能体共享）：整体交通效率
        - 局部奖励（每个智能体独立）：本段交通状况
        - 协调奖励：相邻智能体动作一致性
    """
    
    def __init__(self, ca_config=None, env_config=None):
        super().__init__()
        
        # 加载配置
        self.ca_config = ca_config if ca_config else CA_CONFIG
        self.env_config = env_config if env_config else ENV_CONFIG
        
        # 分段参数
        self.n_segments = self.env_config['n_segments']
        self.n_agents = self.env_config['n_agents']
        self.segment_length = self.ca_config['L'] // self.n_segments
        
        # 初始化CA模型
        self.ca = None
        self._reset_ca()
        
        # 每个智能体的观察范围
        self.local_range = self.env_config['local_observation_range']
        
        # 计算每个智能体的状态维度
        # 每段的状态：密度(3车道) + 速度(3车道) + 应急车信息(2) = 8
        # 观察范围：本段 + 左右各local_range段
        obs_segments = 1 + 2 * self.local_range
        self.obs_dim = obs_segments * 8 + self.env_config['state_history']
        
        # 动作维度（离散）
        self.n_actions = self.env_config['n_actions']
        
        # 定义动作空间（离散）
        self.action_space = spaces.Discrete(self.n_actions)
        
        # 定义观察空间
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32
        )
        
        # 历史状态缓存
        self.state_history = []
        self.action_history = []
        
        # 当前步数
        self.current_step = 0
        self.decision_step = 0

        # 上一决策步的全网归一化平均车速，用于计算"车速改善"奖励项
        self.prev_global_speed = None

        # 初始闸门：全部关闭（在 reset() 后会被实际动作覆盖）
        self._install_emergency_lane_gate(np.zeros(self.n_agents, dtype=int))
        
    def _reset_ca(self):
        """重置CA模型 - 每轮随机采样车辆数和应急车概率"""
        # 从范围中随机采样车辆数
        n_cars = np.random.randint(
            self.ca_config['n_cars_range'][0],
            self.ca_config['n_cars_range'][1] + 1
        )
        
        # 从范围中随机采样应急车出现概率
        P_emergency = np.random.uniform(
            self.ca_config['P_emergency_range'][0],
            self.ca_config['P_emergency_range'][1]
        )
        
        # 保存当前轮次的参数（用于日志）
        self.current_params = {
            'n_cars': n_cars,
            'P_emergency': P_emergency,
            'P1': self.ca_config['P1'],
            'P2': self.ca_config['P2'],
            'dev': self.ca_config['dev'],
            'warm_up_steps': self.ca_config['warm_up_steps']
        }
        
        # 创建CA模型
        self.ca = EmergencyVehicleCA(
            L=self.ca_config['L'],
            n_lanes=self.ca_config['n_lanes'],
            n_cars=n_cars,
            v_max=self.ca_config['v_max'],
            D=self.ca_config['D'],
            P1=self.ca_config['P1'],
            P2=self.ca_config['P2'],
            dev=self.ca_config['dev'],
            emergency_prob=P_emergency,
            max_emergency=self.ca_config['max_emergency'],
            v_init_min=self.ca_config['v_initial_range'][0],
            v_init_max=self.ca_config['v_initial_range'][1],
            yield_distance=self.ca_config.get('emergency_yield_distance', self.ca_config['D']),
        )
        # 准入闸门通过 CA.segment_of() 选择智能体；重建 CA 后必须恢复分段信息。
        self.ca.segment_length = self.segment_length
        self.ca.n_segments = self.n_segments
    
    def _get_segment_info(self, segment_id):
        """
        获取指定段的信息
        返回: [密度(3车道), 速度(3车道), 应急车距离, 应急车速度]
        """
        start_pos = segment_id * self.segment_length
        end_pos = (segment_id + 1) * self.segment_length
        
        # 初始化
        lane_densities = np.zeros(self.ca_config['n_lanes'])
        lane_speeds = np.zeros(self.ca_config['n_lanes'])
        lane_counts = np.zeros(self.ca_config['n_lanes'])
        
        # 统计该段内的车辆（确保不超过数组长度）
        total_vehicles = min(self.ca.total_vehicles, len(self.ca.pos))
        for vid in range(total_vehicles):
            pos = self.ca.pos[vid]
            # 考虑环形道路
            if start_pos <= pos < end_pos or (start_pos > end_pos and (pos >= start_pos or pos < end_pos)):
                lane = self.ca.lane[vid]
                lane_densities[lane] += 1
                lane_speeds[lane] += self.ca.vel[vid]
                lane_counts[lane] += 1
        
        # 计算平均速度
        for l in range(self.ca_config['n_lanes']):
            if lane_counts[l] > 0:
                lane_speeds[l] = lane_speeds[l] / lane_counts[l] / self.ca_config['v_max']
            else:
                lane_speeds[l] = 1.0  # 空车道视为畅通
        
        # 归一化密度
        max_cars_per_segment = self.ca_config['n_cars_range'][1] // self.n_segments + 1
        lane_densities = lane_densities / max_cars_per_segment
        lane_densities = np.clip(lane_densities, 0, 1)
        
        # 应急车信息
        em_distance = self.segment_length  # 默认最远
        em_speed = 0
        
        if self.ca.n_emergency > 0:
            for em_id in self.ca.emergency_ids:
                em_pos = self.ca.pos[em_id]
                em_lane = self.ca.lane[em_id]
                
                # 检查应急车是否在该段或附近
                if em_lane == self.ca_config['n_lanes'] - 1:  # 应急车道
                    # 计算距离
                    if start_pos <= em_pos < end_pos:
                        em_distance = 0
                        em_speed = self.ca.vel[em_id] / self.ca_config['v_max']
                    else:
                        # 计算到该段的距离
                        dist_to_start = (em_pos - start_pos) % self.ca_config['L']
                        if dist_to_start < em_distance:
                            em_distance = dist_to_start
                            em_speed = self.ca.vel[em_id] / self.ca_config['v_max']
        
        em_distance = em_distance / self.segment_length  # 归一化
        em_distance = min(em_distance, 1.0)
        
        return np.concatenate([lane_densities, lane_speeds, [em_distance, em_speed]])

    def _local_congestion_speed(self, segment_id):
        """本段**社会车辆可用车道**（lane 0/1）的平均速度，归一化到 [0,1]。

        为什么不直接用 _get_segment_info 的速度：
            该方法把空车道视为畅通（赋 1.0）。而应急车道在关闭时**必然为空**，
            于是本段平均速度会被恒定抬高约 1/3，导致
                local_avg_speed < congestion_threshold
            几乎永不成立，"本段需要开放"判断失真——奖励反而会惩罚开放。
        因此拥堵判定单独使用本方法，只统计社会车辆实际占用的车道。
        """
        n_lanes = self.ca_config['n_lanes']
        v_max = self.ca_config['v_max']
        start_pos = segment_id * self.segment_length
        end_pos = (segment_id + 1) * self.segment_length

        speeds = []
        total = min(self.ca.total_vehicles, len(self.ca.pos))
        for vid in range(total):
            pos = self.ca.pos[vid]
            if start_pos <= pos < end_pos:
                lane = self.ca.lane[vid]
                # 排除应急车道，避免"关闭导致空车道"污染拥堵判定
                if lane < n_lanes - 1:
                    speeds.append(self.ca.vel[vid])

        if not speeds:
            # 本段社会车道无车 -> 视为畅通
            return 1.0
        return float(np.mean(speeds)) / v_max
    
    def _get_observation(self, agent_id):
        """
        获取指定智能体的观察（局部观察）
        """
        obs_list = []
        
        # 获取本段及相邻段的信息
        for offset in range(-self.local_range, self.local_range + 1):
            segment_id = (agent_id + offset) % self.n_segments
            segment_info = self._get_segment_info(segment_id)
            obs_list.append(segment_info)
        
        # 添加历史动作信息
        if len(self.action_history) > 0:
            recent_actions = self.action_history[-self.env_config['state_history']:]
            # 填充
            while len(recent_actions) < self.env_config['state_history']:
                recent_actions.insert(0, np.zeros(self.n_agents))
            recent_actions_flat = np.concatenate(recent_actions)
        else:
            recent_actions_flat = np.zeros(self.env_config['state_history'] * self.n_agents)
        
        # 只保留本智能体的历史动作
        agent_action_history = recent_actions_flat[agent_id::self.n_agents]
        
        obs = np.concatenate([np.concatenate(obs_list), agent_action_history])
        
        return obs.astype(np.float32)
    
    def _install_emergency_lane_gate(self, actions):
        """把"分段开/关"闸门挂载到 CA 模型。

        修复要点：
          1. 原实现只在 _apply_actions() 内挂载，而 reset() 会重建 CA 对象，
             重建后到首次 step() 之间存在空窗；现在 __init__/reset()/step()
             三处都调用本方法，保证闸门始终生效。
          2. 闸门必须**真的被 CA 调用**。high.py 现在在 _can_change_lane()
             中通过 self._can_enter_emergency_lane(...) 调用本回调。
          3. 闸门叠加在 CA 自身规则之上：分段开放只是"允许"，是否真能进入
             还要看应急车是否在影响范围 D 内（_emergency_lane_default）。

        actions: [n_agents]，元素 0=该段关闭，1=该段开放
        """
        actions = np.asarray(actions)

        def can_enter_emergency_lane(vid, segment_open=True):
            # 应急车不受闸门限制
            if vid >= self.ca.n_cars:
                return True
            segment_id = self.ca.segment_of(self.ca.pos[vid])
            # 分段关闭 -> 一律不允许进入
            if actions[segment_id] == 0:
                return False
            # 分段开放 -> 仍需服从应急车距离规则
            return self.ca._emergency_lane_default(vid)

        self.ca._can_enter_emergency_lane = can_enter_emergency_lane

    def _apply_actions(self, actions):
        """
        应用所有智能体的动作到CA模型（离散动作：0=关闭，1=开放）
        actions: [n_agents] 每个元素的值为0或1，表示该段应急车道的开/关
        """
        self._install_emergency_lane_gate(actions)
    
    def _calculate_rewards(self, actions):
        """计算所有智能体的奖励。

        奖励由动作的实际后果决定，分项见 _reward_parts()。
        返回: [n_agents] 每个智能体的奖励
        """
        rewards, _ = self._reward_parts(actions)
        return rewards

    def _global_speed(self):
        """全网社会车辆归一化平均速度（0~1）。"""
        if self.ca.n_cars <= 0:
            return 1.0
        return float(np.mean(self.ca.vel[:self.ca.n_cars])) / self.ca_config['v_max']

    def _segment_emergency_speed(self, segment_id):
        """本段应急车平均速度（归一化到 [0,1]）；本段无应急车返回 None。"""
        if self.ca.n_emergency == 0:
            return None
        speeds = []
        for em_id in self.ca.emergency_ids:
            if self.ca.segment_of(self.ca.pos[em_id]) == segment_id:
                speeds.append(self.ca.vel[em_id])
        if not speeds:
            return None
        return float(np.mean(speeds)) / self.ca_config['v_max']

    def _reward_parts(self, actions):
        """计算奖励及其分项。

        返回: (rewards[n_agents], parts列表)
              parts[i] 是该智能体的分项字典，便于诊断与测试。

        设计原则：奖励由"动作产生的实际后果"决定。
        与旧版的关键区别：
          - 删除了按动作无条件加分的项（emergency_vehicle_open_reward、
            emergency_lane_capacity_bonus、congestion_open_reward）；
          - 新增 emergency_speed_weight：本段应急车速度越接近 v_max，
            奖励越高。开放应急车道若拖慢了应急车，该项自然下降，
            从而形成"缓解拥堵 vs 保障应急通行"的真实权衡。
        """
        weights = self.env_config['reward_weights']
        rewards = np.zeros(self.n_agents)
        parts = []

        v_max = self.ca_config['v_max']

        # 全局统计（归一化）
        global_avg_speed = self._global_speed()

        congestion_threshold = self.env_config['congestion_threshold'] / v_max
        severe_congestion_threshold = self.env_config['severe_congestion_threshold'] / v_max
        is_congested = global_avg_speed < congestion_threshold
        is_severe_congested = global_avg_speed < severe_congestion_threshold

        # 有应急车的分段
        segments_with_emergency = self.ca.segment_emergency_ids(
            self.segment_length, self.n_segments)

        for agent_id in range(self.n_agents):
            action = int(actions[agent_id])

            segment_info = self._get_segment_info(agent_id)
            # 拥堵判定使用社会车道速度，而非含空应急车道的平均值（见 _local_congestion_speed）
            local_avg_speed = self._local_congestion_speed(agent_id)

            has_emergency = agent_id in segments_with_emergency
            local_need_open = has_emergency or (local_avg_speed < congestion_threshold)

            p = {}

            # ---- 1. 应急车通行后果（核心权衡项）----
            em_speed = self._segment_emergency_speed(agent_id)
            if em_speed is None:
                p['emergency_speed'] = 0.0
            else:
                p['emergency_speed'] = weights['emergency_speed_weight'] * em_speed

            # ---- 2. 资源占用（小权重提示项）----
            p['resource'] = 0.0
            if not local_need_open:
                if action == 0:
                    p['resource'] += weights['correct_close_reward']
                else:
                    p['resource'] += weights['emergency_lane_usage_penalty']

            # ---- 3. 通行效率（后果项）----
            p['efficiency'] = weights['global_avg_speed'] * global_avg_speed
            if is_congested:
                p['efficiency'] += weights['global_congestion_penalty']
                # 拥堵缓解：越堵，车速提升的价值越高。
                p['efficiency'] += weights['congestion_relief_weight'] * global_avg_speed
            p['efficiency'] += weights['local_speed'] * local_avg_speed

            # ---- 3b. 车速改善（直接度量"开放是否真的疏解了拥堵"）----
            # 使用上一决策步到本决策步的全网车速变化量。
            # 这是开放应急车道最主要的正当收益来源，且天然自限：
            # 车速上限为 v_max，不可能靠反复开放无限刷分。
            p['speed_gain'] = 0.0
            if self.prev_global_speed is not None:
                gain = global_avg_speed - self.prev_global_speed
                if gain > 0:
                    p['speed_gain'] = weights['speed_improvement_weight'] * gain

            # ---- 4. 控制代价（作用在动作上，属控制平滑性而非任务奖励）----
            p['control'] = 0.0
            if len(self.action_history) > 0:
                last_action = self.action_history[-1][agent_id]
                p['control'] += weights['local_action_smoothness'] * abs(action - last_action)

            left_agent = (agent_id - 1) % self.n_agents
            right_agent = (agent_id + 1) % self.n_agents
            coord_penalty = (abs(action - actions[left_agent])
                             + abs(action - actions[right_agent])) / 2
            p['control'] += weights['coordination_consistency'] * coord_penalty

            total = sum(p.values())
            rewards[agent_id] = total
            p['total'] = total
            p['has_emergency'] = has_emergency
            p['local_avg_speed'] = local_avg_speed
            parts.append(p)

        return rewards, parts

    def reward_breakdown(self, actions):
        """只读奖励分解：不推进仿真，供诊断与测试使用。"""
        return self._reward_parts(actions)
    
    def reset(self):
        """重置环境"""
        self._reset_ca()
        self.state_history = []
        self.action_history = []
        self.current_step = 0
        self.decision_step = 0

        # CA 对象已被重建，必须重新挂载闸门，否则预热阶段不受控制
        self._install_emergency_lane_gate(np.zeros(self.n_agents, dtype=int))

        # 预热阶段
        for _ in range(self.ca_config['warm_up_steps']):
            self.ca.step()

        # 预热结束后的全网车速作为"车速改善"项的基线
        self.prev_global_speed = self._global_speed()
        
        # 获取所有智能体的初始观察
        observations = [self._get_observation(i) for i in range(self.n_agents)]
        
        return observations
    
    def step(self, actions):
        """
        执行一步环境交互
        actions: [n_agents] 每个智能体的动作
        """
        # 应用动作。历史动作要在奖励计算后写入，否则切换代价会拿当前动作
        # 与自身比较；写入后再生成观测，供下一步看到本次动作。
        self._apply_actions(actions)
        
        # 运行CA仿真多个时间步
        decision_interval = self.env_config['decision_interval']
        for _ in range(decision_interval):
            self.ca.step()
            self.current_step += 1
        
        self.decision_step += 1
        
        # 计算奖励
        rewards = self._calculate_rewards(actions)

        self.action_history.append(actions.copy())
        next_observations = [self._get_observation(i) for i in range(self.n_agents)]

        # 更新"车速改善"基线（供下一决策步使用）
        self.prev_global_speed = self._global_speed()
        
        # 判断终止条件
        done = self.current_step >= self.ca_config['max_steps']
        
        # 额外信息
        info = {
            'episode_step': self.decision_step,
            'total_ca_steps': self.current_step,
        }
        
        return next_observations, rewards, done, info
    
    def render(self, mode='human'):
        """渲染环境"""
        pass
    
    def close(self):
        """关闭环境"""
        pass


if __name__ == '__main__':
    print("Testing MultiAgentEmergencyLaneEnv...")
    env = MultiAgentEmergencyLaneEnv()
    
    obs = env.reset()
    print(f"Number of agents: {env.n_agents}")
    print(f"Observation shape for each agent: {obs[0].shape}")
    
    # 测试一步
    actions = np.random.uniform(0, 1, size=env.n_agents)
    next_obs, rewards, done, info = env.step(actions)
    
    print(f"Actions: {actions}")
    print(f"Rewards: {rewards}")
    print(f"Done: {done}")
    
    print("Environment test completed!")
