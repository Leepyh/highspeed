"""
高速公路元胞自动机模型 - 应急车道版本

模型假设：
1. 应急车只行驶在应急车道（最右侧车道），速度固定，不超车
2. 社会车辆可以在其他车道行驶，遵循元胞自动机规则
3. 应急车出现时，其前方D范围内的社会车辆会避让
4. 只有应急车前方第一辆车会尝试变道避让
5. 如果无法直接变道，会标记右侧车道的车辆，让其加速或变道创造空间
6. 所有车辆同类型，最大速度相同
7. 不考虑随机慢化
8. 只有g=1时可能加速

避让机制：
- 应急车前车首先尝试直接变道避让（向左侧变道）
- 如果无法直接变道，则标记右侧相邻车道的车辆 g=1
- 被标记的车辆会进行加速或变道，为应急车前车创造变道空间
"""

import numpy as np
import random


class EmergencyVehicleCA:
    """
    应急车道元胞自动机模型
    
    车道编号：0=超车道（左），1=行车道（中），2=应急车道（右）
    """
    
    def __init__(self, L, n_lanes, n_cars, v_max, D, P1, P2, dev,
                 emergency_prob=0.3, max_emergency=3, v_init_min=1, v_init_max=None,
                 yield_distance=None):
        """
        参数:
            L: 车道长度（元胞数）
            n_lanes: 车道数（含应急车道）
            n_cars: 社会车辆总数
            v_max: 所有车辆最大速度（元胞/时间步）
            D: 警笛影响范围（元胞数）
            P1: 无应急车时车辆主观想换道的概率
            P2: 有应急车时，被标记车辆（g=1）换道的概率
            dev: 避让车辆换道时允许的最小安全距离
            emergency_prob: 应急车出现的概率（0-1之间）
            max_emergency: 最多应急车数量
            v_init_min: 社会车辆初始速度最小值
            v_init_max: 社会车辆初始速度最大值（默认等于v_max）
            yield_distance: 应急车道让行触发距离；默认取 D
        """
        self.L = L
        self.n_lanes = n_lanes
        self.n_cars = n_cars
        self.v_max = v_max
        self.D = D
        self.P1 = P1
        self.P2 = P2
        self.dev = dev
        self.emergency_prob = emergency_prob
        self.max_emergency = max_emergency
        # 让行触发距离：可大于警笛影响范围 D，使社会车辆更早腾空应急车道
        self.yield_distance = D if yield_distance is None else yield_distance
        # 初始速度收敛到 [1, v_max]：元胞自动机中 v=0 会造成初始死锁，
        # 且 v > v_max 会破坏后续加速/刹车规则的前提。
        self.v_init_min = max(1, min(v_init_min, v_max))
        init_max = v_max if v_init_max is None else v_init_max
        self.v_init_max = max(self.v_init_min, min(init_max, v_max))
        
        # 应急车道准入回调：默认使用本类规则，env_maddpg 会用同名实例属性覆盖
        self._can_enter_emergency_lane = self._can_enter_emergency_lane_impl
        
        # 初始化
        self.reset()
    
    def reset(self):
        """初始化仿真状态"""
        # 道路状态：-1表示空，>=0表示车辆ID
        self.road = np.full((self.n_lanes, self.L), -1, dtype=int)
        
        # 车辆属性
        self.pos = np.zeros(self.n_cars + self.max_emergency, dtype=int)
        self.vel = np.zeros(self.n_cars + self.max_emergency, dtype=int)
        self.lane = np.zeros(self.n_cars + self.max_emergency, dtype=int)
        self.g = np.zeros(self.n_cars + self.max_emergency, dtype=int)  # 避让标记
        self.has_emergency = np.zeros(self.n_cars + self.max_emergency, dtype=bool)
        
        # 统计信息
        self.n_emergency = 0
        self.emergency_ids = []
        self.total_vehicles = 0
        
        # 历史记录
        self.history_density = []
        self.history_speed = []
        self.history_lane_change = []
        self.history_speed_social = []
        self.history_speed_emergency = []
        self.history_lane_density = {i: [] for i in range(self.n_lanes)}
        
        # 初始化车辆
        self._init_vehicles()
    
    def _init_vehicles(self):
        """初始化车辆位置和速度"""
        # 容量校验：位置在 3 条车道上不重叠地随机采样，故总需求不得超过道路长度
        required = self.n_cars + self.max_emergency
        if required > self.L:
            raise ValueError(
                f"车辆总数超出道路容量: n_cars({self.n_cars}) + "
                f"max_emergency({self.max_emergency}) = {required} > L({self.L})"
            )
        
        # 随机生成不重叠的位置
        all_positions = random.sample(range(self.L), required)
        
        # 初始化社会车辆（只在车道0和1）
        social_positions = all_positions[:self.n_cars]
        for i, pos in enumerate(social_positions):
            self.pos[i] = pos
            self.vel[i] = random.randint(self.v_init_min, self.v_init_max)
            self.lane[i] = random.choice([0, 1])  # 只在超车道和行车道
            self.road[self.lane[i], pos] = i
        
        # 随机生成应急车数量
        self.n_emergency = np.random.binomial(self.max_emergency, self.emergency_prob)
        
        if self.n_emergency > 0:
            emergency_positions = all_positions[self.n_cars:self.n_cars + self.n_emergency]
            for i, pos in enumerate(emergency_positions):
                vid = self.n_cars + i
                self.pos[vid] = pos
                self.vel[vid] = self.v_max  # 应急车以最大速度行驶
                self.lane[vid] = self.n_lanes - 1  # 应急车道（最右侧）
                self.has_emergency[vid] = True
                self.road[self.lane[vid], pos] = vid
                self.emergency_ids.append(vid)
        
        self.total_vehicles = self.n_cars + self.n_emergency
    
    def _distance_to_front(self, vid):
        """计算到前车的距离（考虑环形道路）"""
        lane = self.lane[vid]
        pos = self.pos[vid]
        
        # 在当前车道向前搜索
        for d in range(1, self.L + 1):
            check_pos = (pos + d) % self.L
            if self.road[lane, check_pos] != -1:
                return d
        return self.L  # 没有前车
    
    def _emergency_lane_default(self, vid):
        """应急车道准入的默认规则（不含分段闸门）。

        这是 CA 模型自身对"社会车辆能否进入应急车道"的判断：
          - 应急车本人恒可进入；
          - 应急车已在影响范围 D 内（即将到达本车位置）时，禁止进入，
            以便给应急车留出通道；
          - 其余情况允许进入。

        env_maddpg 会用同名实例属性覆盖 _can_enter_emergency_lane，
        在其上再叠加"分段开/关"闸门；本方法作为其中的物理规则被复用。
        """
        if self.has_emergency[vid]:
            return True
        return not self._emergency_lane_blocked(self.pos[vid], self.n_lanes - 1)

    def _emergency_lane_blocked(self, pos, lane):
        """判断位置 pos 所在车道的应急车道入口是否被应急车封锁。

        即：是否有应急车位于同一条车道上、且在 pos 后方的 yield_distance
        个元胞内（环形道路按 (pos - em_pos) % L 计算后方距离）。
        """
        if self.n_emergency == 0:
            return False
        dist = getattr(self, 'yield_distance', self.D)
        for em_id in self.emergency_ids:
            if self.lane[em_id] != lane:
                continue
            behind = (pos - self.pos[em_id]) % self.L
            if 0 < behind <= dist:
                return True
        return False

    def _can_enter_emergency_lane_impl(self, vid, segment_open=True):
        """社会车辆能否进入应急车道（默认实现，可被 env 覆盖）。

        参数:
            vid: 车辆编号
            segment_open: 该车所在分段的应急车道闸门是否开放
                          （high.py 独立运行时无分段概念，默认 True）

        返回:
            True 表示允许进入应急车道。

        说明:
            __init__ 会把本方法绑定到实例属性 self._can_enter_emergency_lane，
            env_maddpg 通过覆盖该实例属性来叠加"分段开/关"闸门。
        """
        if self.has_emergency[vid]:
            return True
        if not segment_open:
            return False
        return self._emergency_lane_default(vid)

    def segment_of(self, pos):
        """位置 -> 分段编号（供 env 观测/奖励复用）。"""
        seg_len = getattr(self, 'segment_length', None)
        n_seg = getattr(self, 'n_segments', None)
        if not seg_len or not n_seg:
            return 0
        return int(pos // seg_len) % n_seg

    def segment_emergency_ids(self, segment_length=None, n_segments=None):
        """返回当前有应急车所在的分段集合（供 env 奖励复用）。"""
        seg_len = segment_length if segment_length else getattr(self, 'segment_length', None)
        n_seg = n_segments if n_segments else getattr(self, 'n_segments', None)
        if not seg_len or not n_seg or self.n_emergency == 0:
            return set()
        return {int(self.pos[em_id] // seg_len) % n_seg for em_id in self.emergency_ids}

    def _can_change_lane(self, vid, direction):
        """检查是否可以向指定方向变道"""
        lane = self.lane[vid]
        pos = self.pos[vid]
        
        # 确定目标车道
        if direction == 'left':
            target_lane = lane - 1
        elif direction == 'right':
            target_lane = lane + 1
        else:
            return False
        
        # 检查边界
        if target_lane < 0 or target_lane >= self.n_lanes:
            return False
        
        # 应急车道检查：应急车恒可进入；社会车辆需通过准入判定。
        # 默认规则只看应急车是否在影响范围 D 内；
        # env_maddpg 会覆盖 _can_enter_emergency_lane，在其上叠加"分段开/关"闸门。
        if target_lane == self.n_lanes - 1 and not self.has_emergency[vid]:
            if not self._can_enter_emergency_lane(vid):
                return False
        
        # 检查安全距离
        # 对于避让车辆（g=1），使用更小的安全距离dev
        safe_dist = self.dev if self.g[vid] == 1 else 1
        
        for d in range(-safe_dist, safe_dist + 1):
            check_pos = (pos + d) % self.L
            if self.road[target_lane, check_pos] != -1:
                return False
        
        return True
    
    def _change_lane(self, vid, direction):
        """执行变道"""
        old_lane = self.lane[vid]
        pos = self.pos[vid]
        
        if direction == 'left':
            new_lane = old_lane - 1
        elif direction == 'right':
            new_lane = old_lane + 1
        else:
            return False
        
        # 更新道路状态
        self.road[old_lane, pos] = -1
        self.road[new_lane, pos] = vid
        self.lane[vid] = new_lane
        
        return True
    
    def _get_gap(self, vid):
        """计算前车距离g"""
        return self._distance_to_front(vid) - 1
    
    def step(self):
        """执行一个时间步"""
        # 重置避让标记
        self.g[:] = 0
        
        # 1. 标记需要避让的车辆
        self._mark_yielding_vehicles()
        
        # 2. 更新所有车辆（按位置从后向前的顺序）
        update_order = sorted(range(self.total_vehicles), 
                            key=lambda i: self.pos[i], 
                            reverse=True)
        
        for vid in update_order:
            if self.has_emergency[vid]:
                self._update_emergency_vehicle(vid)
            else:
                self._update_social_vehicle(vid)
        
        # 3. 记录统计信息
        self._record_statistics()
    
    def _mark_yielding_vehicles(self):
        """标记需要避让应急车的车辆"""
        if self.n_emergency == 0:
            return
        
        for em_id in self.emergency_ids:
            em_lane = self.lane[em_id]
            em_pos = self.pos[em_id]
            
            # 在应急车道前方D范围内查找社会车辆
            for d in range(1, self.D + 1):
                check_pos = (em_pos + d) % self.L
                vid = self.road[em_lane, check_pos]
                
                if vid != -1 and not self.has_emergency[vid]:
                    # 找到应急车前方第一辆社会车
                    # 尝试让它向左变道
                    if self._can_change_lane(vid, 'left'):
                        self.g[vid] = 1
                    else:
                        # 如果不能直接变道，标记右侧车辆
                        right_lane = em_lane - 1
                        if right_lane >= 0:
                            for dd in range(-self.dev, self.dev + 1):
                                check_pos2 = (check_pos + dd) % self.L
                                vid2 = self.road[right_lane, check_pos2]
                                if vid2 != -1 and not self.has_emergency[vid2]:
                                    self.g[vid2] = 1
                    break
    
    def _update_emergency_vehicle(self, vid):
        """更新应急车状态"""
        # 应急车以固定速度行驶，不变道
        gap = self._get_gap(vid)
        self.vel[vid] = min(self.v_max, gap)
        
        # 更新位置
        old_pos = self.pos[vid]
        new_pos = (old_pos + self.vel[vid]) % self.L
        
        self.road[self.lane[vid], old_pos] = -1
        self.road[self.lane[vid], new_pos] = vid
        self.pos[vid] = new_pos
    
    def _update_social_vehicle(self, vid):
        """更新社会车辆状态"""
        lane = self.lane[vid]
        
        # 1. 确定当前状态
        gap = self._get_gap(vid)
        
        # 2. 变道决策
        changed_lane = False
        
        # 2a. 应急车道让行：已在应急车道且应急车即将到达，安全时尽快左移让出通道
        if (self.lane[vid] == self.n_lanes - 1
                and self._emergency_lane_blocked(self.pos[vid], self.lane[vid])):
            if self._can_change_lane(vid, 'left') and random.random() < self.P2:
                self._change_lane(vid, 'left')
                changed_lane = True
                lane = self.lane[vid]
        
        # 2b. 如果有避让标记，优先尝试变道
        if self.g[vid] == 1:
            # 尝试向左变道
            if self._can_change_lane(vid, 'left'):
                if random.random() < self.P2:
                    self._change_lane(vid, 'left')
                    changed_lane = True
                    lane = self.lane[vid]
            # 如果不能向左，尝试向右
            elif self._can_change_lane(vid, 'right'):
                if random.random() < self.P2:
                    self._change_lane(vid, 'right')
                    changed_lane = True
                    lane = self.lane[vid]
        
        # 2c. 常规变道（无应急车影响）
        if not changed_lane:
            # 随机决定是否尝试变道
            if random.random() < self.P1:
                # 优先向左（超车）
                if self._can_change_lane(vid, 'left'):
                    self._change_lane(vid, 'left')
                    lane = self.lane[vid]
                # 其次向右
                elif self._can_change_lane(vid, 'right'):
                    self._change_lane(vid, 'right')
                    lane = self.lane[vid]
        
        # 3. 更新速度
        new_gap = self._get_gap(vid)
        
        if new_gap == 0:
            # 必须刹车
            self.vel[vid] = 0
        elif new_gap == 1:
            # 前方仅空一格时速度不能超过空隙，否则会越过或覆盖前车。
            self.vel[vid] = min(self.vel[vid] + 1, self.v_max, new_gap)
        else:
            # 正常加速
            if self.vel[vid] < self.v_max:
                self.vel[vid] += 1
            # 但不超过前车距离
            self.vel[vid] = min(self.vel[vid], new_gap)
        
        # 4. 更新位置
        old_pos = self.pos[vid]
        new_pos = (old_pos + self.vel[vid]) % self.L
        
        self.road[lane, old_pos] = -1
        self.road[lane, new_pos] = vid
        self.pos[vid] = new_pos
    
    def _record_statistics(self):
        """记录统计信息"""
        # 确保 total_vehicles 不超过数组长度
        total_vehicles = min(self.total_vehicles, len(self.lane))
        
        # 计算各车道密度
        lane_density = []
        lane_speed = []
        
        for l in range(self.n_lanes):
            cars_in_lane = np.sum(self.lane[:total_vehicles] == l)
            lane_density.append(cars_in_lane)
            
            if cars_in_lane > 0:
                # 创建车道掩码，确保索引安全
                lane_mask = self.lane[:total_vehicles] == l
                avg_speed = np.mean(self.vel[:total_vehicles][lane_mask])
                lane_speed.append(avg_speed)
            else:
                lane_speed.append(0)
        
        self.history_lane_density = {i: self.history_lane_density.get(i, []) + [lane_density[i]] 
                                     for i in range(self.n_lanes)}
        
        # 社会车辆平均速度
        if self.n_cars > 0:
            social_speed = np.mean(self.vel[:self.n_cars])
            self.history_speed_social.append(social_speed)
        
        # 应急车平均速度
        if self.n_emergency > 0:
            em_speed = np.mean([self.vel[em_id] for em_id in self.emergency_ids])
            self.history_speed_emergency.append(em_speed)


# 使用示例
if __name__ == "__main__":
    print("EmergencyVehicleCA module loaded successfully")
