# -*- coding: utf-8 -*-
"""
批量运行仿真模型脚本
在不同车辆数初值下运行模型，并生成疏通距离对比图
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False

# 导入模型类
from high import EmergencyVehicleCA, PHOTO_DIR

# 定义保存目录
BASE_PHOTO_DIR = r'E:\桌面\highspeed\PHOTO'

def run_single_scenario(n_cars, n_emergency, warmup_steps=100, steps=500, save_dir=None):
    """
    运行单个场景并保存结果
    参数:
        n_cars: 社会车辆数量
        n_emergency: 应急车数量（1-3）
        warmup_steps: 预热步数
        steps: 总仿真步数
        save_dir: 图片保存目录
    返回:
        clearance_info: 应急车疏通信息
    """
    # 道路参数
    L = 200
    n_lanes = 3
    
    # 车辆参数
    v_max = 5
    v_init_min = 3
    v_init_max = 5
    
    # 应急车参数
    D = 40
    max_emergency = 3
    
    # 行为参数
    P1 = 0.6
    P2 = 0.9
    dev = 2
    
    # 创建模型
    model = EmergencyVehicleCA(L, n_lanes, n_cars, v_max, D, P1, P2, dev,
                               max_emergency=max_emergency,
                               v_init_min=v_init_min, v_init_max=v_init_max,
                               warmup_steps=warmup_steps,
                               add_emergency_after_warmup=True)
    
    # 运行仿真并追踪疏通距离
    clearance_info = model.run_with_clearance_tracking(
        steps=steps,
        warmup_steps=warmup_steps,
        n_emergency=n_emergency,
        max_clearance_steps=500
    )
    
    # 保存图片
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        scenario_name = f"{n_cars}辆_{n_emergency}辆应急车"
        model.plot_results(start=0, end=min(warmup_steps + 200, steps), 
                          save_dir=save_dir, scenario_name=scenario_name)
        model.plot_lane_stats(start=0, end=min(warmup_steps + 200, steps),
                             avg_start=warmup_steps, save_dir=save_dir, scenario_name=scenario_name)
    
    return clearance_info


def run_batch_scenarios(car_range, emergency_range, warmup_steps=100, steps=500, repeats=3):
    """
    批量运行不同车辆数和应急车数的场景
    参数:
        car_range: 车辆数范围 (start, end, step)
        emergency_range: 应急车数范围 (1, 2, 3)
        warmup_steps: 预热步数
        steps: 总仿真步数
        repeats: 每个配置重复次数
    返回:
        results: 结果字典
    """
    results = {
        'n_cars': [],
        'n_emergency': [],
        'avg_clearance_distance': [],
        'avg_clearance_steps': [],
        'density': []
    }
    
    car_counts = range(car_range[0], car_range[1] + 1, car_range[2])
    
    total_scenarios = len(car_counts) * len(emergency_range) * repeats
    current_scenario = 0
    
    print("=" * 70)
    print("开始批量运行仿真")
    print(f"车辆数范围: {car_range[0]}-{car_range[1]}辆, 步长: {car_range[2]}")
    print(f"应急车数量: {list(emergency_range)}")
    print(f"每个配置重复: {repeats} 次")
    print(f"总场景数: {total_scenarios}")
    print("=" * 70)
    
    for n_cars in car_counts:
        for n_emergency in emergency_range:
            distances = []
            step_counts = []
            
            # 创建子文件夹
            subfolder_name = f"{n_cars}辆社会车辆"
            subfolder_path = os.path.join(BASE_PHOTO_DIR, subfolder_name)
            os.makedirs(subfolder_path, exist_ok=True)
            
            for rep in range(repeats):
                current_scenario += 1
                print(f"\n[{current_scenario}/{total_scenarios}] 运行: {n_cars}辆车, {n_emergency}辆应急车, 第{rep+1}次")
                
                # 运行场景
                clearance_info = run_single_scenario(
                    n_cars=n_cars,
                    n_emergency=n_emergency,
                    warmup_steps=warmup_steps,
                    steps=steps,
                    save_dir=subfolder_path
                )
                
                # 统计疏通距离
                for em_id, info in clearance_info.items():
                    if info['cleared']:
                        distances.append(info['distance'])
                        step_counts.append(info['steps'])
                    else:
                        # 未疏通，使用最大值
                        distances.append(info['distance'])
                        step_counts.append(info['steps'])
                        print(f"  警告: 应急车 {em_id} 未在规定时间内完全疏通")
            
            # 计算平均值
            avg_distance = np.mean(distances) if distances else 0
            avg_steps = np.mean(step_counts) if step_counts else 0
            density = n_cars / 200  # 车辆密度 = 车辆数 / 道路长度
            
            results['n_cars'].append(n_cars)
            results['n_emergency'].append(n_emergency)
            results['avg_clearance_distance'].append(avg_distance)
            results['avg_clearance_steps'].append(avg_steps)
            results['density'].append(density)
            
            print(f"  平均疏通距离: {avg_distance:.1f} 元胞")
            print(f"  平均疏通步数: {avg_steps:.1f} 步")
    
    return results


def plot_clearance_comparison(results, save_dir):
    """
    绘制疏通距离对比图
    参数:
        results: 批量运行结果
        save_dir: 保存目录
    """
    # 提取数据
    n_cars_list = sorted(set(results['n_cars']))
    n_emergency_list = sorted(set(results['n_emergency']))
    
    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 颜色映射
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    markers = ['o', 's', '^']
    
    # 图1: 疏通距离 vs 车辆密度（按应急车数量分组）
    ax1 = axes[0]
    for i, n_em in enumerate(n_emergency_list):
        distances = []
        densities = []
        for j, (cars, em, dist) in enumerate(zip(results['n_cars'], results['n_emergency'], results['avg_clearance_distance'])):
            if em == n_em:
                distances.append(dist)
                densities.append(results['density'][j])
        
        ax1.plot(densities, distances, marker=markers[i], color=colors[i], 
                linewidth=2, markersize=8, label=f'{n_em}辆应急车')
    
    ax1.set_xlabel('车辆密度 (车辆数/元胞)', fontsize=12)
    ax1.set_ylabel('平均疏通距离 (元胞)', fontsize=12)
    ax1.set_title('应急车疏通距离 vs 车辆密度', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 图2: 疏通步数 vs 车辆密度（按应急车数量分组）
    ax2 = axes[1]
    for i, n_em in enumerate(n_emergency_list):
        steps_list = []
        densities = []
        for j, (cars, em, steps) in enumerate(zip(results['n_cars'], results['n_emergency'], results['avg_clearance_steps'])):
            if em == n_em:
                steps_list.append(steps)
                densities.append(results['density'][j])
        
        ax2.plot(densities, steps_list, marker=markers[i], color=colors[i],
                linewidth=2, markersize=8, label=f'{n_em}辆应急车')
    
    ax2.set_xlabel('车辆密度 (车辆数/元胞)', fontsize=12)
    ax2.set_ylabel('平均疏通步数', fontsize=12)
    ax2.set_title('应急车疏通步数 vs 车辆密度', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    filepath = os.path.join(save_dir, '应急车疏通距离对比图.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\n对比图已保存: {filepath}")
    
    plt.show()


def plot_heatmap(results, save_dir):
    """
    绘制热力图：疏通距离 vs 车辆数和应急车数
    """
    n_cars_list = sorted(set(results['n_cars']))
    n_emergency_list = sorted(set(results['n_emergency']))
    
    # 创建数据矩阵
    distance_matrix = np.zeros((len(n_emergency_list), len(n_cars_list)))
    
    for i, n_em in enumerate(n_emergency_list):
        for j, n_cars in enumerate(n_cars_list):
            for k, (cars, em, dist) in enumerate(zip(results['n_cars'], results['n_emergency'], results['avg_clearance_distance'])):
                if cars == n_cars and em == n_em:
                    distance_matrix[i, j] = dist
                    break
    
    # 绘制热力图
    fig, ax = plt.subplots(figsize=(14, 5))
    
    im = ax.imshow(distance_matrix, aspect='auto', cmap='YlOrRd')
    
    # 设置坐标轴
    ax.set_xticks(range(len(n_cars_list)))
    ax.set_xticklabels([str(c) for c in n_cars_list])
    ax.set_yticks(range(len(n_emergency_list)))
    ax.set_yticklabels([f'{em}辆' for em in n_emergency_list])
    
    ax.set_xlabel('社会车辆数量', fontsize=12)
    ax.set_ylabel('应急车数量', fontsize=12)
    ax.set_title('应急车疏通距离热力图 (元胞)', fontsize=14)
    
    # 添加数值标注
    for i in range(len(n_emergency_list)):
        for j in range(len(n_cars_list)):
            text = ax.text(j, i, f'{distance_matrix[i, j]:.0f}',
                          ha='center', va='center', color='black', fontsize=9)
    
    # 添加颜色条
    cbar = plt.colorbar(im)
    cbar.set_label('疏通距离 (元胞)', fontsize=10)
    
    plt.tight_layout()
    
    # 保存图片
    filepath = os.path.join(save_dir, '应急车疏通距离热力图.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"热力图已保存: {filepath}")
    
    plt.show()


def main():
    """主函数"""
    print("=" * 70)
    print("高速公路应急车道仿真 - 批量运行脚本")
    print("=" * 70)
    
    # 确保保存目录存在
    os.makedirs(BASE_PHOTO_DIR, exist_ok=True)
    
    # 参数设置
    car_range = (100, 200, 10)  # 车辆数：100-200，步长10
    emergency_range = (1, 2, 3)  # 应急车数量：1-3辆
    warmup_steps = 100
    steps = 500
    repeats = 3  # 每个配置重复3次取平均
    
    # 批量运行
    results = run_batch_scenarios(
        car_range=car_range,
        emergency_range=emergency_range,
        warmup_steps=warmup_steps,
        steps=steps,
        repeats=repeats
    )
    
    # 生成对比图
    print("\n" + "=" * 70)
    print("生成对比图...")
    print("=" * 70)
    
    plot_clearance_comparison(results, BASE_PHOTO_DIR)
    plot_heatmap(results, BASE_PHOTO_DIR)
    
    # 保存结果数据
    data_filepath = os.path.join(BASE_PHOTO_DIR, '仿真结果数据.txt')
    with open(data_filepath, 'w', encoding='utf-8') as f:
        f.write("车辆数\t应急车数\t密度\t平均疏通距离\t平均疏通步数\n")
        for i in range(len(results['n_cars'])):
            f.write(f"{results['n_cars'][i]}\t{results['n_emergency'][i]}\t"
                   f"{results['density'][i]:.3f}\t{results['avg_clearance_distance'][i]:.1f}\t"
                   f"{results['avg_clearance_steps'][i]:.1f}\n")
    print(f"数据已保存: {data_filepath}")
    
    print("\n" + "=" * 70)
    print("批量运行完成！")
    print(f"图片保存目录: {BASE_PHOTO_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
