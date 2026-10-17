#!/usr/bin/env python
"""
MADDPG 训练结果可视化表格
展示仿真结果数据的交互式表格和统计图表
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
import os
import sys

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("正在加载数据...", flush=True)

# 数据来源：优先读取数据文件，避免把数据写死在代码里。
# 若文件缺失则回退到内嵌副本并明确告警（内嵌副本仅为向后兼容保留）。
DATA_FILE = os.path.join('PHOTO', '仿真结果数据.txt')

FALLBACK_DATA = """车辆数	应急车数	密度	平均疏通距离	平均疏通步数
100	1	0.500	2500.0	167.3
100	2	0.500	2500.0	250.5
100	3	0.500	2500.0	56.4
110	1	0.550	2500.0	167.3
110	2	0.550	2500.0	250.5
110	3	0.550	2500.0	167.3
120	1	0.600	2500.0	500.0
120	2	0.600	2500.0	416.8
120	3	0.600	2500.0	167.3
130	1	0.650	2500.0	500.0
130	2	0.650	2500.0	333.7
130	3	0.650	2500.0	222.8
140	1	0.700	2500.0	500.0
140	2	0.700	2500.0	500.0
140	3	0.700	2500.0	167.3
150	1	0.750	2500.0	500.0
150	2	0.750	2500.0	333.7
150	3	0.750	2500.0	167.3
160	1	0.800	2500.0	500.0
160	2	0.800	2500.0	416.8
160	3	0.800	2500.0	167.3
170	1	0.850	2500.0	500.0
170	2	0.850	2500.0	416.8
170	3	0.850	2500.0	278.2
180	1	0.900	2500.0	500.0
180	2	0.900	2500.0	500.0
180	3	0.900	2500.0	222.8
190	1	0.950	2500.0	500.0
190	2	0.950	2500.0	333.7
190	3	0.950	2500.0	222.8
200	1	1.000	2500.0	500.0
200	2	1.000	2500.0	333.7
200	3	1.000	2500.0	222.8"""

if os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        data_text = f.read()
    print(f'数据来源: {DATA_FILE}')
else:
    data_text = FALLBACK_DATA
    print(f'[警告] 未找到 {DATA_FILE}，使用内嵌副本（内容应与该文件一致）。')
    print('        建议补齐数据文件，避免数据与代码耦合。')

# 解析数据
from io import StringIO
df = pd.read_csv(StringIO(data_text), sep='\t')

# 创建透视表
pivot_steps = df.pivot_table(
    values='平均疏通步数',
    index='车辆数',
    columns='应急车数',
    aggfunc='mean'
)

pivot_distance = df.pivot_table(
    values='平均疏通距离',
    index='车辆数',
    columns='应急车数',
    aggfunc='mean'
)

# 打印表格到控制台
print("=" * 80)
print("MADDPG 训练结果 - 仿真数据汇总")
print("=" * 80)
print("\n【原始数据表】")
print(df.to_string(index=False))

print("\n\n【平均疏通步数 - 透视表】")
print(pivot_steps.to_string())

print("\n\n【平均疏通距离 - 透视表】")
print(pivot_distance.to_string())

# 统计分析
print("\n\n【统计分析】")
print("-" * 60)
stats = df.groupby('应急车数')['平均疏通步数'].agg(['mean', 'std', 'min', 'max'])
stats.columns = ['平均值', '标准差', '最小值', '最大值']
print("\n按应急车数分组的疏通步数统计:")
print(stats.to_string())

print("\n\n按车辆数分组的疏通步数统计:")
stats2 = df.groupby('车辆数')['平均疏通步数'].agg(['mean', 'std', 'min', 'max'])
stats2.columns = ['平均值', '标准差', '最小值', '最大值']
print(stats2.to_string())

# 创建可视化图表
fig = plt.figure(figsize=(20, 14))

# 1. 热力图 - 疏通步数
ax1 = plt.subplot(2, 3, 1)
sns.heatmap(pivot_steps, annot=True, fmt='.1f', cmap='RdYlGn_r', 
            cbar_kws={'label': '平均疏通步数'}, ax=ax1, linewidths=0.5)
ax1.set_title('平均疏通步数热力图\n(数值越小越好)', fontsize=12, fontweight='bold')
ax1.set_xlabel('应急车数', fontsize=10)
ax1.set_ylabel('社会车辆数', fontsize=10)

# 2. 热力图 - 疏通距离
ax2 = plt.subplot(2, 3, 2)
sns.heatmap(pivot_distance, annot=True, fmt='.1f', cmap='RdYlGn', 
            cbar_kws={'label': '平均疏通距离'}, ax=ax2, linewidths=0.5)
ax2.set_title('平均疏通距离热力图', fontsize=12, fontweight='bold')
ax2.set_xlabel('应急车数', fontsize=10)
ax2.set_ylabel('社会车辆数', fontsize=10)

# 3. 不同应急车数下的疏通步数对比
ax3 = plt.subplot(2, 3, 3)
for emergency_count in sorted(df['应急车数'].unique()):
    data = df[df['应急车数'] == emergency_count]
    ax3.plot(data['车辆数'], data['平均疏通步数'], 
             marker='o', linewidth=2, markersize=6,
             label=f'{emergency_count}辆应急车')
ax3.set_xlabel('社会车辆数', fontsize=10)
ax3.set_ylabel('平均疏通步数', fontsize=10)
ax3.set_title('不同应急车数下的疏通步数对比', fontsize=12, fontweight='bold')
ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.3)

# 4. 分组柱状图
ax4 = plt.subplot(2, 3, 4)
x = np.arange(len(df['车辆数'].unique()))
width = 0.25
colors = ['#e74c3c', '#3498db', '#2ecc71']

for i, emergency_count in enumerate(sorted(df['应急车数'].unique())):
    data = df[df['应急车数'] == emergency_count]['平均疏通步数'].values
    ax4.bar(x + i * width, data, width, label=f'{emergency_count}辆应急车',
            color=colors[i], alpha=0.8, edgecolor='black')

ax4.set_xlabel('社会车辆数', fontsize=10)
ax4.set_ylabel('平均疏通步数', fontsize=10)
ax4.set_title('疏通步数分组对比', fontsize=12, fontweight='bold')
ax4.set_xticks(x + width)
ax4.set_xticklabels(sorted(df['车辆数'].unique()), rotation=45)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# 5. 箱线图
ax5 = plt.subplot(2, 3, 5)
df.boxplot(column='平均疏通步数', by='应急车数', ax=ax5)
ax5.set_xlabel('应急车数', fontsize=10)
ax5.set_ylabel('平均疏通步数', fontsize=10)
ax5.set_title('疏通步数分布箱线图', fontsize=12, fontweight='bold')
plt.suptitle('')

# 6. 效率提升分析
ax6 = plt.subplot(2, 3, 6)
efficiency_data = []
vehicle_counts = sorted(df['车辆数'].unique())

for vc in vehicle_counts:
    data = df[df['车辆数'] == vc]
    step_1 = data[data['应急车数'] == 1]['平均疏通步数'].values[0]
    step_3 = data[data['应急车数'] == 3]['平均疏通步数'].values[0]
    improvement = ((step_1 - step_3) / step_1) * 100
    efficiency_data.append(improvement)

bars = ax6.bar(vehicle_counts, efficiency_data, color='#9b59b6', alpha=0.8, edgecolor='black')
ax6.set_xlabel('社会车辆数', fontsize=10)
ax6.set_ylabel('效率提升百分比 (%)', fontsize=10)
ax6.set_title('3辆 vs 1辆应急车效率提升', fontsize=12, fontweight='bold')
ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax6.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, efficiency_data):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('training_results_visualization.png', dpi=150, bbox_inches='tight')
print("\n\n可视化图表已保存: training_results_visualization.png")
plt.close()

# 生成HTML报告
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>MADDPG 训练结果报告</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }}
        h1 {{
            text-align: center;
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 15px;
            margin-bottom: 30px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            padding-left: 10px;
            border-left: 4px solid #e74c3c;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: center;
        }}
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        tr:hover {{
            background-color: #e3f2fd;
            transition: background-color 0.3s;
        }}
        .stats-card {{
            display: inline-block;
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            padding: 20px 30px;
            margin: 10px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .stats-card h3 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            opacity: 0.9;
        }}
        .stats-card .value {{
            font-size: 28px;
            font-weight: bold;
        }}
        .highlight {{
            background-color: #fff3cd;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #ffc107;
            margin: 20px 0;
        }}
        .image-container {{
            text-align: center;
            margin: 30px 0;
        }}
        .image-container img {{
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚗 MADDPG 训练结果可视化报告</h1>
        
        <div style="text-align: center;">
            <div class="stats-card">
                <h3>总实验次数</h3>
                <div class="value">{len(df)}</div>
            </div>
            <div class="stats-card">
                <h3>车辆数范围</h3>
                <div class="value">{df['车辆数'].min()}-{df['车辆数'].max()}</div>
            </div>
            <div class="stats-card">
                <h3>平均疏通步数</h3>
                <div class="value">{df['平均疏通步数'].mean():.1f}</div>
            </div>
            <div class="stats-card">
                <h3>最佳性能</h3>
                <div class="value">{df['平均疏通步数'].min():.1f}</div>
            </div>
        </div>

        <h2>📊 原始数据表</h2>
        {df.to_html(index=False, classes='data-table')}
        
        <h2>📈 平均疏通步数透视表</h2>
        <div class="highlight">
            <strong>说明：</strong>行表示社会车辆数，列表示应急车数，数值越小表示疏通效率越高
        </div>
        {pivot_steps.to_html(classes='data-table')}
        
        <h2>📉 统计分析</h2>
        <h3>按应急车数分组</h3>
        {stats.to_html(classes='data-table')}
        
        <h3>按车辆数分组</h3>
        {stats2.to_html(classes='data-table')}
        
        <h2>📊 可视化图表</h2>
        <div class="image-container">
            <img src="training_results_visualization.png" alt="训练结果可视化">
        </div>
        
        <div class="highlight">
            <strong>关键发现：</strong>
            <ul>
                <li>3辆应急车的配置在大多数情况下表现最佳，平均疏通步数最低</li>
                <li>随着社会车辆数增加，疏通难度增大，步数相应增加</li>
                <li>应急车数量从1辆增加到3辆，效率提升明显</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""

with open('training_results_report.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("\nHTML报告已生成: training_results_report.html")
print("\n请在浏览器中打开 training_results_report.html 查看完整报告")
