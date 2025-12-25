import pandas as pd
import json
import numpy as np
import os

# ================= 配置区域 =================
# 把你所有的 CSV 文件名都写在这里
csv_files = [
    '240.csv', 
    '280.csv', 
    '340.csv', 
    '380.csv'
]

# 输出文件名
output_json = 'go1_amp_dataset_merged.json'
# ===========================================

def process_csv(file_path):
    """读取并清洗单个 CSV，返回帧列表"""
    print(f"正在处理: {file_path} ...")
    
    # 1. 读取 CSV (自动处理逗号错位问题)
    try:
        df = pd.read_csv(file_path, index_col=False)
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {file_path}")
        return []

    # 清理可能存在的 Unnamed 列
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    frames = []
    
    for idx, row in df.iterrows():
        frame = []
        
        # --- A. 基座位置 (x, y, z) ---
        # 即使不同 CSV 的起始位置不同也没关系，AMP 主要关注相对姿态和关节运动
        frame.extend([row['base_x'], row['base_y'], row['base_z']])

        # --- B. 基座姿态 (Quaternion x, y, z, w) ---
        q = np.array([row['base_quat_x'], row['base_quat_y'], row['base_quat_z'], row['base_quat_w']])
        # 归一化，防止警告
        norm = np.linalg.norm(q)
        if norm > 1e-6:
            q = q / norm
        frame.extend(q.tolist())

        # --- C. 12个关节角度 ---
        # 假设列名格式统一为 q0, q1 ... q11
        try:
            joints = [row[f'q{i}'] for i in range(12)]
            frame.extend(joints)
        except KeyError:
            print(f"❌ 错误: 在 {file_path} 中找不到关节列 (q0~q11)")
            return []

        frames.append(frame)
        
    print(f"  -> 提取了 {len(frames)} 帧")
    return frames

# ================= 主程序 =================
all_frames = []

print(f"开始合并 {len(csv_files)} 个动作文件...")

for csv_file in csv_files:
    # 处理每一个文件，并追加到总列表中
    frames = process_csv(csv_file)
    all_frames.extend(frames)

if not all_frames:
    print("❌ 错误: 没有提取到任何数据！请检查 CSV 路径。")
else:
    # 封装成 JSON
    data = {
        "LoopMode": "Wrap",           
        "FrameDuration": 0.02,        # 假设所有文件的采样间隔都是 0.02s
        "EnableFrameInterpolation": True,
        "Frames": all_frames          # 这是一个包含所有 CSV 数据的超级长列表
    }

    with open(output_json, 'w') as f:
        json.dump(data, f)

    print("-" * 50)
    print(f"✅ 合并完成！数据集已保存为: {output_json}")
    print(f"📊 总数据量: {len(all_frames)} 帧")
    print("-" * 50)
    print("💡 提示: 在 Isaac Gym 的配置文件中，将 motion_file 指向这个 json 即可。")