#!/usr/bin/env python3
"""
检查可用的模型检查点
"""

import os
import glob

def main():
    print("=== 检查可用的模型检查点 ===")
    
    ckpt_dir = "output/ckpt/ESD-Chinese-Singing-MFA/"
    
    if not os.path.exists(ckpt_dir):
        print(f"错误: 检查点目录不存在: {ckpt_dir}")
        print("请确保训练已经开始并保存了检查点")
        return
    
    # 查找所有.pth.tar文件
    ckpt_files = glob.glob(os.path.join(ckpt_dir, "*.pth.tar"))
    
    if not ckpt_files:
        print(f"在 {ckpt_dir} 中没有找到任何检查点文件")
        print("请等待训练保存第一个检查点（通常在1000步）")
        return
    
    # 提取步数并排序
    steps = []
    for file in ckpt_files:
        basename = os.path.basename(file)
        step = basename.replace('.pth.tar', '')
        try:
            steps.append(int(step))
        except ValueError:
            continue
    
    steps.sort()
    
    print(f"找到 {len(steps)} 个检查点:")
    for step in steps:
        file_path = os.path.join(ckpt_dir, f"{step}.pth.tar")
        file_size = os.path.getsize(file_path) / (1024*1024)  # MB
        print(f"  步数: {step:>6} - 文件大小: {file_size:.1f} MB")
    
    if steps:
        print(f"\n推荐使用的检查点:")
        if len(steps) >= 3:
            print(f"  早期测试: {steps[0]} (可能音质较差)")
            print(f"  中期测试: {steps[len(steps)//2]} (平衡质量)")
            print(f"  最新模型: {steps[-1]} (可能最佳质量)")
        else:
            print(f"  可用检查点: {', '.join(map(str, steps))}")
        
        print(f"\n使用示例:")
        latest_step = steps[-1]
        print(f"  python generate_emotion_samples.py {latest_step}")

if __name__ == "__main__":
    main() 