#!/usr/bin/env python3
"""
ESD-Chinese 情感语音合成推理示例
"""

import os
import json

def show_available_options():
    """显示可用的说话人和情感选项"""
    print("=== 可用选项 ===")
    
    # 读取说话人信息
    with open("preprocessed_data/ESD-Chinese/speakers.json", "r") as f:
        speakers = json.load(f)
    print("可用说话人ID:", list(speakers.keys()))
    
    # 读取情感信息
    with open("preprocessed_data/ESD-Chinese/emotions.json", "r") as f:
        emotions = json.load(f)
    print("可用情感类别:", list(emotions["emotion_dict"].keys()))
    
    # 读取一些示例音素
    with open("preprocessed_data/ESD-Chinese/train.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()[:5]
    
    print("\n=== 音素序列示例 ===")
    for i, line in enumerate(lines):
        parts = line.strip().split("|")
        if len(parts) >= 4:
            phoneme_ids = parts[2]
            text = parts[3]
            print(f"示例{i+1}: {text}")
            print(f"  音素ID序列: {phoneme_ids}")
            print()

def run_inference_examples():
    """运行推理示例"""
    print("=== 推理示例 ===")
    
    # 示例1: 单句合成
    print("1. 单句合成示例:")
    cmd1 = """python synthesize_chinese.py \\
    --restore_step 100000 \\
    --mode single \\
    --phonemes "42 51 13 67 14 41 30 9 29 64 25 33 52 8" \\
    --text "他对谁都那么友好" \\
    --speaker_id "0008" \\
    --emotion_id "惊讶" \\
    -p config/ESD-Chinese/preprocess.yaml \\
    -m config/ESD-Chinese/model.yaml \\
    -t config/ESD-Chinese/train.yaml"""
    print(cmd1)
    print()
    
    # 示例2: 不同情感
    print("2. 不同情感示例:")
    emotions = ["开心", "中立", "伤心", "愤怒", "惊讶"]
    for emotion in emotions:
        cmd = f"""python synthesize_chinese.py \\
    --restore_step 100000 \\
    --mode single \\
    --phonemes "43 56 45 23 42 21 67 71 46 22 59 41 43 56 45 23 43 51 35" \\
    --text "自己的事情要自己做" \\
    --speaker_id "0005" \\
    --emotion_id "{emotion}" \\
    -p config/ESD-Chinese/preprocess.yaml \\
    -m config/ESD-Chinese/model.yaml \\
    -t config/ESD-Chinese/train.yaml"""
        print(f"情感-{emotion}:")
        print(cmd)
        print()
    
    # 示例3: 批量合成
    print("3. 批量合成示例:")
    cmd3 = """python synthesize_chinese.py \\
    --restore_step 100000 \\
    --mode batch \\
    --source preprocessed_data/ESD-Chinese/val.txt \\
    -p config/ESD-Chinese/preprocess.yaml \\
    -m config/ESD-Chinese/model.yaml \\
    -t config/ESD-Chinese/train.yaml"""
    print(cmd3)
    print()
    
    # 示例4: 控制语音特征
    print("4. 控制语音特征示例:")
    cmd4 = """python synthesize_chinese.py \\
    --restore_step 100000 \\
    --mode single \\
    --phonemes "42 51 13 67 14 41 30 9 29 64 25 33 52 8" \\
    --text "他对谁都那么友好" \\
    --speaker_id "0008" \\
    --emotion_id "开心" \\
    --pitch_control 1.2 \\
    --energy_control 1.1 \\
    --duration_control 0.9 \\
    -p config/ESD-Chinese/preprocess.yaml \\
    -m config/ESD-Chinese/model.yaml \\
    -t config/ESD-Chinese/train.yaml"""
    print(cmd4)
    print()

def show_phoneme_mapping():
    """显示音素映射表的前20个"""
    print("=== 音素映射表 (前20个) ===")
    with open("preprocessed_data/ESD-Chinese/phoneme_mapping.json", "r", encoding="utf-8") as f:
        mapping = json.load(f)
    
    phoneme_list = mapping["phoneme_list"]
    for i, phoneme in enumerate(phoneme_list[:20]):
        print(f"{i:2d}: {phoneme}")
    print("...")

def main():
    print("🎤 ESD-Chinese 情感语音合成推理指南")
    print("=" * 50)
    
    show_available_options()
    show_phoneme_mapping()
    run_inference_examples()
    
    print("📝 使用说明:")
    print("1. 首先确保模型已训练完成，checkpoint文件位于 output/ckpt/ESD-Chinese/")
    print("2. 根据 --restore_step 参数指定要加载的checkpoint步数")
    print("3. 音素序列可以从训练数据中复制，或使用MFA工具生成")
    print("4. 输出音频文件将保存在 output/result/ESD-Chinese/ 目录")
    print("5. 支持的控制参数:")
    print("   - pitch_control: 音调控制 (0.5-2.0, 默认1.0)")
    print("   - energy_control: 音量控制 (0.5-2.0, 默认1.0)")
    print("   - duration_control: 语速控制 (0.5-2.0, 默认1.0)")

if __name__ == "__main__":
    main() 