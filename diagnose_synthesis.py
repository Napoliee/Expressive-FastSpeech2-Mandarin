#!/usr/bin/env python3
"""
语音合成质量诊断脚本
"""

import os
import json
import librosa
import numpy as np
import matplotlib.pyplot as plt

def check_training_progress():
    """检查训练进度"""
    print("=== 训练进度检查 ===")
    
    # 检查checkpoint文件
    ckpt_dir = "output/ckpt/ESD-Chinese"
    if os.path.exists(ckpt_dir):
        ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth.tar')]
        if ckpts:
            steps = [int(f.replace('.pth.tar', '')) for f in ckpts]
            print(f"可用checkpoint: {sorted(steps)}")
            print(f"最新checkpoint: {max(steps)} 步")
            
            # 判断训练是否充分
            if max(steps) < 300000:
                print("⚠️  警告: 训练步数可能不足，建议继续训练到300,000步以上")
            else:
                print("✅ 训练步数充分")
        else:
            print("❌ 未找到checkpoint文件")
    else:
        print("❌ checkpoint目录不存在")

def check_audio_quality():
    """检查生成音频的质量"""
    print("\n=== 音频质量检查 ===")
    
    result_dir = "output/result/ESD-Chinese"
    if not os.path.exists(result_dir):
        print("❌ 结果目录不存在")
        return
    
    wav_files = [f for f in os.listdir(result_dir) if f.endswith('.wav')]
    if not wav_files:
        print("❌ 未找到生成的音频文件")
        return
    
    for wav_file in wav_files:
        wav_path = os.path.join(result_dir, wav_file)
        audio, sr = librosa.load(wav_path, sr=None)
        
        print(f"\n文件: {wav_file}")
        print(f"  采样率: {sr} Hz")
        print(f"  时长: {len(audio)/sr:.2f} 秒")
        print(f"  最大振幅: {np.max(np.abs(audio)):.3f}")
        print(f"  RMS能量: {np.sqrt(np.mean(audio**2)):.3f}")
        
        # 检查音频是否有问题
        if np.max(np.abs(audio)) < 0.01:
            print("  ⚠️  音频振幅过小，可能听不清")
        if sr != 22050:
            print(f"  ⚠️  采样率异常，期望22050Hz，实际{sr}Hz")
        if len(audio) < sr * 0.5:
            print("  ⚠️  音频时长过短")

def check_phoneme_mapping():
    """检查音素映射"""
    print("\n=== 音素映射检查 ===")
    
    mapping_file = "preprocessed_data/ESD-Chinese/phoneme_mapping.json"
    if not os.path.exists(mapping_file):
        print("❌ 音素映射文件不存在")
        return
    
    with open(mapping_file, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    
    print(f"音素总数: {len(mapping['phoneme_list'])}")
    print("前10个音素:")
    for i, phoneme in enumerate(mapping['phoneme_list'][:10]):
        print(f"  {i}: {phoneme}")

def suggest_improvements():
    """提供改进建议"""
    print("\n=== 改进建议 ===")
    
    print("1. 继续训练:")
    print("   python train.py config/ESD-Chinese/preprocess.yaml config/ESD-Chinese/model.yaml config/ESD-Chinese/train.yaml")
    
    print("\n2. 尝试不同的音素序列:")
    print("   # 从验证集选择一个清晰的样本")
    print("   head -10 preprocessed_data/ESD-Chinese/val.txt")
    
    print("\n3. 调整合成参数:")
    print("   # 尝试调整语音控制参数")
    print("   --pitch_control 1.0 --energy_control 1.2 --duration_control 1.0")
    
    print("\n4. 使用不同说话人/情感:")
    print("   # 尝试不同的说话人ID和情感")
    print("   --speaker_id 0001 --emotion_id 中立")
    
    print("\n5. 检查原始数据:")
    print("   # 播放原始训练数据中的音频，确认质量")
    print("   ls raw_data/ESD-Chinese/0008/ | head -5")

def create_test_samples():
    """创建测试样本"""
    print("\n=== 生成测试样本 ===")
    
    # 从验证集获取几个样本
    val_file = "preprocessed_data/ESD-Chinese/val.txt"
    if os.path.exists(val_file):
        with open(val_file, "r", encoding="utf-8") as f:
            lines = f.readlines()[:5]
        
        print("建议测试以下样本:")
        for i, line in enumerate(lines):
            parts = line.strip().split("|")
            if len(parts) >= 7:
                phonemes = parts[2]
                text = parts[3]
                speaker = parts[1]
                emotion = parts[6]
                
                print(f"\n样本{i+1}: {text}")
                print(f"命令:")
                print(f'CUDA_VISIBLE_DEVICES=2 python synthesize_chinese.py \\')
                print(f'  --restore_step 100000 \\')
                print(f'  --mode single \\')
                print(f'  --phonemes "{phonemes}" \\')
                print(f'  --text "{text}" \\')
                print(f'  --speaker_id "{speaker}" \\')
                print(f'  --emotion_id "{emotion}" \\')
                print(f'  -p config/ESD-Chinese/preprocess.yaml \\')
                print(f'  -m config/ESD-Chinese/model.yaml \\')
                print(f'  -t config/ESD-Chinese/train.yaml')

def main():
    print("🔍 语音合成质量诊断")
    print("=" * 50)
    
    check_training_progress()
    check_audio_quality() 
    check_phoneme_mapping()
    suggest_improvements()
    create_test_samples()

if __name__ == "__main__":
    main() 