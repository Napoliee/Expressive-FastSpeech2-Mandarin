#!/usr/bin/env python3
"""
使用IPA音素系统重新预处理数据
"""

import os
import json
import numpy as np
import textgrid
from tqdm import tqdm

def extract_ipa_data():
    """从TextGrid提取IPA音素和时长特征"""
    
    print("=== 使用IPA音素重新预处理数据 ===")
    
    # 读取说话人和情感映射
    with open("preprocessed_data/ESD-Chinese/speakers.json", "r") as f:
        speaker_map = json.load(f)
    
    with open("preprocessed_data/ESD-Chinese/emotions.json", "r") as f:
        emotion_data = json.load(f)
    
    train_data = []
    val_data = []
    
    textgrid_dir = "preprocessed_data/ESD-Chinese/TextGrid"
    
    for speaker in tqdm(os.listdir(textgrid_dir)):
        speaker_dir = os.path.join(textgrid_dir, speaker)
        if not os.path.isdir(speaker_dir):
            continue
        
        textgrid_files = [f for f in os.listdir(speaker_dir) if f.endswith('.TextGrid')]
        
        for tg_file in textgrid_files:
            basename = tg_file.replace('.TextGrid', '')
            tg_path = os.path.join(speaker_dir, tg_file)
            
            try:
                # 提取IPA音素和时长
                result = extract_phonemes_and_duration(tg_path)
                if not result:
                    continue
                
                phonemes, durations = result
                
                # 检查对应的文本和情感
                original_files = [
                    "preprocessed_data/ESD-Chinese/train_original.txt",
                    "preprocessed_data/ESD-Chinese/val_original.txt"
                ]
                
                text_info = None
                for orig_file in original_files:
                    if os.path.exists(orig_file):
                        text_info = find_text_info(orig_file, basename, speaker)
                        if text_info:
                            break
                
                if not text_info:
                    continue
                
                raw_text, emotion = text_info
                
                # 构建数据行
                ipa_phonemes_str = "{" + " ".join(phonemes) + "}"
                data_line = f"{basename}|{speaker}|{ipa_phonemes_str}|{raw_text}|{speaker}|{raw_text}|{emotion}"
                
                # 保存特征文件
                save_features(speaker, basename, phonemes, durations)
                
                # 分配到训练集或验证集（简单分割）
                if len(train_data) < 17000:  # 大概比例
                    train_data.append(data_line)
                else:
                    val_data.append(data_line)
                    
            except Exception as e:
                print(f"跳过 {tg_path}: {e}")
                continue
    
    # 保存新的数据文件
    with open("preprocessed_data/ESD-Chinese/train_ipa.txt", "w", encoding="utf-8") as f:
        for line in train_data:
            f.write(line + "\n")
    
    with open("preprocessed_data/ESD-Chinese/val_ipa.txt", "w", encoding="utf-8") as f:
        for line in val_data:
            f.write(line + "\n")
    
    print(f"生成 {len(train_data)} 个训练样本")
    print(f"生成 {len(val_data)} 个验证样本")

def extract_phonemes_and_duration(textgrid_path):
    """从TextGrid提取音素和时长"""
    try:
        tg = textgrid.TextGrid.fromFile(textgrid_path)
        phone_tier = None
        
        for tier in tg.tiers:
            if tier.name.lower() in ['phones', 'phone']:
                phone_tier = tier
                break
        
        if phone_tier is None:
            return None
        
        phonemes = []
        durations = []
        
        for interval in phone_tier:
            phone = interval.mark.strip()
            duration_frames = int((interval.maxTime - interval.minTime) * 22050 / 256)
            
            if phone and phone != '':
                phonemes.append(phone)
                durations.append(max(1, duration_frames))
        
        return phonemes, durations
        
    except Exception as e:
        return None

def find_text_info(file_path, basename, speaker):
    """从原始文件中查找文本和情感信息"""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 6 and parts[0] == basename and parts[1] == speaker:
                raw_text = parts[3]
                emotion = parts[-1]
                return raw_text, emotion
    return None

def save_features(speaker, basename, phonemes, durations):
    """保存特征文件"""
    # Duration
    duration_path = f"preprocessed_data/ESD-Chinese/duration/{speaker}-duration-{basename}.npy"
    np.save(duration_path, np.array(durations))
    
    # 对于pitch和energy，如果原文件存在就复制调整，否则生成默认值
    pitch_path = f"preprocessed_data/ESD-Chinese/pitch/{speaker}-pitch-{basename}.npy"
    energy_path = f"preprocessed_data/ESD-Chinese/energy/{speaker}-energy-{basename}.npy"
    
    new_length = len(durations)
    
    if os.path.exists(pitch_path) and os.path.exists(energy_path):
        old_pitch = np.load(pitch_path)
        old_energy = np.load(energy_path)
        
        # 插值到新长度
        if len(old_pitch) != new_length:
            indices = np.linspace(0, len(old_pitch)-1, new_length)
            new_pitch = np.interp(indices, np.arange(len(old_pitch)), old_pitch)
            new_energy = np.interp(indices, np.arange(len(old_energy)), old_energy)
        else:
            new_pitch = old_pitch
            new_energy = old_energy
    else:
        # 生成默认值
        new_pitch = np.random.normal(5.0, 1.0, new_length)
        new_energy = np.random.normal(0.5, 0.2, new_length)
    
    np.save(pitch_path, new_pitch)
    np.save(energy_path, new_energy)

if __name__ == "__main__":
    extract_ipa_data()
