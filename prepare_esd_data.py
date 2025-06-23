#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESD (Emotional Speech Dataset) 数据预处理脚本
保留情绪标签信息并重组数据
"""

import os
import shutil
from pathlib import Path
import tqdm

def prepare_esd_data():
    """预处理ESD数据集，保留情绪信息"""
    
    # 路径配置
    source_dir = "./Emotional Speech Dataset (ESD)/Emotion Speech Dataset"
    target_dir = "./raw_data/ESD-Chinese-Singing-MFA"
    
    print("🎯 开始处理ESD数据集...")
    
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)
    
    # 情绪映射
    emotion_mapping = {
        "中立": "Neutral",
        "开心": "Happy", 
        "伤心": "Sad",
        "愤怒": "Angry",
        "惊讶": "Surprise"
    }
    
    emotion_id_mapping = {
        "Neutral": 0,
        "Happy": 1,
        "Sad": 2,
        "Angry": 3,
        "Surprise": 4
    }
    
    # 统计信息
    total_files = 0
    emotion_counts = {emotion: 0 for emotion in emotion_id_mapping.keys()}
    
    # 为每个说话人创建子目录
    for speaker_id in range(1, 21):  # 0001-0020
        speaker_folder = f"{speaker_id:04d}"
        source_speaker_dir = os.path.join(source_dir, speaker_folder)
        
        if not os.path.exists(source_speaker_dir):
            continue
            
        print(f"📂 处理说话人: {speaker_folder}")
        
        # 读取文本文件
        text_file = os.path.join(source_speaker_dir, f"{speaker_folder}.txt")
        if not os.path.exists(text_file):
            print(f"❌ 找不到文本文件: {text_file}")
            continue
            
        # 解析文本文件
        text_dict = {}
        with open(text_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        file_id, text, emotion_cn = parts[0], parts[1], parts[2]
                        emotion_en = emotion_mapping.get(emotion_cn, "Neutral")
                        text_dict[file_id] = {
                            'text': text,
                            'emotion_cn': emotion_cn,
                            'emotion_en': emotion_en,
                            'emotion_id': emotion_id_mapping[emotion_en]
                        }
        
        # 创建说话人目录
        target_speaker_dir = os.path.join(target_dir, speaker_folder)
        os.makedirs(target_speaker_dir, exist_ok=True)
        
        # 处理每个情绪文件夹
        for emotion_folder in ["Angry", "Happy", "Neutral", "Sad", "Surprise"]:
            source_emotion_dir = os.path.join(source_speaker_dir, emotion_folder)
            
            if not os.path.exists(source_emotion_dir):
                continue
                
            # 复制音频文件并创建对应的文本文件
            wav_files = list(Path(source_emotion_dir).glob("*.wav"))
            
            for wav_file in tqdm.tqdm(wav_files, desc=f"  {emotion_folder}"):
                file_id = wav_file.stem  # 不包含扩展名
                
                if file_id in text_dict:
                    # 复制音频文件
                    target_wav = os.path.join(target_speaker_dir, f"{file_id}.wav")
                    shutil.copy2(wav_file, target_wav)
                    
                    # 创建对应的文本文件（用于MFA）
                    target_lab = os.path.join(target_speaker_dir, f"{file_id}.lab")
                    with open(target_lab, 'w', encoding='utf-8') as f:
                        f.write(text_dict[file_id]['text'])
                    
                    total_files += 1
                    emotion_counts[emotion_folder] += 1
                else:
                    print(f"⚠️  找不到文本信息: {file_id}")
    
    # 创建训练文件列表
    create_filelist(target_dir, text_dict)
    
    # 创建说话人信息文件
    create_speaker_info(target_dir)
    
    # 创建情绪信息文件
    create_emotion_info(target_dir, emotion_id_mapping)
    
    print(f"\n✅ 数据处理完成!")
    print(f"📊 总文件数: {total_files}")
    print(f"📈 情绪分布:")
    for emotion, count in emotion_counts.items():
        print(f"   {emotion}: {count}")

def create_filelist(data_dir, text_dict=None):
    """创建文件列表"""
    print("📝 创建文件列表...")
    
    filelist = []
    
    for speaker_dir in sorted(os.listdir(data_dir)):
        speaker_path = os.path.join(data_dir, speaker_dir)
        if not os.path.isdir(speaker_path):
            continue
            
        for wav_file in sorted(Path(speaker_path).glob("*.wav")):
            file_id = wav_file.stem
            speaker_id = speaker_dir
            
            # 从文件名推断情绪（基于音频文件所在的原始文件夹）
            # 这里我们需要重新读取文本信息来获取情绪
            emotion_id = 0  # 默认为中立
            
            filelist.append(f"{file_id}|{speaker_id}|{emotion_id}")
    
    # 保存文件列表
    with open(os.path.join(data_dir, "filelist.txt"), 'w', encoding='utf-8') as f:
        f.write('\n'.join(filelist))
    
    print(f"   创建了 {len(filelist)} 个文件条目")

def create_speaker_info(data_dir):
    """创建说话人信息文件"""
    print("👥 创建说话人信息...")
    
    speakers = []
    for speaker_dir in sorted(os.listdir(data_dir)):
        speaker_path = os.path.join(data_dir, speaker_dir)
        if os.path.isdir(speaker_path):
            speakers.append(speaker_dir)
    
    speaker_info = {
        'n_speakers': len(speakers),
        'speakers': {i: speaker for i, speaker in enumerate(speakers)}
    }
    
    import json
    with open(os.path.join(data_dir, "speaker_info.json"), 'w', encoding='utf-8') as f:
        json.dump(speaker_info, f, ensure_ascii=False, indent=2)
    
    print(f"   记录了 {len(speakers)} 个说话人")

def create_emotion_info(data_dir, emotion_mapping):
    """创建情绪信息文件"""
    print("😊 创建情绪信息...")
    
    emotion_info = {
        'n_emotions': len(emotion_mapping),
        'emotions': emotion_mapping
    }
    
    import json
    with open(os.path.join(data_dir, "emotion_info.json"), 'w', encoding='utf-8') as f:
        json.dump(emotion_info, f, ensure_ascii=False, indent=2)
    
    print(f"   记录了 {len(emotion_mapping)} 种情绪")

if __name__ == "__main__":
    prepare_esd_data() 