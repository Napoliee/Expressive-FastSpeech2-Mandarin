#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESD (Emotional Speech Dataset) 中文数据预处理脚本
只处理中文说话人（0001-0010），正确保留情绪标签信息
"""

import os
import shutil
from pathlib import Path
import tqdm
import json
import random
from collections import defaultdict

def prepare_esd_chinese_data():
    """预处理ESD中文数据集，保留情绪信息"""
    
    # 路径配置
    source_dir = "./Emotional Speech Dataset (ESD)/Emotion Speech Dataset"
    target_dir = "./raw_data/ESD-Chinese-Singing-MFA"
    
    print("🎯 开始处理ESD中文数据集...")
    
    # 清理并创建目标目录
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    
    # 情绪映射（中文到英文）
    emotion_mapping = {
        "中立": "Neutral",
        "开心": "Happy", 
        "伤心": "Sad",
        "愤怒": "Angry",
        "惊讶": "Surprise"
    }
    
    # 英文情绪到ID的映射
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
    
    # 存储所有文件信息（用于创建filelist）
    all_files_info = []
    
    # 只处理中文说话人（0001-0010）
    for speaker_id in range(1, 11):  # 0001-0010为中文
        speaker_folder = f"{speaker_id:04d}"
        source_speaker_dir = os.path.join(source_dir, speaker_folder)
        
        if not os.path.exists(source_speaker_dir):
            continue
            
        print(f"📂 处理中文说话人: {speaker_folder}")
        
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
                    
                    # 记录文件信息（使用实际的情绪文件夹名称）
                    actual_emotion_id = emotion_id_mapping[emotion_folder]
                    all_files_info.append({
                        'file_id': file_id,
                        'speaker_id': speaker_folder,
                        'speaker_idx': speaker_id - 1,  # 从0开始
                        'emotion_folder': emotion_folder,
                        'emotion_id': actual_emotion_id,
                        'text': text_dict[file_id]['text']
                    })
                    
                    total_files += 1
                    emotion_counts[emotion_folder] += 1
                else:
                    print(f"⚠️  找不到文本信息: {file_id}")
    
    # 创建训练文件列表（使用分层采样）
    create_stratified_split(target_dir, all_files_info)
    
    # 创建说话人信息文件
    create_speaker_info(target_dir, list(range(1, 11)))
    
    # 创建情绪信息文件
    create_emotion_info(target_dir, emotion_id_mapping)
    
    print(f"\n✅ 中文数据处理完成!")
    print(f"📊 总文件数: {total_files}")
    print(f"👥 说话人数: 10 (中文)")
    print(f"📈 情绪分布:")
    for emotion, count in emotion_counts.items():
        print(f"   {emotion}: {count}")

def create_stratified_split(data_dir, files_info, val_ratio=0.15, test_ratio=0.05):
    """创建分层采样的训练/验证/测试集划分"""
    print("📝 创建分层采样的数据划分...")
    
    # 设置随机种子保证可重复性
    random.seed(42)
    
    # 按说话人和情绪分组
    speaker_emotion_groups = defaultdict(lambda: defaultdict(list))
    
    for file_info in files_info:
        speaker_idx = file_info['speaker_idx']
        emotion_id = file_info['emotion_id']
        speaker_emotion_groups[speaker_idx][emotion_id].append(file_info)
    
    train_files = []
    val_files = []
    test_files = []
    
    # 为每个说话人的每种情绪进行分层采样
    for speaker_idx in speaker_emotion_groups:
        for emotion_id in speaker_emotion_groups[speaker_idx]:
            emotion_files = speaker_emotion_groups[speaker_idx][emotion_id]
            random.shuffle(emotion_files)  # 随机打乱
            
            n_files = len(emotion_files)
            n_test = max(1, int(n_files * test_ratio))  # 至少1个测试样本
            n_val = max(1, int(n_files * val_ratio))    # 至少1个验证样本
            n_train = n_files - n_test - n_val
            
            # 分配文件
            test_files.extend(emotion_files[:n_test])
            val_files.extend(emotion_files[n_test:n_test + n_val])
            train_files.extend(emotion_files[n_test + n_val:])
    
    # 创建文件列表格式：file_id|speaker_id|emotion_id|text
    def create_filelist(files):
        filelist = []
        for file_info in files:
            line = f"{file_info['file_id']}|{file_info['speaker_idx']}|{file_info['emotion_id']}|{file_info['text']}"
            filelist.append(line)
        return filelist
    
    train_list = create_filelist(train_files)
    val_list = create_filelist(val_files)
    test_list = create_filelist(test_files)
    
    # 保存文件列表
    with open(os.path.join(data_dir, "train.txt"), 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_list))
    
    with open(os.path.join(data_dir, "val.txt"), 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_list))
    
    with open(os.path.join(data_dir, "test.txt"), 'w', encoding='utf-8') as f:
        f.write('\n'.join(test_list))
    
    # 保存完整文件列表
    all_list = train_list + val_list + test_list
    with open(os.path.join(data_dir, "filelist.txt"), 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_list))
    
    # 统计信息
    print(f"   📊 数据划分统计:")
    print(f"     训练集: {len(train_list)} 条目 ({len(train_list)/len(all_list)*100:.1f}%)")
    print(f"     验证集: {len(val_list)} 条目 ({len(val_list)/len(all_list)*100:.1f}%)")
    print(f"     测试集: {len(test_list)} 条目 ({len(test_list)/len(all_list)*100:.1f}%)")
    
    # 验证每个集合中说话人和情绪的分布
    validate_split_distribution(train_files, val_files, test_files)

def validate_split_distribution(train_files, val_files, test_files):
    """验证数据划分的分布是否合理"""
    print("   🔍 验证数据分布:")
    
    def get_distribution(files, name):
        speaker_count = defaultdict(int)
        emotion_count = defaultdict(int)
        
        for file_info in files:
            speaker_count[file_info['speaker_idx']] += 1
            emotion_count[file_info['emotion_id']] += 1
        
        print(f"     {name}:")
        print(f"       说话人分布: {dict(speaker_count)}")
        print(f"       情绪分布: {dict(emotion_count)}")
    
    get_distribution(train_files, "训练集")
    get_distribution(val_files, "验证集")
    get_distribution(test_files, "测试集")

def create_speaker_info(data_dir, speaker_ids):
    """创建说话人信息文件"""
    print("👥 创建说话人信息...")
    
    speakers = [f"{sid:04d}" for sid in speaker_ids]
    
    speaker_info = {
        'n_speakers': len(speakers),
        'speakers': {str(i): speaker for i, speaker in enumerate(speakers)}
    }
    
    with open(os.path.join(data_dir, "speakers.json"), 'w', encoding='utf-8') as f:
        json.dump(speaker_info, f, ensure_ascii=False, indent=2)
    
    print(f"   记录了 {len(speakers)} 个中文说话人")

def create_emotion_info(data_dir, emotion_mapping):
    """创建情绪信息文件"""
    print("😊 创建情绪信息...")
    
    emotion_info = {
        'n_emotions': len(emotion_mapping),
        'emotions': {str(v): k for k, v in emotion_mapping.items()}
    }
    
    with open(os.path.join(data_dir, "emotions.json"), 'w', encoding='utf-8') as f:
        json.dump(emotion_info, f, ensure_ascii=False, indent=2)
    
    print(f"   记录了 {len(emotion_mapping)} 种情绪")
    print("   情绪映射:")
    for emotion_id, emotion_name in emotion_info['emotions'].items():
        print(f"     {emotion_id}: {emotion_name}")

if __name__ == "__main__":
    prepare_esd_chinese_data() 