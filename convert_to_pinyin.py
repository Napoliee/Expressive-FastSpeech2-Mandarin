#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量转换中文数据为拼音
"""

import os
import shutil
from tqdm import tqdm
from pypinyin import lazy_pinyin, Style

def chinese_to_pinyin(text):
    """将中文转换为拼音（带声调数字）"""
    # 移除标点符号，只保留中文字符
    chinese_chars = ''.join([char for char in text if '\u4e00' <= char <= '\u9fff'])
    
    if not chinese_chars:
        return ""
    
    # 转换为拼音
    pinyin_list = lazy_pinyin(chinese_chars, style=Style.TONE3, neutral_tone_with_five=True)
    
    # 用空格连接拼音
    pinyin_text = " ".join(pinyin_list)
    return pinyin_text

def convert_esd_to_pinyin():
    """转换ESD数据集为拼音版本"""
    
    # 源目录和目标目录
    source_dir = "./raw_data/ESD-Chinese"
    target_dir = "./raw_data/ESD-Chinese-Pinyin"
    
    print("🚀 开始批量转换中文到拼音")
    print(f"📁 源目录: {source_dir}")
    print(f"📁 目标目录: {target_dir}")
    
    if not os.path.exists(source_dir):
        print(f"❌ 源目录不存在: {source_dir}")
        return
    
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)
    
    # 读取filelist
    filelist_path = os.path.join(source_dir, "filelist.txt")
    if not os.path.exists(filelist_path):
        print(f"❌ 找不到filelist: {filelist_path}")
        return
    
    # 统计信息
    total_files = 0
    converted_files = 0
    pinyin_entries = []
    
    print("📝 读取原始数据...")
    
    with open(filelist_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"📊 找到 {len(lines)} 条数据")
    print("🔄 开始转换...")
    
    for line in tqdm(lines):
        line = line.strip()
        if not line:
            continue
            
        parts = line.split('|')
        if len(parts) < 6:
            continue
            
        wav_path, text, speaker_id = parts[0], parts[1], parts[2]
        emotion = parts[5]  # 第6个字段是情感
        total_files += 1
        
        # 转换中文为拼音
        pinyin_text = chinese_to_pinyin(text)
        
        if not pinyin_text:
            print(f"⚠️  跳过空拼音: {text}")
            continue
        
        # 创建说话人目录
        speaker_target_dir = os.path.join(target_dir, speaker_id)
        os.makedirs(speaker_target_dir, exist_ok=True)
        
        # 文件名
        basename = os.path.splitext(os.path.basename(wav_path))[0]
        
        # 实际音频文件路径：音频文件在说话人目录下，不在clips子目录
        actual_source_wav = os.path.join(source_dir, speaker_id, f"{basename}.wav")
        target_wav = os.path.join(speaker_target_dir, f"{basename}.wav")
        
        if os.path.exists(actual_source_wav):
            if not os.path.exists(target_wav):
                shutil.copy2(actual_source_wav, target_wav)
            
            # 创建拼音lab文件
            lab_path = os.path.join(speaker_target_dir, f"{basename}.lab")
            with open(lab_path, 'w', encoding='utf-8') as f:
                f.write(pinyin_text)
            
            # 记录拼音版本的filelist条目
            pinyin_wav_path = os.path.join(speaker_id, f"{basename}.wav")
            pinyin_entries.append(f"{pinyin_wav_path}|{pinyin_text}|{speaker_id}|{emotion}")
            
            converted_files += 1
            
            # 显示前几个转换示例
            if converted_files <= 10:
                print(f"📝 示例 {converted_files}: {text} → {pinyin_text}")
    
    # 保存拼音版本的filelist
    pinyin_filelist_path = os.path.join(target_dir, "filelist.txt")
    with open(pinyin_filelist_path, 'w', encoding='utf-8') as f:
        for entry in pinyin_entries:
            f.write(entry + '\n')
    
    print(f"\n✅ 拼音转换完成!")
    print(f"📊 统计信息:")
    print(f"   总文件数: {total_files}")
    print(f"   成功转换: {converted_files}")
    print(f"   转换率: {converted_files/total_files*100:.1f}%")
    print(f"📁 拼音数据目录: {target_dir}")
    print(f"📋 拼音filelist: {pinyin_filelist_path}")
    
    return target_dir

def test_pinyin_conversion():
    """测试拼音转换功能"""
    
    test_sentences = [
        "他对谁都那么友好。",
        "今天天气真不错。",
        "我很高兴见到你。",
        "这是一个测试句子。",
        "语音合成技术很有趣。"
    ]
    
    print("🧪 测试拼音转换:")
    for sentence in test_sentences:
        pinyin = chinese_to_pinyin(sentence)
        print(f"   {sentence} → {pinyin}")

if __name__ == "__main__":
    # 先测试转换功能
    test_pinyin_conversion()
    
    print("\n" + "="*50)
    
    # 批量转换
    pinyin_dir = convert_esd_to_pinyin()
    
    if pinyin_dir:
        print(f"\n🎯 下一步: 运行MFA拼音对齐")
        print(f"命令: conda run -n aligner mfa align {pinyin_dir} mandarin_pinyin mandarin_mfa ./preprocessed_data/ESD-Chinese-Pinyin/TextGrid") 