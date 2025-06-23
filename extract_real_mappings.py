#!/usr/bin/env python3
"""
从训练数据中提取真实的字符到音素映射
"""

import re
from collections import defaultdict, Counter

def extract_real_char_phoneme_mappings():
    """从训练数据中提取真实的字符到音素映射"""
    
    char_to_phonemes = defaultdict(list)
    
    # 读取训练数据
    with open("preprocessed_data/ESD-Chinese/train_ipa.txt", 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 4:
                phoneme_text = parts[2]  # IPA音素字段 "{...}"
                raw_text = parts[3]     # 原文
                
                if phoneme_text.startswith('{') and phoneme_text.endswith('}'):
                    phonemes = phoneme_text[1:-1].split()
                    
                    # 移除空格的原文
                    clean_text = raw_text.replace(' ', '')
                    
                    # 尝试对齐字符和音素（简单策略）
                    if len(clean_text) > 0:
                        # 记录每个字符对应的音素段
                        for char in clean_text:
                            if '\u4e00' <= char <= '\u9fff':  # 中文字符
                                char_to_phonemes[char].append(phonemes)
    
    # 统计每个字符最常见的音素模式
    char_mappings = {}
    for char, phoneme_lists in char_to_phonemes.items():
        # 计算音素模式的频率
        pattern_counter = Counter()
        for phoneme_list in phoneme_lists:
            # 将音素列表转换为字符串模式
            pattern = ' '.join(phoneme_list)
            pattern_counter[pattern] += 1
        
        # 选择最常见的模式
        if pattern_counter:
            most_common_pattern = pattern_counter.most_common(1)[0][0]
            char_mappings[char] = most_common_pattern.split()
            
            # 只显示有足够样本的字符
            if pattern_counter.most_common(1)[0][1] >= 2:
                print(f"'{char}': {char_mappings[char]} (出现{pattern_counter.most_common(1)[0][1]}次)")
    
    return char_mappings

def create_safe_chinese_to_ipa_mapping():
    """创建基于真实训练数据的安全映射"""
    
    print("=== 从训练数据中提取字符到音素映射 ===")
    
    char_mappings = extract_real_char_phoneme_mappings()
    
    print(f"\n总共提取了 {len(char_mappings)} 个字符的映射")
    
    # 生成Python代码
    print("\n=== 生成Python字典代码 ===")
    print("char_to_ipa = {")
    
    # 按拼音顺序排序
    for char in sorted(char_mappings.keys()):
        phonemes = char_mappings[char]
        phonemes_str = "', '".join(phonemes)
        print(f"    '{char}': ['{phonemes_str}'],")
    
    print("}")
    
    return char_mappings

if __name__ == "__main__":
    create_safe_chinese_to_ipa_mapping() 