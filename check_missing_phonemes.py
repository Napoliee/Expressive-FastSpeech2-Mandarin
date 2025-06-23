#!/usr/bin/env python3
"""
检查训练数据中的音素，找出symbols表中缺失的
"""

import re
from collections import Counter
from text.symbols_ipa import _ipa_phonemes

def extract_all_phonemes():
    """从训练数据中提取所有音素"""
    
    all_phonemes = set()
    
    # 读取训练和验证数据
    files = [
        "preprocessed_data/ESD-Chinese/train_ipa.txt",
        "preprocessed_data/ESD-Chinese/val_ipa.txt"
    ]
    
    for file_path in files:
        print(f"处理文件: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 3:
                    phoneme_text = parts[2]  # IPA音素字段
                    
                    # 提取花括号中的音素
                    if phoneme_text.startswith('{') and phoneme_text.endswith('}'):
                        phonemes = phoneme_text[1:-1].split()
                        all_phonemes.update(phonemes)
    
    return all_phonemes

def check_missing_phonemes():
    """检查缺失的音素"""
    
    print("=== 检查缺失的音素 ===")
    
    # 获取训练数据中的所有音素
    training_phonemes = extract_all_phonemes()
    print(f"训练数据中发现 {len(training_phonemes)} 个唯一音素")
    
    # 获取当前symbols表中的音素（去掉@前缀）
    current_phonemes = set()
    for symbol in _ipa_phonemes:
        if symbol.startswith('@'):
            current_phonemes.add(symbol[1:])  # 去掉@前缀
    
    print(f"symbols表中有 {len(current_phonemes)} 个音素")
    
    # 找出缺失的音素
    missing_phonemes = training_phonemes - current_phonemes
    extra_phonemes = current_phonemes - training_phonemes
    
    print(f"\n缺失的音素 ({len(missing_phonemes)} 个):")
    for phoneme in sorted(missing_phonemes):
        print(f"  '{phoneme}'")
    
    print(f"\n多余的音素 ({len(extra_phonemes)} 个):")
    for phoneme in sorted(extra_phonemes):
        print(f"  '{phoneme}'")
    
    # 统计缺失音素的使用频率
    if missing_phonemes:
        print(f"\n缺失音素的使用频率:")
        phoneme_counter = Counter()
        
        files = [
            "preprocessed_data/ESD-Chinese/train_ipa.txt",
            "preprocessed_data/ESD-Chinese/val_ipa.txt"
        ]
        
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('|')
                    if len(parts) >= 3:
                        phoneme_text = parts[2]
                        if phoneme_text.startswith('{') and phoneme_text.endswith('}'):
                            phonemes = phoneme_text[1:-1].split()
                            for phoneme in phonemes:
                                if phoneme in missing_phonemes:
                                    phoneme_counter[phoneme] += 1
        
        for phoneme, count in phoneme_counter.most_common():
            print(f"  '{phoneme}': {count} 次")
    
    return missing_phonemes, extra_phonemes

def generate_updated_symbols():
    """生成更新后的symbols表"""
    
    missing_phonemes, extra_phonemes = check_missing_phonemes()
    
    if missing_phonemes:
        print(f"\n=== 生成更新的symbols表 ===")
        
        # 当前的音素列表
        current_phonemes = []
        for symbol in _ipa_phonemes:
            if symbol.startswith('@'):
                current_phonemes.append(symbol[1:])
        
        # 添加缺失的音素
        all_phonemes = set(current_phonemes) | missing_phonemes
        sorted_phonemes = sorted(all_phonemes)
        
        # 生成新的symbols列表
        new_ipa_phonemes = ['@' + phoneme for phoneme in sorted_phonemes]
        
        print("新的_ipa_phonemes列表:")
        print("_ipa_phonemes = [")
        for i, phoneme in enumerate(new_ipa_phonemes):
            if i % 10 == 0:
                print("    ", end="")
            print(f"'{phoneme}', ", end="")
            if (i + 1) % 10 == 0:
                print()
        if len(new_ipa_phonemes) % 10 != 0:
            print()
        print("]")
        
        print(f"\n总音素数: {len(new_ipa_phonemes)}")
        print(f"添加了 {len(missing_phonemes)} 个新音素")

if __name__ == "__main__":
    generate_updated_symbols() 