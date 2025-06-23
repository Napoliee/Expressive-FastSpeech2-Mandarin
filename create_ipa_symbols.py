#!/usr/bin/env python3
"""
收集MFA生成的所有IPA音素，创建新的符号表
"""

import os
import textgrid
from collections import Counter
from tqdm import tqdm

def collect_all_ipa_phonemes():
    """从所有TextGrid文件中收集IPA音素"""
    
    print("=== 收集MFA生成的所有IPA音素 ===")
    
    phoneme_counter = Counter()
    textgrid_dir = "preprocessed_data/ESD-Chinese/TextGrid"
    
    # 遍历所有TextGrid文件
    for speaker in os.listdir(textgrid_dir):
        speaker_dir = os.path.join(textgrid_dir, speaker)
        if not os.path.isdir(speaker_dir):
            continue
            
        textgrid_files = [f for f in os.listdir(speaker_dir) if f.endswith('.TextGrid')]
        
        for tg_file in tqdm(textgrid_files, desc=f"处理说话人{speaker}"):
            tg_path = os.path.join(speaker_dir, tg_file)
            
            try:
                tg = textgrid.TextGrid.fromFile(tg_path)
                
                # 查找phones层
                phone_tier = None
                for tier in tg.tiers:
                    if tier.name.lower() in ['phones', 'phone']:
                        phone_tier = tier
                        break
                
                if phone_tier:
                    for interval in phone_tier:
                        phone = interval.mark.strip()
                        if phone and phone != '':
                            phoneme_counter[phone] += 1
                            
            except Exception as e:
                print(f"跳过文件 {tg_path}: {e}")
                continue
    
    print(f"\n发现 {len(phoneme_counter)} 个唯一音素")
    print("音素使用频率（前20个）:")
    for phone, count in phoneme_counter.most_common(20):
        print(f"  {phone}: {count}")
    
    return list(phoneme_counter.keys())

def create_ipa_symbols_file(ipa_phonemes):
    """创建基于IPA的symbols.py文件"""
    
    print(f"\n=== 创建IPA符号表 ===")
    
    # 基本符号
    _pad = "_"
    _punctuation = "!'(),.:;? "
    _special = "-"
    _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    
    # 将IPA音素按前缀分组，添加@前缀确保唯一性
    _ipa_phonemes = ["@" + p for p in sorted(ipa_phonemes)]
    
    # 生成symbols.py内容
    symbols_content = f'''""" IPA-based symbols for ESD-Chinese dataset """

_pad = "{_pad}"
_punctuation = "{_punctuation}"
_special = "{_special}"
_letters = "{_letters}"

# IPA phonemes from MFA alignment (with @ prefix for uniqueness)
_ipa_phonemes = {_ipa_phonemes}

# Export all symbols
symbols = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    + list(_letters)
    + _ipa_phonemes
)

# Create symbol to ID mapping
_symbol_to_id = {{s: i for i, s in enumerate(symbols)}}
_id_to_symbol = {{i: s for i, s in enumerate(symbols)}}
'''

    # 保存新的symbols.py
    with open("text/symbols_ipa.py", "w", encoding="utf-8") as f:
        f.write(symbols_content)
    
    print(f"创建了新的符号表: text/symbols_ipa.py")
    print(f"总符号数: {len(_ipa_phonemes) + len(_pad + _special + _punctuation + _letters)}")
    
    return _ipa_phonemes

def create_ipa_text_processor():
    """创建IPA文本处理器"""
    
    ipa_processor_content = '''"""
IPA-based text processing for ESD-Chinese
"""

import re
import numpy as np
from text.symbols_ipa import symbols, _symbol_to_id, _id_to_symbol

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r"(.*?)\\{(.+?)\\}(.*)")

def text_to_sequence_ipa(text, cleaner_names=None):
    """
    Convert IPA phoneme text to sequence of IDs
    
    Args:
        text: IPA phoneme string like "{t w ej˥˩ ʂ ej˧˥ spn n a˥˩}"
        cleaner_names: ignored for IPA processing
    
    Returns:
        List of integers corresponding to the phonemes
    """
    sequence = []
    
    # Check for curly braces and extract phonemes
    if text.startswith('{') and text.endswith('}'):
        # Extract phonemes from curly braces
        phonemes = text[1:-1].split()
        sequence = _phonemes_to_sequence(phonemes)
    else:
        # Treat as space-separated phonemes
        phonemes = text.split()
        sequence = _phonemes_to_sequence(phonemes)
    
    return sequence

def _phonemes_to_sequence(phonemes):
    """Convert phoneme list to ID sequence"""
    sequence = []
    for phoneme in phonemes:
        # Add @ prefix for IPA phonemes
        ipa_symbol = "@" + phoneme
        if ipa_symbol in _symbol_to_id:
            sequence.append(_symbol_to_id[ipa_symbol])
        else:
            # Unknown phoneme, use a default
            print(f"Warning: Unknown phoneme '{phoneme}', using '@spn'")
            if "@spn" in _symbol_to_id:
                sequence.append(_symbol_to_id["@spn"])
            else:
                sequence.append(1)  # UNK token
    
    return sequence

def sequence_to_text_ipa(sequence):
    """Convert sequence back to text"""
    result = []
    for id in sequence:
        if id < len(symbols):
            symbol = symbols[id]
            if symbol.startswith('@'):
                result.append(symbol[1:])  # Remove @ prefix
            else:
                result.append(symbol)
    return ' '.join(result)
'''

    with open("text/ipa_processor.py", "w", encoding="utf-8") as f:
        f.write(ipa_processor_content)
    
    print("创建了IPA文本处理器: text/ipa_processor.py")

def create_preprocessing_script():
    """创建重新预处理的脚本"""
    
    script_content = '''#!/usr/bin/env python3
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
            f.write(line + "\\n")
    
    with open("preprocessed_data/ESD-Chinese/val_ipa.txt", "w", encoding="utf-8") as f:
        for line in val_data:
            f.write(line + "\\n")
    
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
'''

    with open("reprocess_with_ipa.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print("创建了重新预处理脚本: reprocess_with_ipa.py")

def main():
    print("🎵 创建基于MFA IPA音素的新处理系统")
    print("=" * 50)
    
    # 1. 收集所有IPA音素
    ipa_phonemes = collect_all_ipa_phonemes()
    
    # 2. 创建新的符号表
    create_ipa_symbols_file(ipa_phonemes)
    
    # 3. 创建IPA文本处理器
    create_ipa_text_processor()
    
    # 4. 创建重新预处理脚本
    create_preprocessing_script()
    
    print("\\n✅ 完成！接下来的步骤:")
    print("1. 运行: python reprocess_with_ipa.py")
    print("2. 修改dataset.py使用新的IPA处理器")
    print("3. 修改符号表导入")
    print("4. 重新训练模型")

if __name__ == "__main__":
    main() 