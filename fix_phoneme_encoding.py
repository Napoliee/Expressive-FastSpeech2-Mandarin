import os
import json
from tqdm import tqdm

def create_phoneme_mapping():
    """创建从IPA音素到数字编码的映射"""
    # 收集所有唯一的音素
    unique_phonemes = set()
    
    print("收集所有唯一音素...")
    for filepath in ["preprocessed_data/ESD-Chinese/train.txt", "preprocessed_data/ESD-Chinese/val.txt"]:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in tqdm(f.readlines()):
                line_split = line.strip().split("|")
                phoneme_text = line_split[2]  # 音素字段
                
                if phoneme_text.startswith('{') and phoneme_text.endswith('}'):
                    phonemes = phoneme_text[1:-1].split()
                    unique_phonemes.update(phonemes)
    
    # 创建映射表 (添加特殊标记)
    phoneme_list = ['_PAD_', '_UNK_'] + sorted(list(unique_phonemes))
    phoneme_to_id = {phoneme: i for i, phoneme in enumerate(phoneme_list)}
    id_to_phoneme = {i: phoneme for i, phoneme in enumerate(phoneme_list)}
    
    print(f"发现 {len(unique_phonemes)} 个唯一音素")
    print("前20个音素:", phoneme_list[:22])
    
    # 保存映射表
    mapping = {
        'phoneme_to_id': phoneme_to_id,
        'id_to_phoneme': id_to_phoneme,
        'phoneme_list': phoneme_list
    }
    
    with open("preprocessed_data/ESD-Chinese/phoneme_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    
    return phoneme_to_id

def convert_training_data(phoneme_to_id):
    """转换训练数据，将音素序列替换为数字编码序列"""
    
    for split in ["train", "val"]:
        input_file = f"preprocessed_data/ESD-Chinese/{split}.txt"
        output_file = f"preprocessed_data/ESD-Chinese/{split}_encoded.txt"
        
        print(f"转换 {split} 数据...")
        
        with open(input_file, "r", encoding="utf-8") as f_in, \
             open(output_file, "w", encoding="utf-8") as f_out:
            
            for line in tqdm(f_in.readlines()):
                line_split = line.strip().split("|")
                
                if len(line_split) >= 3:
                    basename = line_split[0]
                    speaker = line_split[1]
                    phoneme_text = line_split[2]
                    
                    # 处理音素序列
                    if phoneme_text.startswith('{') and phoneme_text.endswith('}'):
                        phonemes = phoneme_text[1:-1].split()
                        # 转换为数字序列
                        phoneme_ids = [phoneme_to_id.get(p, 1) for p in phonemes]  # 1是_UNK_的ID
                        phoneme_encoded = " ".join(map(str, phoneme_ids))
                    else:
                        # 如果格式不正确，跳过
                        continue
                    
                    # 重构行
                    new_line_parts = [basename, speaker, phoneme_encoded] + line_split[3:]
                    f_out.write("|".join(new_line_parts) + "\n")

def main():
    print("开始修复音素编码问题...")
    
    # 创建音素映射
    phoneme_to_id = create_phoneme_mapping()
    
    # 转换训练数据
    convert_training_data(phoneme_to_id)
    
    print("完成！生成的文件:")
    print("- preprocessed_data/ESD-Chinese/phoneme_mapping.json")
    print("- preprocessed_data/ESD-Chinese/train_encoded.txt")
    print("- preprocessed_data/ESD-Chinese/val_encoded.txt")
    
    print("\n接下来需要:")
    print("1. 备份原始文件")
    print("2. 将编码后的文件重命名为原始文件名")
    print("3. 修改dataset.py以使用数字编码而不是text_to_sequence")

if __name__ == "__main__":
    main() 