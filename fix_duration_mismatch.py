import os
import numpy as np
import textgrid
from tqdm import tqdm
import json

def extract_phonemes_from_textgrid(textgrid_path):
    """从TextGrid文件提取音素序列（只包含非空音素）"""
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
            duration_frames = int((interval.maxTime - interval.minTime) * 22050 / 256)  # hop_length=256
            
            # 只保留非空音素
            if phone and phone != '' and phone != 'sil':
                phonemes.append(phone)
                durations.append(max(1, duration_frames))  # 至少1帧
        
        return phonemes, durations
        
    except Exception as e:
        print(f"Error processing {textgrid_path}: {e}")
        return None

def rebuild_duration_features():
    """重建Duration特征，确保与音素编码一致"""
    
    print("=== 重建Duration特征 ===")
    
    # 加载音素映射
    with open("preprocessed_data/ESD-Chinese/phoneme_mapping.json", "r", encoding="utf-8") as f:
        mapping = json.load(f)
    phoneme_to_id = mapping["phoneme_to_id"]
    
    # 处理训练数据
    for split in ["train", "val"]:
        print(f"\n处理 {split} 数据...")
        
        data_file = f"preprocessed_data/ESD-Chinese/{split}.txt"
        
        with open(data_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        updated_lines = []
        mismatch_count = 0
        
        for line in tqdm(lines):
            parts = line.strip().split("|")
            if len(parts) < 4:
                updated_lines.append(line)
                continue
            
            basename = parts[0]
            speaker = parts[1]
            phoneme_ids = parts[2].split()
            
            # 检查TextGrid文件
            textgrid_path = f"preprocessed_data/ESD-Chinese/TextGrid/{speaker}/{basename}.TextGrid"
            
            if os.path.exists(textgrid_path):
                result = extract_phonemes_from_textgrid(textgrid_path)
                if result:
                    textgrid_phonemes, textgrid_durations = result
                    
                    # 转换TextGrid音素为ID序列
                    textgrid_ids = [phoneme_to_id.get(p, 1) for p in textgrid_phonemes]
                    
                    # 检查是否匹配当前的编码序列
                    if len(textgrid_ids) != len(phoneme_ids):
                        mismatch_count += 1
                        
                        # 使用TextGrid的音素和时长
                        new_phoneme_ids = " ".join(map(str, textgrid_ids))
                        parts[2] = new_phoneme_ids
                        
                        # 保存新的Duration特征
                        duration_path = f"preprocessed_data/ESD-Chinese/duration/{speaker}-duration-{basename}.npy"
                        np.save(duration_path, np.array(textgrid_durations))
                        
                        # 同时需要重新生成pitch和energy特征以匹配新长度
                        pitch_path = f"preprocessed_data/ESD-Chinese/pitch/{speaker}-pitch-{basename}.npy"
                        energy_path = f"preprocessed_data/ESD-Chinese/energy/{speaker}-energy-{basename}.npy"
                        
                        if os.path.exists(pitch_path) and os.path.exists(energy_path):
                            old_pitch = np.load(pitch_path)
                            old_energy = np.load(energy_path)
                            
                            # 插值调整到新长度
                            new_length = len(textgrid_durations)
                            old_length = len(old_pitch)
                            
                            if old_length != new_length:
                                # 简单线性插值
                                indices = np.linspace(0, old_length-1, new_length)
                                new_pitch = np.interp(indices, np.arange(old_length), old_pitch)
                                new_energy = np.interp(indices, np.arange(old_length), old_energy)
                                
                                np.save(pitch_path, new_pitch)
                                np.save(energy_path, new_energy)
            
            updated_lines.append("|".join(parts) + "\n")
        
        # 保存更新的数据文件
        with open(data_file, "w", encoding="utf-8") as f:
            f.writelines(updated_lines)
        
        print(f"{split}: 修复了 {mismatch_count} 个样本")

def verify_consistency():
    """验证修复后的一致性"""
    print("\n=== 验证修复结果 ===")
    
    with open("preprocessed_data/ESD-Chinese/train.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()[:5]
    
    for i, line in enumerate(lines):
        parts = line.strip().split("|")
        if len(parts) < 4:
            continue
        
        basename = parts[0]
        speaker = parts[1]
        phoneme_ids = parts[2].split()
        text = parts[3]
        
        # 检查各种特征长度
        duration_path = f"preprocessed_data/ESD-Chinese/duration/{speaker}-duration-{basename}.npy"
        pitch_path = f"preprocessed_data/ESD-Chinese/pitch/{speaker}-pitch-{basename}.npy"
        energy_path = f"preprocessed_data/ESD-Chinese/energy/{speaker}-energy-{basename}.npy"
        
        if all(os.path.exists(p) for p in [duration_path, pitch_path, energy_path]):
            duration = np.load(duration_path)
            pitch = np.load(pitch_path)
            energy = np.load(energy_path)
            
            print(f"\n样本{i+1}: {text}")
            print(f"  音素数量: {len(phoneme_ids)}")
            print(f"  Duration长度: {len(duration)}")
            print(f"  Pitch长度: {len(pitch)}")
            print(f"  Energy长度: {len(energy)}")
            print(f"  Duration总和: {duration.sum()}")
            
            if len(phoneme_ids) == len(duration) == len(pitch) == len(energy):
                print("  ✅ 所有长度一致")
            else:
                print("  ❌ 长度仍不一致")

if __name__ == "__main__":
    print("🔧 修复音素-Duration不匹配问题")
    print("=" * 50)
    
    rebuild_duration_features()
    verify_consistency()
    
    print("\n✅ 修复完成！现在可以重新测试推理效果。") 