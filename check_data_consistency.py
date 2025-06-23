import os
import numpy as np
from tqdm import tqdm

def check_data_consistency():
    train_file = 'preprocessed_data/ESD-Chinese/train.txt'
    preprocessed_path = 'preprocessed_data/ESD-Chinese'
    
    print("检查所有训练数据的维度一致性...")
    
    inconsistent_files = []
    
    with open(train_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in tqdm(lines):  # 检查所有样本
        parts = line.strip().split('|')
        basename, speaker = parts[0], parts[1]
        
        # 提取音素数量
        phones_str = parts[2].strip('{}')
        phones = phones_str.split()
        phone_count = len(phones)
        
        # 检查特征文件
        try:
            duration_file = os.path.join(preprocessed_path, 'duration', f'{speaker}-duration-{basename}.npy')
            pitch_file = os.path.join(preprocessed_path, 'pitch', f'{speaker}-pitch-{basename}.npy')
            energy_file = os.path.join(preprocessed_path, 'energy', f'{speaker}-energy-{basename}.npy')
            mel_file = os.path.join(preprocessed_path, 'mel', f'{speaker}-mel-{basename}.npy')
            
            duration = np.load(duration_file)
            pitch = np.load(pitch_file)
            energy = np.load(energy_file)
            mel = np.load(mel_file)
            
            duration_len = len(duration)
            pitch_len = len(pitch)
            energy_len = len(energy)
            mel_frames = mel.shape[0]
            duration_sum = duration.sum()
            
            # 检查一致性
            if not (phone_count == duration_len == pitch_len == energy_len):
                inconsistent_files.append({
                    'file': basename,
                    'phone_count': phone_count,
                    'duration_len': duration_len,
                    'pitch_len': pitch_len,
                    'energy_len': energy_len,
                    'mel_frames': mel_frames,
                    'duration_sum': duration_sum
                })
                if len(inconsistent_files) <= 10:  # 只打印前10个
                    print(f"不一致: {basename}")
                    print(f"  音素: {phone_count}, Duration: {duration_len}, Pitch: {pitch_len}, Energy: {energy_len}")
                    print(f"  Mel: {mel_frames}, Duration总和: {duration_sum}")
            
            if duration_sum != mel_frames:
                if len(inconsistent_files) <= 10:
                    print(f"Mel帧数不匹配: {basename}, Duration总和: {duration_sum}, Mel帧数: {mel_frames}")
            
        except Exception as e:
            print(f"文件读取错误 {basename}: {e}")
            inconsistent_files.append({
                'file': basename,
                'error': str(e)
            })
    
    print(f"\n发现 {len(inconsistent_files)} 个不一致的文件")
    
    if inconsistent_files:
        print("\n不一致的文件列表（前10个）:")
        for item in inconsistent_files[:10]:
            print(item)
    
    return inconsistent_files

if __name__ == "__main__":
    check_data_consistency() 