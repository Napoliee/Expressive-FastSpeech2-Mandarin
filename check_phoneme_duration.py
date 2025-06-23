import os
import numpy as np
import librosa
from tqdm import tqdm

def check_phoneme_duration_alignment():
    """检查音素序列长度与音频时长的对应关系"""
    
    print("=== 检查音素-时长对应关系 ===")
    
    # 检查几个训练样本
    with open("preprocessed_data/ESD-Chinese/train.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()[:10]  # 检查前10个样本
    
    mismatches = []
    
    for i, line in enumerate(lines):
        parts = line.strip().split("|")
        if len(parts) < 4:
            continue
            
        basename = parts[0]
        speaker = parts[1]
        phoneme_ids = parts[2].split()
        text = parts[3]
        
        # 1. 检查音素长度
        phoneme_count = len(phoneme_ids)
        
        # 2. 检查duration特征
        duration_path = f"preprocessed_data/ESD-Chinese/duration/{speaker}-duration-{basename}.npy"
        if os.path.exists(duration_path):
            duration = np.load(duration_path)
            duration_len = len(duration)
            duration_sum = duration.sum()
        else:
            print(f"Duration文件不存在: {duration_path}")
            continue
        
        # 3. 检查mel特征长度
        mel_path = f"preprocessed_data/ESD-Chinese/mel/{speaker}-mel-{basename}.npy"
        if os.path.exists(mel_path):
            mel = np.load(mel_path)
            mel_frames = mel.shape[0]
        else:
            print(f"Mel文件不存在: {mel_path}")
            continue
        
        # 4. 检查原始音频时长
        audio_path = f"raw_data/ESD-Chinese/{speaker}/{basename}.wav"
        if os.path.exists(audio_path):
            audio, sr = librosa.load(audio_path, sr=22050)
            audio_duration = len(audio) / sr
            expected_mel_frames = int(audio_duration * sr / 256)  # hop_length=256
        else:
            print(f"原始音频不存在: {audio_path}")
            continue
        
        # 5. 比较各种长度
        print(f"\n样本{i+1}: {text}")
        print(f"  音素数量: {phoneme_count}")
        print(f"  Duration长度: {duration_len}")
        print(f"  Duration总和: {duration_sum}")
        print(f"  Mel帧数: {mel_frames}")
        print(f"  音频时长: {audio_duration:.2f}秒")
        print(f"  期望Mel帧数: {expected_mel_frames}")
        
        # 检查不一致的情况
        if phoneme_count != duration_len:
            print(f"  ❌ 音素数量({phoneme_count}) != Duration长度({duration_len})")
            mismatches.append(("phoneme_duration", basename))
        
        if abs(duration_sum - mel_frames) > 5:  # 允许小误差
            print(f"  ❌ Duration总和({duration_sum}) != Mel帧数({mel_frames})")
            mismatches.append(("duration_mel", basename))
        
        if abs(mel_frames - expected_mel_frames) > 10:
            print(f"  ❌ Mel帧数({mel_frames}) != 期望帧数({expected_mel_frames})")
            mismatches.append(("mel_audio", basename))
        
        # 计算每个音素的平均时长
        avg_phoneme_duration = duration_sum / phoneme_count if phoneme_count > 0 else 0
        avg_phoneme_time = audio_duration / phoneme_count if phoneme_count > 0 else 0
        print(f"  平均每音素帧数: {avg_phoneme_duration:.1f}")
        print(f"  平均每音素时长: {avg_phoneme_time:.3f}秒")
        
    print(f"\n=== 检查结果 ===")
    print(f"发现 {len(mismatches)} 个不匹配问题")
    
    # 统计问题类型
    problem_types = {}
    for problem_type, _ in mismatches:
        problem_types[problem_type] = problem_types.get(problem_type, 0) + 1
    
    for problem_type, count in problem_types.items():
        print(f"{problem_type}: {count} 个样本")

def check_original_alignment():
    """检查原始对齐数据"""
    print("\n=== 检查原始对齐数据 ===")
    
    # 检查原始对齐文件
    sample_basename = "0008_001723"
    sample_speaker = "0008"
    
    textgrid_path = f"preprocessed_data/ESD-Chinese/TextGrid/{sample_speaker}/{sample_basename}.TextGrid"
    
    if os.path.exists(textgrid_path):
        print(f"检查TextGrid文件: {textgrid_path}")
        
        # 读取TextGrid文件（简单解析）
        with open(textgrid_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # 简单统计intervals数量
        interval_count = content.count('intervals [')
        print(f"TextGrid中的interval数量: {interval_count}")
        
        # 检查对应的duration文件
        duration_path = f"preprocessed_data/ESD-Chinese/duration/{sample_speaker}-duration-{sample_basename}.npy"
        if os.path.exists(duration_path):
            duration = np.load(duration_path)
            print(f"Duration文件长度: {len(duration)}")
            print(f"Duration总和: {duration.sum()}")
        
    else:
        print(f"TextGrid文件不存在: {textgrid_path}")

def suggest_fixes():
    """建议修复方案"""
    print("\n=== 修复建议 ===")
    
    print("如果发现音素-时长不匹配问题，可能的原因和解决方案:")
    print("1. 音素编码转换问题:")
    print("   - 检查fix_phoneme_encoding.py是否正确保持了音素数量")
    print("   - 验证转换前后的音素序列长度是否一致")
    
    print("\n2. 预处理阶段的问题:")
    print("   - 重新运行预处理，确保Duration特征提取正确")
    print("   - 检查MFA对齐结果是否正确")
    
    print("\n3. 数据格式问题:")
    print("   - 确认训练数据格式与原始预处理结果一致")
    print("   - 检查是否有数据截断或补零问题")

if __name__ == "__main__":
    check_phoneme_duration_alignment()
    check_original_alignment()
    suggest_fixes() 