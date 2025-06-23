import os
import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
from pypinyin import lazy_pinyin, Style

from text import _clean_text


def chinese_to_pinyin(text):
    """将中文转换为拼音（带声调数字）"""
    
    # 使用pypinyin转换为带声调数字的拼音
    pinyin_list = lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True)
    
    # 过滤掉标点符号，只保留拼音
    filtered_pinyin = []
    for py in pinyin_list:
        # 只保留字母和数字（拼音格式）
        if py.isalnum() or any(char.isalpha() for char in py):
            filtered_pinyin.append(py)
    
    pinyin_text = " ".join(filtered_pinyin)
    return pinyin_text


def prepare_align_pinyin(config):
    """为ESD数据集准备基于拼音的MFA对齐文件"""
    
    raw_path = config["path"]["raw_path"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]
    
    print("🎯 准备基于拼音的MFA对齐文件...")
    
    # 读取filelist
    filelist_path = os.path.join(raw_path, "filelist.txt")
    if not os.path.exists(filelist_path):
        print(f"找不到 {filelist_path}，请先运行数据准备")
        return
    
    # 创建拼音版本的数据目录
    pinyin_raw_path = raw_path.replace("ESD-Chinese", "ESD-Chinese-Pinyin")
    os.makedirs(pinyin_raw_path, exist_ok=True)
    
    # 统计信息
    total_files = 0
    converted_files = 0
    pinyin_filelist = []
    
    # 为每个说话人目录创建拼音版本的lab文件
    with open(filelist_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines(), desc="转换中文到拼音"):
            parts = line.strip().split('|')
            if len(parts) < 4:
                continue
                
            wav_path, text, speaker_id, emotion = parts[:4]
            basename = os.path.splitext(os.path.basename(wav_path))[0]
            total_files += 1
            
            # 清理文本
            cleaned_text = _clean_text(text, cleaners)
            
            # 转换为拼音
            pinyin_text = chinese_to_pinyin(cleaned_text)
            
            if not pinyin_text:
                print(f"⚠️  跳过空拼音: {cleaned_text}")
                continue
            
            # 创建拼音版本的说话人目录
            pinyin_speaker_dir = os.path.join(pinyin_raw_path, speaker_id)
            os.makedirs(pinyin_speaker_dir, exist_ok=True)
            
            # 复制音频文件到拼音目录
            original_wav_path = os.path.join(raw_path, speaker_id, f"{basename}.wav")
            pinyin_wav_path = os.path.join(pinyin_speaker_dir, f"{basename}.wav")
            
            if os.path.exists(original_wav_path):
                if not os.path.exists(pinyin_wav_path):
                    os.system(f"cp '{original_wav_path}' '{pinyin_wav_path}'")
                
                # 创建拼音lab文件
                pinyin_lab_path = os.path.join(pinyin_speaker_dir, f"{basename}.lab")
                with open(pinyin_lab_path, 'w', encoding='utf-8') as f_lab:
                    f_lab.write(pinyin_text)
                
                # 更新拼音版本的filelist条目
                pinyin_wav_rel_path = os.path.join(speaker_id, f"{basename}.wav")
                pinyin_filelist.append(f"{pinyin_wav_rel_path}|{pinyin_text}|{speaker_id}|{emotion}")
                
                converted_files += 1
                
                # 显示转换示例（前10个）
                if converted_files <= 10:
                    print(f"📝 示例 {converted_files}: {cleaned_text} → {pinyin_text}")
    
    # 保存拼音版本的filelist
    pinyin_filelist_path = os.path.join(pinyin_raw_path, "filelist.txt")
    with open(pinyin_filelist_path, 'w', encoding='utf-8') as f:
        for line in pinyin_filelist:
            f.write(line + '\n')
    
    print(f"\n✅ 拼音转换完成!")
    print(f"   总文件数: {total_files}")
    print(f"   成功转换: {converted_files}")
    print(f"   拼音数据目录: {pinyin_raw_path}")
    print(f"   拼音filelist: {pinyin_filelist_path}")
    
    print("\n🔄 接下来请运行MFA进行拼音对齐：")
    print("1. 确保已安装MFA: conda install -c conda-forge montreal-forced-alignment")
    print("2. 下载拼音词典: mfa download dictionary mandarin_pinyin")
    print("3. 下载中文声学模型: mfa download acoustic mandarin_mfa")
    print("4. 运行拼音对齐:")
    print(f"   conda run -n aligner mfa align {pinyin_raw_path} mandarin_pinyin mandarin_mfa ./preprocessed_data/ESD-Chinese-Pinyin/TextGrid")
    
    return pinyin_raw_path


def prepare_align_from_pinyin_data(pinyin_raw_path):
    """直接从已有的拼音数据准备MFA对齐"""
    
    print(f"🔧 从拼音数据准备MFA对齐: {pinyin_raw_path}")
    
    if not os.path.exists(pinyin_raw_path):
        print(f"❌ 拼音数据目录不存在: {pinyin_raw_path}")
        return None
    
    # 检查拼音数据结构
    filelist_path = os.path.join(pinyin_raw_path, "filelist.txt")
    if not os.path.exists(filelist_path):
        print(f"❌ 找不到拼音filelist: {filelist_path}")
        return None
    
    # 统计拼音数据
    with open(filelist_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"📊 拼音数据统计:")
    print(f"   文件总数: {len(lines)}")
    
    # 检查几个样例
    print(f"📝 拼音样例:")
    for i, line in enumerate(lines[:5]):
        parts = line.strip().split('|')
        if len(parts) >= 2:
            print(f"   {i+1}. {parts[1]}")
    
    print(f"\n🎯 准备运行MFA拼音对齐命令:")
    textgrid_output = "./preprocessed_data/ESD-Chinese-Pinyin/TextGrid"
    print(f"conda run -n aligner mfa align {pinyin_raw_path} mandarin_pinyin mandarin_mfa {textgrid_output}")
    
    return textgrid_output


if __name__ == "__main__":
    # 测试拼音转换
    test_texts = [
        "他对谁都那么友好。",
        "今天天气真不错。", 
        "我很高兴见到你。"
    ]
    
    print("🧪 测试拼音转换:")
    for text in test_texts:
        pinyin = chinese_to_pinyin(text)
        print(f"   {text} → {pinyin}") 