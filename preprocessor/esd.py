import os
import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from text import _clean_text


def prepare_align(config):
    """为ESD数据集准备MFA对齐所需的文件"""
    
    raw_path = config["path"]["raw_path"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]
    
    print("准备MFA对齐文件...")
    
    # 读取filelist
    filelist_path = os.path.join(raw_path, "filelist.txt")
    if not os.path.exists(filelist_path):
        print(f"找不到 {filelist_path}，请先运行数据准备")
        return
    
    # 为每个说话人目录创建清理后的lab文件
    with open(filelist_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            parts = line.strip().split('|')
            if len(parts) < 4:
                continue
                
            wav_path, text, speaker_id, emotion = parts[:4]
            basename = os.path.splitext(os.path.basename(wav_path))[0]
            
            # 清理文本
            cleaned_text = _clean_text(text, cleaners)
            
            # 更新lab文件
            speaker_dir = os.path.join(raw_path, speaker_id)
            lab_path = os.path.join(speaker_dir, f"{basename}.lab")
            
            if os.path.exists(lab_path):
                with open(lab_path, 'w', encoding='utf-8') as f_lab:
                    f_lab.write(cleaned_text)
    
    print("MFA对齐文件准备完成!")
    print("接下来请运行MFA进行强制对齐：")
    print("1. 安装MFA: conda install -c conda-forge montreal-forced-alignment")
    print("2. 下载中文模型: mfa download acoustic mandarin_mfa")
    print("3. 下载中文词典: mfa download dictionary mandarin_mfa")
    print("4. 运行对齐:")
    print(f"   mfa align {raw_path} mandarin_mfa mandarin_mfa ./preprocessed_data/ESD-Chinese/TextGrid")