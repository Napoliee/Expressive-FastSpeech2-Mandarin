import os
import json
import librosa
import numpy as np
from scipy.io.wavfile import write
from tqdm import tqdm


emotion_dict = {
    "Neutral": "中立",
    "Happy": "开心", 
    "Sad": "伤心",
    "Angry": "愤怒",
    "Surprise": "惊讶"
}


def create_dataset(preprocess_config):
    """处理ESD中文数据集，生成filelist和speaker信息"""
    
    corpus_path = preprocess_config["path"]["corpus_path"]
    raw_path = preprocess_config["path"]["raw_path"]
    
    # 只处理中文说话人 (0001-0010)
    chinese_speakers = [f"{i:04d}" for i in range(1, 11)]
    
    os.makedirs(raw_path, exist_ok=True)
    os.makedirs(os.path.join(raw_path, "clips"), exist_ok=True)
    
    filelist = []
    speaker_info = {}
    
    for speaker in chinese_speakers:
        speaker_dir = os.path.join(corpus_path, speaker)
        if not os.path.exists(speaker_dir):
            continue
            
        # 读取文本文件
        text_file = os.path.join(speaker_dir, f"{speaker}.txt")
        if not os.path.exists(text_file):
            continue
            
        # 解析文本文件
        text_dict = {}
        with open(text_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    utt_id, text, emotion = parts[0], parts[1], parts[2]
                    text_dict[utt_id] = {
                        'text': text.strip(),
                        'emotion': emotion.strip()
                    }
        
        # 处理每种情感
        for emotion_eng in ['Neutral', 'Happy', 'Sad', 'Angry', 'Surprise']:
            emotion_dir = os.path.join(speaker_dir, emotion_eng)
            if not os.path.exists(emotion_dir):
                continue
                
            # 获取该情感下的所有音频文件
            wav_files = [f for f in os.listdir(emotion_dir) if f.endswith('.wav')]
            
            for wav_file in wav_files:
                utt_id = wav_file.replace('.wav', '')
                
                # 检查是否有对应的文本
                if utt_id not in text_dict:
                    continue
                
                text = text_dict[utt_id]['text']
                emotion_zh = emotion_dict.get(emotion_eng, emotion_eng)
                
                # 复制音频文件到clips目录
                src_path = os.path.join(emotion_dir, wav_file)
                dst_path = os.path.join(raw_path, "clips", wav_file)
                
                # 加载和保存音频 (确保格式一致)
                audio, sr = librosa.load(src_path, sr=22050)
                write(dst_path, 22050, (audio * 32767).astype(np.int16))
                
                # 添加到filelist
                filelist.append({
                    'audio_path': f"clips/{wav_file}",
                    'text': text,
                    'speaker': speaker,
                    'emotion': emotion_zh,
                    'emotion_eng': emotion_eng
                })
                
                # 更新speaker信息
                if speaker not in speaker_info:
                    speaker_info[speaker] = {
                        'emotions': set(),
                        'count': 0
                    }
                speaker_info[speaker]['emotions'].add(emotion_zh)
                speaker_info[speaker]['count'] += 1
    
    # 转换set为list用于JSON序列化
    for speaker in speaker_info:
        speaker_info[speaker]['emotions'] = list(speaker_info[speaker]['emotions'])
    
    # 保存filelist
    print(f"保存 {len(filelist)} 个音频文件到filelist...")
    with open(os.path.join(raw_path, "filelist.txt"), 'w', encoding='utf-8') as f:
        for item in filelist:
            f.write(f"{item['audio_path']}|{item['speaker']}|{item['text']}|{item['emotion']}\n")
    
    # 保存speaker信息
    with open(os.path.join(raw_path, "speaker_info.json"), 'w', encoding='utf-8') as f:
        json.dump(speaker_info, f, ensure_ascii=False, indent=2)
    
    print(f"数据集处理完成!")
    print(f"总计: {len(filelist)} 个音频文件")
    print(f"说话人: {len(speaker_info)} 个")
    for speaker, info in speaker_info.items():
        print(f"  {speaker}: {info['count']} 个文件, 情感: {info['emotions']}")


def extract_lexicon(preprocess_config):
    """提取词典用于MFA对齐"""
    
    raw_path = preprocess_config["path"]["raw_path"]
    lexicon_path = preprocess_config["path"]["lexicon_path"]
    
    # 读取filelist获取所有文本
    filelist_path = os.path.join(raw_path, "filelist.txt")
    if not os.path.exists(filelist_path):
        print("请先运行数据集处理生成filelist.txt")
        return
    
    all_chars = set()
    
    with open(filelist_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 3:
                text = parts[2]
                # 收集所有中文字符
                for char in text:
                    if '\u4e00' <= char <= '\u9fff':  # 中文Unicode范围
                        all_chars.add(char)
    
    # 生成简单的字符级词典 (每个字符映射到自己)
    # 这是一个简化的做法，实际可能需要更复杂的拼音映射
    os.makedirs(os.path.dirname(lexicon_path), exist_ok=True)
    
    with open(lexicon_path, 'w', encoding='utf-8') as f:
        for char in sorted(all_chars):
            f.write(f"{char}\t{char}\n")
    
    print(f"词典已保存到: {lexicon_path}")
    print(f"包含 {len(all_chars)} 个中文字符") 