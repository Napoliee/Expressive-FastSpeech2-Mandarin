import os
import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
from collections import defaultdict
import json
import random
from pypinyin import lazy_pinyin, Style

from text import _clean_text


def get_sorted_items(items):
    """按键排序"""
    return sorted(items, key=lambda x: x[0])


def chinese_to_pinyin(text):
    """将中文转换为拼音（无声调）"""
    try:
        # 移除标点符号，只保留中文字符
        chinese_chars = ''.join([char for char in text if '\u4e00' <= char <= '\u9fff'])
        
        if not chinese_chars:
            return ""
        
        # 转换为拼音（无声调）
        pinyin_list = lazy_pinyin(
            chinese_chars,
            style=Style.NORMAL  # 使用无声调格式
        )
        
        # 用空格连接拼音
        pinyin_text = " ".join(pinyin_list)
        return pinyin_text
        
    except Exception as e:
        print(f"❌ 拼音转换失败: {text} - {str(e)}")
        return ""


def prepare_align(config):
    """为ESD中文数据集准备MFA对齐所需的文件，匹配preprocessor_en.py格式"""
    
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]
    val_ratio = config["preprocessing"].get("val_ratio", 0.15)
    test_ratio = config["preprocessing"].get("test_ratio", 0.05)
    
    print("🎯 开始准备ESD中文数据集（拼音格式，匹配preprocessor_en.py格式）...")
    
    # 创建输出目录
    os.makedirs(out_dir, exist_ok=True)
    
    # 情绪映射（中文到英文）
    emotion_mapping_cn = {
        "中立": "Neutral",
        "开心": "Happy", 
        "伤心": "Sad",
        "愤怒": "Angry",
        "惊讶": "Surprise"
    }
    
    # 情绪到数值的映射（用于arousal和valence）
    emotion_values = {
        "Neutral": {"arousal": "0.5", "valence": "0.5"},
        "Happy": {"arousal": "0.8", "valence": "0.8"},
        "Sad": {"arousal": "0.3", "valence": "0.2"},
        "Angry": {"arousal": "0.9", "valence": "0.1"},
        "Surprise": {"arousal": "0.8", "valence": "0.6"}
    }
    
    # 存储所有文件信息
    all_files_info = []
    speaker_info = {}
    
    # 统计转换情况
    total_files = 0
    converted_files = 0
    
    # 只处理中文说话人（0001-0010）
    for speaker_id in range(1, 11):  # 0001-0010为中文
        speaker_folder = f"{speaker_id:04d}"
        source_speaker_dir = os.path.join(in_dir, speaker_folder)
        
        if not os.path.exists(source_speaker_dir):
            continue
            
        print(f"📂 处理中文说话人: {speaker_folder}")
        
        # 读取文本文件
        text_file = os.path.join(source_speaker_dir, f"{speaker_folder}.txt")
        if not os.path.exists(text_file):
            print(f"❌ 找不到文本文件: {text_file}")
            continue
            
        # 解析文本文件
        text_dict = {}
        with open(text_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        file_id, text, emotion_cn = parts[0], parts[1], parts[2]
                        emotion_en = emotion_mapping_cn.get(emotion_cn, "Neutral")
                        text_dict[file_id] = {
                            'text': text,
                            'emotion_cn': emotion_cn,
                            'emotion_en': emotion_en
                        }
        
        # 记录说话人信息（假设性别，可以根据实际情况调整）
        speaker_info[speaker_folder] = {
            'gender': 'M' if speaker_id <= 5 else 'F'  # 简单假设前5个是男性
        }
        
        # 创建说话人输出目录
        speaker_out_dir = os.path.join(out_dir, speaker_folder)
        os.makedirs(speaker_out_dir, exist_ok=True)
        
        # 处理每个情绪文件夹
        for emotion_folder in ["Angry", "Happy", "Neutral", "Sad", "Surprise"]:
            source_emotion_dir = os.path.join(source_speaker_dir, emotion_folder)
            
            if not os.path.exists(source_emotion_dir):
                continue
                
            # 处理音频文件
            wav_files = [f for f in os.listdir(source_emotion_dir) if f.endswith('.wav')]
            
            for wav_file in tqdm(wav_files, desc=f"  {emotion_folder}"):
                file_id = os.path.splitext(wav_file)[0]
                total_files += 1
                
                if file_id in text_dict:
                    # 处理音频文件
                    source_wav = os.path.join(source_emotion_dir, wav_file)
                    target_wav = os.path.join(speaker_out_dir, f"{file_id}.wav")
                    
                    # 重采样音频
                    wav, _ = librosa.load(source_wav, sr=sampling_rate)
                    wav = wav / max(abs(wav)) * max_wav_value
                    wavfile.write(target_wav, sampling_rate, wav.astype(np.int16))
                    
                    # 清理文本并转换为拼音
                    text = text_dict[file_id]['text']
                    cleaned_text = _clean_text(text, cleaners)
                    
                    # 转换为拼音
                    pinyin_text = chinese_to_pinyin(cleaned_text)
                    
                    if not pinyin_text:
                        print(f"⚠️  跳过空拼音: {cleaned_text}")
                        continue
                    
                    # 创建拼音lab文件
                    target_lab = os.path.join(speaker_out_dir, f"{file_id}.lab")
                    with open(target_lab, 'w', encoding='utf-8') as f:
                        f.write(pinyin_text)
                    
                    # 记录文件信息（匹配preprocessor_en.py格式）
                    emotion_en = emotion_folder
                    emotion_vals = emotion_values[emotion_en]
                    
                    # 创建标准basename（与音频文件名保持一致）
                    # preprocessor能够从basename.split("_")[0]提取speaker
                    basename = file_id
                    
                    all_files_info.append({
                        'basename': basename,
                        'text': pinyin_text,  # 拼音文本
                        'original_text': cleaned_text,  # 保留原始中文
                        'speaker_id': speaker_folder,
                        'emotion': emotion_en,
                        'arousal': emotion_vals["arousal"],
                        'valence': emotion_vals["valence"]
                    })
                    
                    converted_files += 1
                    
                    # 显示前几个转换示例
                    if converted_files <= 10:
                        print(f"📝 示例 {converted_files}: {cleaned_text} → {pinyin_text}")
    
    # 创建preprocessor_en.py格式的文件列表
    create_en_style_filelist(out_dir, all_files_info, val_ratio, test_ratio)
    
    # 保存说话人信息（preprocessor_en.py格式）
    save_en_style_speaker_info(out_dir, speaker_info)
    
    print(f"\n✅ ESD中文数据预处理完成（拼音格式，preprocessor_en.py格式）!")
    print(f"📊 总文件数: {total_files}")
    print(f"🔤 成功转换: {converted_files}")
    print(f"👥 说话人数: {len(speaker_info)}")
    print(f"🎯 输出格式: 拼音 (用于拼音MFA模型)")
    print(f"📝 文件格式: preprocessor_en.py格式")


def create_en_style_filelist(out_dir, files_info, val_ratio=0.15, test_ratio=0.05):
    """创建preprocessor_en.py格式的filelist文件"""
    print("📝 创建preprocessor_en.py格式的文件列表...")
    
    # 设置随机种子
    random.seed(42)
    
    # 按说话人和情绪分组进行分层采样
    speaker_emotion_groups = defaultdict(lambda: defaultdict(list))
    
    for file_info in files_info:
        speaker_id = file_info['speaker_id']
        emotion = file_info['emotion']
        speaker_emotion_groups[speaker_id][emotion].append(file_info)
    
    train_files = []
    val_files = []
    test_files = []
    
    # 为每个说话人的每种情绪进行分层采样
    for speaker_id in speaker_emotion_groups:
        for emotion in speaker_emotion_groups[speaker_id]:
            emotion_files = speaker_emotion_groups[speaker_id][emotion]
            random.shuffle(emotion_files)
            
            n_files = len(emotion_files)
            n_test = max(1, int(n_files * test_ratio))
            n_val = max(1, int(n_files * val_ratio))
            n_train = n_files - n_test - n_val
            
            # 分配文件
            test_files.extend(emotion_files[:n_test])
            val_files.extend(emotion_files[n_test:n_test + n_val])
            train_files.extend(emotion_files[n_test + n_val:])
    
    # 创建preprocessor_en.py格式的filelist：basename|text|speaker_id|其他字段|emotion|arousal|valence
    def create_en_style_lines(files):
        lines = []
        for file_info in files:
            line = "|".join([
                file_info['basename'],     # basename（preprocessor从这里提取）
                file_info['text'],         # 拼音文本
                file_info['speaker_id'],   # 说话人ID
                "esd_chinese",             # 数据集标识
                "default",                 # 占位符
                file_info['emotion'],      # 情感（倒数第3个）
                file_info['arousal'],      # arousal值（倒数第2个）
                file_info['valence']       # valence值（最后一个）
            ])
            lines.append(line)
        return lines
    
    train_lines = create_en_style_lines(train_files)
    val_lines = create_en_style_lines(val_files)
    test_lines = create_en_style_lines(test_files)
    
    # 保存文件列表
    with open(os.path.join(out_dir, "train.txt"), 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_lines))
    
    with open(os.path.join(out_dir, "val.txt"), 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_lines))
    
    with open(os.path.join(out_dir, "test.txt"), 'w', encoding='utf-8') as f:
        f.write('\n'.join(test_lines))
    
    # 保存完整文件列表（preprocessor_en.py格式）
    all_lines = train_lines + val_lines + test_lines
    with open(os.path.join(out_dir, "filelist.txt"), 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_lines))
    
    # 另外保存一个包含原始中文的映射文件（用于后续参考）
    mapping_lines = []
    for file_info in files_info:
        mapping_line = f"{file_info['basename']}|{file_info['original_text']}|{file_info['text']}"
        mapping_lines.append(mapping_line)
    
    with open(os.path.join(out_dir, "chinese_pinyin_mapping.txt"), 'w', encoding='utf-8') as f:
        f.write('\n'.join(mapping_lines))
    
    print(f"   📊 数据划分统计:")
    print(f"     训练集: {len(train_lines)} 条目 ({len(train_lines)/len(all_lines)*100:.1f}%)")
    print(f"     验证集: {len(val_lines)} 条目 ({len(val_lines)/len(all_lines)*100:.1f}%)")
    print(f"     测试集: {len(test_lines)} 条目 ({len(test_lines)/len(all_lines)*100:.1f}%)")
    print(f"   📝 已保存中文-拼音映射文件: chinese_pinyin_mapping.txt")
    print(f"   ✅ filelist格式: basename|text|speaker_id|dataset|default|emotion|arousal|valence")


def save_en_style_speaker_info(out_dir, speaker_info):
    """保存preprocessor_en.py格式的说话人信息文件"""
    print("👥 保存说话人信息（preprocessor_en.py格式）...")
    
    # preprocessor_en.py格式：说话人ID|其他信息
    with open(os.path.join(out_dir, "speaker_info.txt"), 'w', encoding='utf-8') as f:
        for speaker_id, info in get_sorted_items(speaker_info.items()):
            gender = info['gender']
            f.write(f'{speaker_id}|{gender}\n')
    
    print(f"   记录了 {len(speaker_info)} 个说话人信息")
    print(f"   ✅ speaker_info格式: 说话人ID|性别") 