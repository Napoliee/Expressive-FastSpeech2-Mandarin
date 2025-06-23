import os
import librosa
from tqdm import tqdm
import textgrid

def create_dummy_textgrid(wav_path, lab_path, output_path):
    """创建一个简单的TextGrid文件"""
    
    # 读取音频时长
    duration = librosa.get_duration(filename=wav_path)
    
    # 读取文本
    with open(lab_path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    
    # 创建TextGrid
    tg = textgrid.TextGrid()
    tg.maxTime = duration
    
    # 创建phones层，简单地把每个字符当作一个phone
    phone_tier = textgrid.IntervalTier('phones', 0, duration)
    
    if text:
        chars = list(text.replace(' ', ''))  # 移除空格并转为字符列表
        if chars:
            char_duration = duration / len(chars)
            
            for i, char in enumerate(chars):
                start_time = i * char_duration
                end_time = (i + 1) * char_duration
                phone_tier.add(start_time, end_time, char)
        else:
            # 如果没有字符，添加一个静音
            phone_tier.add(0, duration, 'sil')
    else:
        # 空文本，添加静音
        phone_tier.add(0, duration, 'sil')
    
    tg.append(phone_tier)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存TextGrid
    tg.write(output_path)

def main():
    raw_data_dir = './raw_data/ESD-Chinese'
    textgrid_dir = './preprocessed_data/ESD-Chinese/TextGrid'
    
    print("创建虚拟TextGrid文件...")
    
    # 遍历所有说话人目录
    for speaker_id in os.listdir(raw_data_dir):
        speaker_path = os.path.join(raw_data_dir, speaker_id)
        
        if not os.path.isdir(speaker_path):
            continue
            
        print(f"处理说话人: {speaker_id}")
        
        # 创建输出目录
        output_speaker_dir = os.path.join(textgrid_dir, speaker_id)
        os.makedirs(output_speaker_dir, exist_ok=True)
        
        # 处理每个wav文件
        wav_files = [f for f in os.listdir(speaker_path) if f.endswith('.wav')]
        
        for wav_file in tqdm(wav_files):
            basename = wav_file.replace('.wav', '')
            
            wav_path = os.path.join(speaker_path, wav_file)
            lab_path = os.path.join(speaker_path, f"{basename}.lab")
            textgrid_path = os.path.join(output_speaker_dir, f"{basename}.TextGrid")
            
            if os.path.exists(lab_path):
                try:
                    create_dummy_textgrid(wav_path, lab_path, textgrid_path)
                except Exception as e:
                    print(f"跳过 {wav_file}: {e}")
                    continue
    
    print("虚拟TextGrid文件创建完成!")

if __name__ == "__main__":
    main() 