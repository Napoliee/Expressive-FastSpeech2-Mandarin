import os
import librosa
import soundfile as sf
from tqdm import tqdm

def resample_audio(input_path, output_path, target_sr=22050):
    """将音频文件重新采样到目标采样率"""
    # 加载音频
    y, sr = librosa.load(input_path, sr=None)
    
    # 重新采样
    if sr != target_sr:
        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    else:
        y_resampled = y
    
    # 保存重新采样的音频
    sf.write(output_path, y_resampled, target_sr)

def main():
    raw_data_dir = './raw_data/ESD-Chinese'
    target_sr = 22050
    
    print(f"将音频文件重新采样回 {target_sr}Hz...")
    
    # 遍历所有说话人目录
    for speaker_id in os.listdir(raw_data_dir):
        speaker_path = os.path.join(raw_data_dir, speaker_id)
        
        if not os.path.isdir(speaker_path):
            continue
            
        print(f"处理说话人: {speaker_id}")
        
        # 处理每个wav文件
        wav_files = [f for f in os.listdir(speaker_path) if f.endswith('.wav')]
        
        for wav_file in tqdm(wav_files):
            wav_path = os.path.join(speaker_path, wav_file)
            
            try:
                # 检查当前采样率
                y, sr = librosa.load(wav_path, sr=None)
                
                if sr != target_sr:
                    # 重新采样并覆盖原文件
                    resample_audio(wav_path, wav_path, target_sr)
                    
            except Exception as e:
                print(f"跳过 {wav_file}: {e}")
                continue
    
    print("音频重新采样完成!")

if __name__ == "__main__":
    main() 