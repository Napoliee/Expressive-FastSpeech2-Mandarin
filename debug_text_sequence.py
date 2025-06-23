import yaml
import numpy as np
from text import text_to_sequence

def debug_text_processing():
    # 加载配置
    preprocess_config = yaml.load(open("config/ESD-Chinese/preprocess.yaml", "r"), Loader=yaml.FullLoader)
    cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
    
    # 从训练数据中读取第一行作为测试
    with open("preprocessed_data/ESD-Chinese/train.txt", "r", encoding="utf-8") as f:
        line = f.readline().strip()
    
    line_split = line.split("|")
    basename = line_split[0]
    speaker = line_split[1]
    text = line_split[2]
    
    print(f"Basename: {basename}")
    print(f"Speaker: {speaker}")
    print(f"Original text: {text}")
    print(f"Text length (chars): {len(text)}")
    
    # 处理文本
    phone_sequence = text_to_sequence(text, cleaners)
    print(f"Processed sequence: {phone_sequence}")
    print(f"Sequence length: {len(phone_sequence)}")
    
    # 检查对应的特征文件
    duration_path = f"preprocessed_data/ESD-Chinese/duration/{speaker}-duration-{basename}.npy"
    pitch_path = f"preprocessed_data/ESD-Chinese/pitch/{speaker}-pitch-{basename}.npy"
    energy_path = f"preprocessed_data/ESD-Chinese/energy/{speaker}-energy-{basename}.npy"
    
    duration = np.load(duration_path)
    pitch = np.load(pitch_path)  
    energy = np.load(energy_path)
    
    print(f"Duration length: {len(duration)}")
    print(f"Pitch length: {len(pitch)}")
    print(f"Energy length: {len(energy)}")
    
    print(f"\n匹配情况:")
    print(f"text_to_sequence结果长度: {len(phone_sequence)}")
    print(f"预处理特征长度: {len(duration)}")
    print(f"是否匹配: {len(phone_sequence) == len(duration)}")
    
    # 如果不匹配，显示原始文本的音素拆分
    if text.startswith('{') and text.endswith('}'):
        phonemes = text[1:-1].split()
        print(f"\n原始音素序列: {phonemes}")
        print(f"原始音素数量: {len(phonemes)}")

if __name__ == "__main__":
    debug_text_processing() 