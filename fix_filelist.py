import os

# 读取原始文件
input_file = 'raw_data/ESD-Chinese/filelist.txt'
output_file = 'raw_data/ESD-Chinese/filelist_new.txt'

# 情感映射到数值（用于arousal和valence）
emotion_map = {
    '中立': ('0', '0'),  # neutral: low arousal, neutral valence
    '开心': ('1', '1'),  # happy: high arousal, positive valence
    '伤心': ('0', '-1'), # sad: low arousal, negative valence
    '愤怒': ('1', '-1'), # angry: high arousal, negative valence
    '惊讶': ('1', '0'),  # surprised: high arousal, neutral valence
}

with open(input_file, 'r', encoding='utf-8') as f_in:
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            line = line.strip()
            if line:
                parts = line.split('|')
                if len(parts) == 4:
                    wav_file, speaker, text, emotion = parts
                    
                    # 添加缺失的字段
                    # 格式：文件名|文本|说话人|intent|strategy|emotion|arousal|valence
                    arousal, valence = emotion_map.get(emotion, ('0', '0'))
                    
                    new_line = f"{wav_file}|{text}|{speaker}|default|default|{emotion}|{arousal}|{valence}"
                    f_out.write(new_line + '\n')
                else:
                    f_out.write(line + '\n')

# 替换原文件
os.rename(output_file, input_file)
print(f"已更新 {input_file} 文件格式")
print("新格式：文件名|文本|说话人|intent|strategy|emotion|arousal|valence") 