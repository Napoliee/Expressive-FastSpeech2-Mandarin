from g2pc import G2pC
import re
import os
import glob
import shutil
from tqdm import tqdm
from m_text_normalizer import TextNormalizer

def has_non_english_chars(text):
    return bool(re.search(r'[^A-Za-z]', text))

def preprocess(text, g2p):
    pinyin = ''
    pinyin_result = g2p(text)
    for pys in pinyin_result:
        for py in pys[3].split(' '):
            if py[-1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                # 去掉所有特殊符号
                if has_non_english_chars(py[:-1][-1]):
                    continue
                else:
                    pinyin += py[:-1] + ' '
            else:
                # 去掉所有特殊符号
                if has_non_english_chars(py):
                    continue
                else:
                    pinyin += py + ' '
    return pinyin

def process_data():
    # 初始化G2P转换器
    g2p = G2pC()
    normalizer = TextNormalizer()

    # 输入和输出目录
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 获取项目根目录
    input_dir = os.path.join(base_dir, "Expressive-FastSpeech2-continuous/raw_data/ESD-Chinese")
    output_dir = os.path.join(base_dir, "Expressive-FastSpeech2-continuous/raw_data/ESD-Chinese-Singing-MFA")
    os.makedirs(output_dir, exist_ok=True)

    # 读取文本文件
    with open(os.path.join(input_dir, "filelist.txt"), "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 处理每一行
    for line in tqdm(lines):
        parts = line.strip().split("|")
        if len(parts) < 2:  # 至少需要wav路径和文本
            print(f"警告: 无效的行格式 {line}")
            continue

        # 解析文件信息
        wav_info = parts[0]  # 格式: clips/0001_000001.wav
        text = parts[1]
        
        # 从wav路径中提取说话人ID
        basename = os.path.basename(wav_info)  # 0001_000001.wav
        speaker_id = basename.split("_")[0]  # 0001

        # 构建源文件和目标文件路径
        src_wav = os.path.join(input_dir, speaker_id, basename)  # .../raw_data/ESD-Chinese/0001/0001_000001.wav
        dst_wav = os.path.join(output_dir, basename)  # .../raw_data/ESD-Chinese-Singing-MFA/0001_000001.wav
        lab_file = os.path.join(output_dir, basename.replace(".wav", ".lab"))

        # 复制音频文件
        if os.path.exists(src_wav):
            shutil.copy2(src_wav, dst_wav)
        else:
            print(f"警告: 找不到音频文件 {src_wav}")
            continue

        # 文本预处理
        normalized_text = normalizer.normalize(text)
        pinyin = preprocess(normalized_text, g2p)

        # 保存拼音文本
        with open(lab_file, "w", encoding="utf-8") as f:
            f.write(pinyin.strip())

    print("数据预处理完成！")
    print(f"输出目录: {output_dir}")

if __name__ == "__main__":
    process_data() 