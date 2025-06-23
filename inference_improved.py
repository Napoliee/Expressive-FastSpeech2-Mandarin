#!/usr/bin/env python3
"""
改进的推理脚本 - 使用更准确的IPA音素映射
"""

import argparse
import os
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from text.ipa_processor import text_to_sequence_ipa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_better_chinese_to_ipa():
    """
    基于训练数据中观察到的IPA音素创建更好的映射
    """
    
    # 从实际训练数据中提取的常见字符到IPA映射
    # 这些是在MFA对齐中实际出现的音素
    char_to_ipa = {
        # 常用字符 - 基于实际MFA结果
        '我': ['w', 'o˨˩˦'],
        '一': ['i˥˩'],
        '直': ['ʈʂ', 'ʐ̩˧˥'],
        '到': ['t', 'aw˥˩'],
        '清': ['tɕʰ', 'iŋ˥˩'],
        '晨': ['tʂʰ', 'ən˧˥'],
        '四': ['s', 'z̩˥˩'],
        '点': ['tj', 'an˨˩˦'],
        '才': ['tsʰ', 'aj˧˥'],
        '家': ['tɕ', 'j', 'a˥˩'],
        '，': ['spn'],
        
        '就': ['tɕ', 'j', 'ow˥˩'],
        '是': ['ʂ', 'ʐ̩˥˩'],
        '这': ['ʈʂ', 'ə˥˩'],
        '个': ['k', 'ə˥˩'],
        '意': ['i˥˩'],
        '思': ['s', 'z̩˥˩'],
        '你': ['n', 'i˨˩˦'],
        '又': ['j', 'ow˥˩'],
        '聪': ['ts', 'oŋ˥˩'],
        '明': ['m', 'iŋ˧˥'],
        '好': ['x', 'aw˨˩˦'],
        '看': ['kʰ', 'an˥˩'],
        '。': ['spn'],
        
        '所': ['s', 'u̯o˨˩˦'],
        '以': ['i˨˩˦'],
        '永': ['j', 'oŋ˨˩˦'],
        '不': ['p', 'u˥˩'],
        '喝': ['x', 'ə˥˩'],
        '它': ['tʰ', 'a˥˩'],
        '的': ['t', 'i˥˩'],
        
        # 更多常用字符
        '了': ['l', 'i̯aw˨˩˦'],
        '有': ['j', 'ow˨˩˦'],
        '在': ['ts', 'aj˥˩'],
        '会': ['x', 'uej˥˩'],
        '说': ['ʂ', 'u̯o˥˩'],
        '要': ['j', 'aw˥˩'],
        '都': ['t', 'ow˥˩'],
        '很': ['x', 'ən˨˩˦'],
        '也': ['j', 'e˨˩˦'],
        '可': ['kʰ', 'ə˨˩˦'],
        '什': ['ʂ', 'ən˧˥'],
        '么': ['m', 'ə˥˩'],
        '没': ['m', 'ej˧˥'],
        '时': ['ʂ', 'ʐ̩˧˥'],
        '候': ['x', 'ow˥˩'],
        '还': ['x', 'aj˧˥'],
        '能': ['n', 'əŋ˧˥'],
        '去': ['tɕʰ', 'y˥˩'],
        '来': ['l', 'aj˧˥'],
        '用': ['j', 'oŋ˥˩'],
        '那': ['n', 'a˥˩'],
        '些': ['ɕ', 'j', 'e˥˩'],
        '为': ['w', 'ej˧˥'],
        
        # 数字
        '零': ['l', 'iŋ˧˥'],
        '二': ['ʌ˧˥', 'ʐ̩˥˩'],
        '三': ['s', 'an˥˩'],
        '五': ['w', 'u˨˩˦'],
        '六': ['l', 'j', 'ow˥˩'],
        '七': ['tɕʰ', 'i˥˩'],
        '八': ['p', 'a˥˩'],
        '九': ['tɕ', 'j', 'ow˨˩˦'],
        '十': ['ʂ', 'ʐ̩˧˥'],
    }
    
    return char_to_ipa

def improved_chinese_to_ipa(text):
    """
    改进的中文到IPA转换
    """
    char_to_ipa = create_better_chinese_to_ipa()
    
    # 清理文本
    text = text.strip()
    
    ipa_phones = []
    
    for char in text:
        if char in char_to_ipa:
            # 使用已知映射
            phones = char_to_ipa[char]
            ipa_phones.extend(phones)
        elif '\u4e00' <= char <= '\u9fff':  # 中文字符
            # 未知中文字符，使用统计上最常见的音素模式
            # 根据中文音韵学，大多数字符是声母+韵母结构
            import random
            fallback_options = [
                ['ʂ', 'ʐ̩˥˩'],   # 类似"是"
                ['t', 'i˥˩'],    # 类似"的"
                ['x', 'aw˨˩˦'],  # 类似"好"
                ['l', 'i˥˩'],    # 类似"李"
                ['m', 'ej˧˥'],   # 类似"没"
                ['n', 'i˨˩˦'],   # 类似"你"
                ['k', 'ə˥˩'],    # 类似"个"
                ['tɕ', 'i˥˩'],   # 类似"及"
                ['w', 'o˨˩˦'],   # 类似"我"
                ['j', 'ow˥˩'],   # 类似"有"
            ]
            chosen_phones = random.choice(fallback_options)
            ipa_phones.extend(chosen_phones)
            print(f"⚠️  未知字符 '{char}' -> {chosen_phones}")
        elif char in '，。？！；：、':
            # 标点符号
            ipa_phones.append('spn')
        elif char.strip():  # 非空白字符
            # 其他字符当作静音
            ipa_phones.append('spn')
    
    # 如果结果为空，添加默认音素
    if not ipa_phones:
        ipa_phones = ['n', 'i˨˩˦', 'x', 'aw˨˩˦']  # "你好"
    
    # 添加句间停顿（对于长句子）
    if len(ipa_phones) > 8:
        enhanced_phones = []
        for i, phone in enumerate(ipa_phones):
            enhanced_phones.append(phone)
            # 适当位置添加停顿
            if i > 0 and (i + 1) % 6 == 0 and i < len(ipa_phones) - 1:
                enhanced_phones.append('spn')
        ipa_phones = enhanced_phones
    
    return ipa_phones

def get_latest_checkpoint():
    """获取最新的checkpoint"""
    checkpoint_dir = "output/ckpt/ESD-Chinese"
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth.tar')]
    if not checkpoints:
        return None
    
    # 找到最新的checkpoint
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('.')[0]))
    return int(latest_checkpoint.split('.')[0])

def synthesize(model, configs, vocoder, batchs, control_values):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model(
                *(batch[2:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
            )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=None, help="使用特定步数的checkpoint，默认使用最新的")
    parser.add_argument("--text", type=str, default="你好世界", help="要合成的中文文本")
    parser.add_argument("--speaker_id", type=str, default="0001", help="说话人ID")
    parser.add_argument("--emotion", type=str, default="中立", help="情感: 开心/伤心/惊讶/愤怒/中立")
    parser.add_argument("--pitch_control", type=float, default=1.0)
    parser.add_argument("--energy_control", type=float, default=1.0)
    parser.add_argument("--duration_control", type=float, default=1.0)
    args = parser.parse_args()

    # 获取最新checkpoint
    if args.restore_step is None:
        latest_step = get_latest_checkpoint()
        if latest_step is None:
            print("❌ 未找到任何checkpoint文件")
            return
        args.restore_step = latest_step
        print(f"🔄 使用最新checkpoint: {args.restore_step}")
    
    # Read Config
    preprocess_config = yaml.load(open("config/ESD-Chinese/preprocess.yaml", "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open("config/ESD-Chinese/model.yaml", "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open("config/ESD-Chinese/train.yaml", "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    print(f"🎤 改进版推理脚本")
    print(f"检查点: {args.restore_step}")
    print(f"输入文本: {args.text}")
    print(f"说话人: {args.speaker_id}")
    print(f"情感: {args.emotion}")

    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # 转换文本为IPA音素
    ipa_phones = improved_chinese_to_ipa(args.text)
    ipa_text = "{" + " ".join(ipa_phones) + "}"
    
    print(f"IPA音素序列: {ipa_text}")
    print(f"音素数量: {len(ipa_phones)}")

    # 准备输入数据
    ids = [args.speaker_id + "_improved"]
    raw_texts = [args.text]
    speakers = np.array([int(args.speaker_id)], dtype=np.int64)
    
    # 情感映射
    emotion_map = {"开心": 0, "伤心": 1, "惊讶": 2, "愤怒": 3, "中立": 4}
    emotions = np.array([emotion_map.get(args.emotion, 4)], dtype=np.int64)
    arousals = np.array([0.5], dtype=np.float32)
    valences = np.array([0.5], dtype=np.float32)
    
    # 转换为序列
    try:
        text_sequence = np.array(text_to_sequence_ipa(ipa_text), dtype=np.int64)
        text_lens = np.array([len(text_sequence)], dtype=np.int64)
        
        print(f"音素序列长度: {text_lens[0]}")
        print(f"音素ID序列: {text_sequence}")
        
        # 对texts进行padding处理
        from utils.tools import pad_1D
        texts = pad_1D([text_sequence])
        
        batchs = [(ids, raw_texts, speakers, emotions, arousals, valences, texts, text_lens, max(text_lens))]

        control_values = args.pitch_control, args.energy_control, args.duration_control

        print(f"🎵 开始合成...")
        synthesize(model, configs, vocoder, batchs, control_values)
        print(f"✅ 合成完成！")
        print(f"📁 输出目录: {train_config['path']['result_path']}")
        print(f"🎧 音频文件: {args.speaker_id}_improved.wav")
        
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 