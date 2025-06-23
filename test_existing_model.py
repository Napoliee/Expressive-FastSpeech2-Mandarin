#!/usr/bin/env python3
"""
测试现有100k模型的推理功能（使用原始拼音系统）
"""

import argparse
import os
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from dataset import TextDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_chinese_pinyin(text, speaker="0001"):
    """
    简单的中文到拼音映射（用于测试现有模型）
    """
    
    # 简单的中文词对应拼音映射
    chinese_to_pinyin = {
        "你好": "ni3 hao3",
        "世界": "shi4 jie4", 
        "美丽": "mei3 li4",
        "中国": "zhong1 guo2",
        "今天": "jin1 tian1",
        "明天": "ming2 tian1",
        "谢谢": "xie4 xie4",
        "开心": "kai1 xin1",
        "生日快乐": "sheng1 ri4 kuai4 le4",
    }
    
    # 如果找到映射就使用，否则使用默认
    if text in chinese_to_pinyin:
        pinyin_text = chinese_to_pinyin[text]
    else:
        # 默认拼音
        pinyin_text = "ni3 hao3"
    
    return pinyin_text

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
    parser.add_argument("--restore_step", type=int, default=100000)
    parser.add_argument("--text", type=str, default="你好", help="测试文本")
    parser.add_argument("--speaker_id", type=str, default="0001", help="speaker ID")
    parser.add_argument("--emotion", type=str, default="开心", help="emotion: 开心/伤心/惊讶/愤怒/中立")
    parser.add_argument("--pitch_control", type=float, default=1.0)
    parser.add_argument("--energy_control", type=float, default=1.0)
    parser.add_argument("--duration_control", type=float, default=1.0)
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(open("config/ESD-Chinese/preprocess.yaml", "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open("config/ESD-Chinese/model.yaml", "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open("config/ESD-Chinese/train.yaml", "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # 恢复原始模型配置
    model_config["max_seq_len"] = 2000
    if "vocab_size" in model_config:
        del model_config["vocab_size"]  # 移除IPA特定配置

    print(f"🎤 测试现有模型推理")
    print(f"检查点: {args.restore_step}")
    print(f"输入文本: {args.text}")
    print(f"说话人: {args.speaker_id}")
    print(f"情感: {args.emotion}")

    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # 准备输入数据
    ids = [args.speaker_id + "_test"]
    raw_texts = [args.text]
    speakers = np.array([int(args.speaker_id)])
    
    # 情感映射
    emotion_map = {"开心": 0, "伤心": 1, "惊讶": 2, "愤怒": 3, "中立": 4}
    emotions = np.array([emotion_map.get(args.emotion, 0)])
    arousals = np.array([0.5])
    valences = np.array([0.5])
    
    # 转换为拼音（使用原始文本处理系统）
    try:
        from text import text_to_sequence
        pinyin_text = preprocess_chinese_pinyin(args.text, args.speaker_id)
        print(f"拼音转换: {pinyin_text}")
        
        texts = [np.array(text_to_sequence(pinyin_text, ["basic_cleaners"]))]
        text_lens = np.array([len(texts[0])])
        
        batchs = [(ids, raw_texts, speakers, emotions, arousals, valences, texts, text_lens, max(text_lens))]

        control_values = args.pitch_control, args.energy_control, args.duration_control

        print(f"🎵 开始合成...")
        synthesize(model, configs, vocoder, batchs, control_values)
        print(f"✅ 合成完成！请检查输出目录: {train_config['path']['result_path']}")
        
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        print("这可能是因为符号表不匹配，建议重新训练IPA模型")

if __name__ == "__main__":
    main() 