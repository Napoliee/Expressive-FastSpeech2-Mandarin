#!/usr/bin/env python3
"""
完全遵循原始synthesize.py风格的推理脚本
使用正确的文本处理流程
"""

import re
import argparse
import os
import json
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from dataset import TextDataset
from text import text_to_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_chinese(text, preprocess_config):
    """
    中文文本预处理，遵循原始代码风格
    """
    print("Raw Text Sequence: {}".format(text))
    
    # 使用chinese_cleaners进行文本清理
    cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
    
    # 将文本放入花括号中，模拟音素格式
    # 注意：这里我们需要一个将中文转换为音素的方法
    # 但原始代码没有提供，所以我们需要创建一个兼容的方法
    
    # 对于中文，我们需要查找现有的lexicon或使用MFA结果
    # 让我们先尝试从训练数据中查找相同的文本
    phoneme_sequence = find_phoneme_sequence_from_training_data(text, preprocess_config)
    
    if phoneme_sequence is not None:
        print("Phoneme Sequence: {}".format(phoneme_sequence))
        # 使用找到的音素序列
        sequence = np.array(phoneme_sequence)
    else:
        # 如果没找到，使用默认的text_to_sequence处理
        # 这里会调用chinese_cleaners
        phones = "{" + " ".join(list(text.replace(" ", ""))) + "}"
        print("Phoneme Sequence: {}".format(phones))
        sequence = np.array(
            text_to_sequence(phones, cleaners)
        )
    
    return sequence

def find_phoneme_sequence_from_training_data(target_text, preprocess_config):
    """
    从训练数据中查找相同文本的音素序列（ID格式）
    """
    data_files = [
        os.path.join(preprocess_config["path"]["preprocessed_path"], "train.txt"),
        os.path.join(preprocess_config["path"]["preprocessed_path"], "val.txt")
    ]
    
    for data_file in data_files:
        if not os.path.exists(data_file):
            continue
            
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) >= 4:
                    basename = parts[0]
                    speaker = parts[1]
                    phoneme_ids = parts[2]  # 空格分隔的ID序列
                    raw_text = parts[3]
                    
                    if raw_text.strip() == target_text.strip():
                        print(f"✅ 在训练数据中找到匹配文本: {basename}")
                        print(f"   说话人: {speaker}")
                        print(f"   音素ID序列: {phoneme_ids}")
                        
                        # 返回ID序列
                        return [int(x) for x in phoneme_ids.split()]
    
    print(f"⚠️  在训练数据中未找到文本: {target_text}")
    return None

def synthesize(model, step, configs, vocoder, batchs, control_values, tag):
    """
    完全遵循原始synthesize函数
    """
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
                tag,
            )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        default="single",
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--speaker_id",
        type=str,
        default="0001",
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "--emotion_id",
        type=str,
        default="开心",
        help="emotion ID for multi-emotion synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "--arousal",
        type=str,
        default=None,
        help="arousal value for multi-emotion synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "--valence",
        type=str,
        default=None,
        help="valence value for multi-emotion synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        default="config/ESD-Chinese/preprocess.yaml",
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", 
        "--model_config", 
        type=str, 
        default="config/ESD-Chinese/model.yaml",
        help="path to model.yaml"
    )
    parser.add_argument(
        "-t", 
        "--train_config", 
        type=str, 
        default="config/ESD-Chinese/train.yaml",
        help="path to train.yaml"
    )
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    args = parser.parse_args()

    # Check source texts
    if args.mode == "batch":
        assert args.source is not None and args.text is None
    if args.mode == "single":
        assert args.source is None and args.text is not None

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    print(f"🎤 原始风格推理脚本")
    print(f"检查点: {args.restore_step}")
    print(f"模式: {args.mode}")
    print(f"文本: {args.text}")
    print(f"说话人: {args.speaker_id}")
    print(f"情感: {args.emotion_id}")

    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Preprocess texts
    if args.mode == "batch":
        # Get dataset
        dataset = TextDataset(args.source, preprocess_config, model_config)
        batchs = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=dataset.collate_fn,
        )
        tag = None
    if args.mode == "single":
        emotions = arousals = valences = None
        ids = raw_texts = [args.text[:100]]
        
        # 加载speaker映射
        with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json")) as f:
            speaker_map = json.load(f)
        speakers = np.array([speaker_map[args.speaker_id]])
        
        # 加载情感映射
        if model_config["multi_emotion"]:
            with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "emotions.json")) as f:
                json_raw = json.load(f)
                emotion_map = json_raw["emotion_dict"]
                arousal_map = json_raw["arousal_dict"]
                valence_map = json_raw["valence_dict"]
            emotions = np.array([emotion_map[args.emotion_id]])
            # arousal和valence使用与emotion相同的键
            arousal_key = args.arousal if args.arousal else args.emotion_id
            valence_key = args.valence if args.valence else args.emotion_id
            arousals = np.array([arousal_map[arousal_key]])
            valences = np.array([valence_map[valence_key]])
        
        # 处理中文文本 - 关键部分！
        print(f"\n=== 文本预处理 ===")
        texts = np.array([preprocess_chinese(args.text, preprocess_config)])
        text_lens = np.array([len(texts[0])])
        
        print(f"音素序列长度: {text_lens[0]}")
        print(f"音素ID序列: {texts[0]}")
        
        batchs = [(ids, raw_texts, speakers, emotions, arousals, valences, texts, text_lens, max(text_lens))]
        tag = f"{args.speaker_id}_{args.emotion_id}"

    control_values = args.pitch_control, args.energy_control, args.duration_control

    print(f"\n🎵 开始合成...")
    synthesize(model, args.restore_step, configs, vocoder, batchs, control_values, tag)
    print(f"✅ 合成完成！")
    print(f"📁 输出目录: {train_config['path']['result_path']}")
    print(f"🎧 音频文件: {tag}.wav")

if __name__ == "__main__":
    main() 