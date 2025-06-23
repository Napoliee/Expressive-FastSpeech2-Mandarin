#!/usr/bin/env python3
"""
从训练数据中查找相同文本的音素序列进行推理
确保与训练时完全一致
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

def find_phonemes_from_training_data(target_text):
    """从训练数据中查找相同文本的音素序列"""
    
    print(f"在训练数据中查找文本: {target_text}")
    
    # 搜索训练和验证数据
    data_files = [
        "preprocessed_data/ESD-Chinese/train_ipa.txt",
        "preprocessed_data/ESD-Chinese/val_ipa.txt"
    ]
    
    found_phonemes = []
    
    for data_file in data_files:
        if not os.path.exists(data_file):
            continue
            
        with open(data_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                parts = line.strip().split("|")
                if len(parts) >= 4:
                    basename = parts[0]
                    speaker = parts[1]
                    phonemes = parts[2]  # IPA音素字符串 "{...}"
                    raw_text = parts[3]
                    
                    if raw_text.strip() == target_text.strip():
                        print(f"找到匹配文本在 {data_file}:{line_num+1}")
                        print(f"  文件: {basename}")
                        print(f"  说话人: {speaker}")
                        print(f"  原文: {raw_text}")
                        print(f"  音素: {phonemes}")
                        
                        found_phonemes.append({
                            'basename': basename,
                            'speaker': speaker,
                            'phonemes': phonemes,
                            'raw_text': raw_text,
                            'source_file': data_file
                        })
    
    if found_phonemes:
        print(f"总共找到 {len(found_phonemes)} 个匹配项")
        return found_phonemes
    else:
        print("未找到匹配的文本")
        return None

def list_available_texts():
    """列出训练数据中可用的文本样本"""
    
    print("=== 训练数据中的可用文本样本 ===")
    
    data_files = [
        "preprocessed_data/ESD-Chinese/train_ipa.txt",
        "preprocessed_data/ESD-Chinese/val_ipa.txt"
    ]
    
    all_texts = set()
    
    for data_file in data_files:
        if not os.path.exists(data_file):
            continue
            
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) >= 4:
                    raw_text = parts[3].strip()
                    all_texts.add(raw_text)
    
    # 显示前20个文本样本
    texts_list = sorted(list(all_texts))
    print("前20个可用文本:")
    for i, text in enumerate(texts_list[:20]):
        print(f"  {i+1}: {text}")
    
    print(f"\n总共 {len(texts_list)} 个不同的文本")
    print("\n你可以使用这些文本进行推理，确保音素匹配训练数据")
    
    return texts_list

def get_latest_checkpoint():
    """获取最新的checkpoint"""
    checkpoint_dir = "output/ckpt/ESD-Chinese"
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint目录不存在: {checkpoint_dir}")
        return None
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth.tar')]
    if not checkpoints:
        print(f"在{checkpoint_dir}中未找到.pth.tar文件")
        return None
    
    print(f"找到checkpoint文件: {checkpoints}")
    
    # 提取步数并排序
    checkpoint_steps = []
    for cp in checkpoints:
        try:
            # 文件名格式应该是 "100000.pth.tar"
            step = int(cp.split('.')[0])
            checkpoint_steps.append(step)
        except ValueError:
            print(f"无法从文件名{cp}中提取步数")
            continue
    
    if checkpoint_steps:
        max_step = max(checkpoint_steps)
        print(f"最新checkpoint步数: {max_step}")
        return max_step
    return None

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
    parser.add_argument("--text", type=str, default=None, help="要合成的中文文本（必须在训练数据中存在）")
    parser.add_argument("--speaker_id", type=str, default=None, help="说话人ID（可选，如果不指定则使用原始说话人）")
    parser.add_argument("--emotion", type=str, default=None, help="情感（可选）: 开心/伤心/惊讶/愤怒/中立")
    parser.add_argument("--list_texts", action="store_true", help="列出所有可用的文本")
    parser.add_argument("--pitch_control", type=float, default=1.0)
    parser.add_argument("--energy_control", type=float, default=1.0)
    parser.add_argument("--duration_control", type=float, default=1.0)
    args = parser.parse_args()

    # 如果用户要求列出可用文本
    if args.list_texts:
        list_available_texts()
        return

    if args.text is None:
        print("请提供要合成的文本，或使用 --list_texts 查看可用文本")
        return

    # 获取最新checkpoint
    if args.restore_step is None:
        latest_step = get_latest_checkpoint()
        if latest_step is None:
            print("❌ 未找到任何checkpoint文件")
            return
        args.restore_step = latest_step
        print(f"🔄 使用最新checkpoint: {args.restore_step}")

    # 从训练数据中查找匹配的音素
    found_items = find_phonemes_from_training_data(args.text)
    
    if not found_items:
        print("❌ 未在训练数据中找到此文本")
        print("请使用 --list_texts 查看可用的文本")
        return
    
    # 选择第一个匹配项
    selected_item = found_items[0]
    phonemes_str = selected_item['phonemes']
    original_speaker = selected_item['speaker']
    
    # 使用指定的说话人ID，或者使用原始说话人
    target_speaker = args.speaker_id if args.speaker_id else original_speaker
    
    print(f"使用音素序列: {phonemes_str}")
    print(f"原始说话人: {original_speaker}")
    print(f"目标说话人: {target_speaker}")
    
    # Read Config
    preprocess_config = yaml.load(open("config/ESD-Chinese/preprocess.yaml", "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open("config/ESD-Chinese/model.yaml", "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open("config/ESD-Chinese/train.yaml", "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    print(f"🎤 使用训练数据音素进行推理")
    print(f"检查点: {args.restore_step}")
    print(f"输入文本: {args.text}")
    print(f"说话人: {target_speaker}")
    print(f"情感: {args.emotion}")

    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # 准备输入数据
    ids = [f"{target_speaker}_from_training"]
    raw_texts = [args.text]
    speakers = np.array([int(target_speaker)], dtype=np.int64)
    
    # 情感映射
    emotion_map = {"开心": 0, "伤心": 1, "惊讶": 2, "愤怒": 3, "中立": 4}
    if args.emotion:
        emotion_id = emotion_map.get(args.emotion, 4)
    else:
        emotion_id = 4  # 默认中立
        
    emotions = np.array([emotion_id], dtype=np.int64)
    arousals = np.array([0.5], dtype=np.float32)
    valences = np.array([0.5], dtype=np.float32)
    
    # 转换为序列
    try:
        text_sequence = np.array(text_to_sequence_ipa(phonemes_str), dtype=np.int64)
        text_lens = np.array([len(text_sequence)], dtype=np.int64)
        
        print(f"音素序列长度: {text_lens[0]}")
        print(f"音素ID序列: {text_sequence}")
        
        # 对texts进行padding处理（手动padding，因为我们只有一个样本）
        from utils.tools import pad_1D
        texts = pad_1D([text_sequence])
        
        batchs = [(ids, raw_texts, speakers, emotions, arousals, valences, texts, text_lens, max(text_lens))]

        control_values = args.pitch_control, args.energy_control, args.duration_control

        print(f"🎵 开始合成...")
        synthesize(model, configs, vocoder, batchs, control_values)
        print(f"✅ 合成完成！")
        print(f"📁 输出目录: {train_config['path']['result_path']}")
        print(f"🎧 音频文件: {target_speaker}_from_training.wav")
        
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 