#!/usr/bin/env python3
"""
基于当前IPA训练模型的推理脚本
"""

import argparse
import os
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from dataset_ipa_fixed import TextDataset
from text.ipa_processor import text_to_sequence_ipa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_chinese_ipa(text, speaker="0001"):
    """
    使用IPA音素预处理中文文本
    """
    
    # 示例：简单的中文词对应IPA音素映射
    chinese_to_ipa = {
        "你好": ["n", "i˨˩˦", "x", "aw˨˩˦"],
        "世界": ["ʂ", "ʐ̩˥˩", "tɕ", "j", "e˥˩"],
        "美丽": ["m", "ej˨˩˦", "l", "i˥˩"],
        "中国": ["ʈʂ", "oŋ˥˩", "k", "u̯o˥˩"],
        "今天": ["tɕ", "in˥˩", "tʰ", "j", "an˥˩"],
        "明天": ["m", "iŋ˥˩", "tʰ", "j", "an˥˩"],
        "谢谢": ["ɕ", "j", "e˥˩", "ɕ", "j", "e˥˩"],
        "开心": ["kʰ", "aj˥˩", "ɕ", "in˥˩"],
        "生日快乐": ["ʂ", "əŋ˥˩", "ʐ̩˥˩", "kʰ", "uaj˥˩", "l", "ə˥˩"],
        "测试": ["tʰs", "ə˥˩", "ʂ", "ʐ̩˥˩"],
        "语音": ["y˨˩˦", "in˥˩"],
        "合成": ["x", "ə˧˥", "ʈʂʰ", "əŋ˧˥"],
    }
    
    # 如果找到映射就使用，否则使用默认音素
    if text in chinese_to_ipa:
        ipa_phones = chinese_to_ipa[text]
    else:
        # 默认音素序列
        ipa_phones = ["n", "i˨˩˦", "x", "aw˨˩˦"]  # 默认"你好"
    
    # 转换为IPA格式字符串
    ipa_text = "{" + " ".join(ipa_phones) + "}"
    
    return ipa_text

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

def get_latest_checkpoint():
    """获取最新的checkpoint步数"""
    ckpt_dir = "./output/ckpt/ESD-Chinese"
    if not os.path.exists(ckpt_dir):
        return None
    
    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth.tar')]
    if not ckpt_files:
        return None
    
    # 提取步数并找到最大值
    steps = []
    for f in ckpt_files:
        try:
            step = int(f.replace('.pth.tar', ''))
            steps.append(step)
        except:
            continue
    
    return max(steps) if steps else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=None, help="使用特定步数的checkpoint，默认使用最新的")
    parser.add_argument("--text", type=str, default="你好", help="测试文本")
    parser.add_argument("--speaker_id", type=str, default="0001", help="speaker ID")
    parser.add_argument("--emotion", type=str, default="开心", help="emotion: 开心/伤心/惊讶/愤怒/中立")
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

    print(f"🎤 使用IPA模型推理")
    print(f"检查点: {args.restore_step}")
    print(f"输入文本: {args.text}")
    print(f"说话人: {args.speaker_id}")
    print(f"情感: {args.emotion}")

    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # 准备输入数据
    ids = [args.speaker_id + "_ipa_test"]
    raw_texts = [args.text]
    speakers = np.array([int(args.speaker_id)], dtype=np.int64)
    
    # 情感映射
    emotion_map = {"开心": 0, "伤心": 1, "惊讶": 2, "愤怒": 3, "中立": 4}
    emotions = np.array([emotion_map.get(args.emotion, 0)], dtype=np.int64)
    arousals = np.array([0.5], dtype=np.float32)
    valences = np.array([0.5], dtype=np.float32)
    
    # 转换为IPA音素
    try:
        ipa_text = preprocess_chinese_ipa(args.text, args.speaker_id)
        print(f"IPA音素: {ipa_text}")
        
        # 转换为序列
        text_sequence = np.array(text_to_sequence_ipa(ipa_text), dtype=np.int64)
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
        print(f"🎧 音频文件: {args.speaker_id}_ipa_test.wav")
        
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 