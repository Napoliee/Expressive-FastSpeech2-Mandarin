#!/usr/bin/env python3
"""
使用MFA对齐的正确推理脚本
训练时怎么处理，推理时就怎么处理
"""

import argparse
import os
import tempfile
import shutil
import subprocess
import torch
import yaml
import numpy as np
import textgrid
from torch.utils.data import DataLoader

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from text.ipa_processor import text_to_sequence_ipa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_temp_file_for_mfa(text, temp_dir):
    """为MFA创建临时文件"""
    
    # 创建临时音频文件 - 使用一个短的静音音频作为占位符
    wav_path = os.path.join(temp_dir, "temp.wav")
    # 创建一个短的静音音频（1秒，22050采样率）
    import soundfile as sf
    import numpy as np
    silent_audio = np.zeros(22050, dtype=np.float32)
    sf.write(wav_path, silent_audio, 22050)
    
    # 创建文本文件
    lab_path = os.path.join(temp_dir, "temp.lab")
    with open(lab_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    return wav_path, lab_path

def run_mfa_alignment(text, temp_dir):
    """运行MFA对齐获取音素"""
    
    print(f"正在对文本进行MFA对齐: {text}")
    
    # 创建临时文件
    wav_path, lab_path = create_temp_file_for_mfa(text, temp_dir)
    
    # MFA对齐命令
    output_dir = os.path.join(temp_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用与训练时相同的MFA模型和词典
    mfa_cmd = [
        "mfa", "align",
        temp_dir,
                    "mandarin_mfa",  # 使用预训练的中文词典
            "mandarin_mfa",  # 使用预训练的中文声学模型
        output_dir,
        "--clean"
    ]
    
    try:
        print("运行MFA对齐...")
        result = subprocess.run(mfa_cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"MFA对齐失败:")
            print(f"stderr: {result.stderr}")
            print(f"stdout: {result.stdout}")
            return None
        
        # 读取生成的TextGrid文件
        textgrid_path = os.path.join(output_dir, "temp.TextGrid")
        if os.path.exists(textgrid_path):
            return extract_phonemes_from_textgrid(textgrid_path)
        else:
            print("未生成TextGrid文件")
            return None
            
    except subprocess.TimeoutExpired:
        print("MFA对齐超时")
        return None
    except Exception as e:
        print(f"MFA对齐异常: {e}")
        return None

def extract_phonemes_from_textgrid(textgrid_path):
    """从TextGrid文件提取音素序列（与训练时相同的逻辑）"""
    try:
        tg = textgrid.TextGrid.fromFile(textgrid_path)
        phone_tier = None
        
        # 查找phones层
        for tier in tg.tiers:
            if tier.name.lower() in ['phones', 'phone']:
                phone_tier = tier
                break
        
        if phone_tier is None:
            print("TextGrid中未找到phones层")
            return None
        
        phonemes = []
        durations = []
        
        # 提取音素和时长（与training时完全相同的逻辑）
        for interval in phone_tier:
            phone = interval.mark.strip()
            duration_frames = int((interval.maxTime - interval.minTime) * 22050 / 256)  # hop_length=256
            
            # 只保留非空音素（与训练逻辑一致）
            if phone and phone != '':
                phonemes.append(phone)
                durations.append(max(1, duration_frames))
        
        print(f"提取到音素: {phonemes}")
        print(f"音素数量: {len(phonemes)}")
        
        return phonemes, durations
        
    except Exception as e:
        print(f"TextGrid解析失败: {e}")
        return None

def get_latest_checkpoint():
    """获取最新的checkpoint"""
    checkpoint_dir = "output/ckpt/ESD-Chinese"
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth.tar')]
    if not checkpoints:
        return None
    
    # 提取步数并排序
    checkpoint_steps = []
    for cp in checkpoints:
        try:
            step = int(cp.split('_')[0])
            checkpoint_steps.append(step)
        except:
            continue
    
    if checkpoint_steps:
        return max(checkpoint_steps)
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

    print(f"🎤 使用MFA对齐进行正确推理")
    print(f"检查点: {args.restore_step}")
    print(f"输入文本: {args.text}")
    print(f"说话人: {args.speaker_id}")
    print(f"情感: {args.emotion}")

    # 创建临时目录进行MFA对齐
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"临时目录: {temp_dir}")
        
        # 运行MFA对齐
        alignment_result = run_mfa_alignment(args.text, temp_dir)
        
        if alignment_result is None:
            print("❌ MFA对齐失败，使用备用方法")
            # 备用：使用一个已知的音素序列
            print("⚠️  使用默认音素序列")
            phonemes = ["n", "i˨˩˦", "x", "aw˨˩˦"]  # "你好"的音素
        else:
            phonemes, durations = alignment_result
    
    # 转换为IPA格式
    ipa_text = "{" + " ".join(phonemes) + "}"
    print(f"IPA音素: {ipa_text}")
    
    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # 准备输入数据
    ids = [args.speaker_id + "_mfa_test"]
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
        
        # 对texts进行padding处理（手动padding，因为我们只有一个样本）
        from utils.tools import pad_1D
        texts = pad_1D([text_sequence])
        
        batchs = [(ids, raw_texts, speakers, emotions, arousals, valences, texts, text_lens, max(text_lens))]

        control_values = args.pitch_control, args.energy_control, args.duration_control

        print(f"🎵 开始合成...")
        synthesize(model, configs, vocoder, batchs, control_values)
        print(f"✅ 合成完成！")
        print(f"📁 输出目录: {train_config['path']['result_path']}")
        print(f"🎧 音频文件: {args.speaker_id}_mfa_test.wav")
        
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 