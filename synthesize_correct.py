#!/usr/bin/env python3
"""
正确的中文TTS推理脚本
完全遵循预处理流程：MFA对齐 → 提取IPA音素 → ID转换
"""

import argparse
import os
import tempfile
import subprocess
import torch
import yaml
import numpy as np
import textgrid
import soundfile as sf
import json

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from text.ipa_processor import text_to_sequence_ipa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CorrectChineseInference:
    def __init__(self, configs):
        self.preprocess_config, self.model_config, self.train_config = configs
        
    def create_temp_files_for_mfa(self, text, temp_dir):
        """为MFA创建临时文件"""
        
        # 创建临时音频文件（1秒静音）
        wav_path = os.path.join(temp_dir, "temp.wav")
        silent_audio = np.zeros(22050, dtype=np.float32)
        sf.write(wav_path, silent_audio, 22050)
        
        # 创建文本文件
        lab_path = os.path.join(temp_dir, "temp.lab")
        with open(lab_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        return wav_path, lab_path

    def run_mfa_alignment(self, text, temp_dir):
        """运行MFA对齐，与预处理时完全相同的流程"""
        
        print(f"🔤 正在对文本进行MFA对齐: {text}")
        
        # 创建临时文件
        wav_path, lab_path = self.create_temp_files_for_mfa(text, temp_dir)
        
        # 输出目录
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # 使用与训练时完全相同的MFA命令
        mfa_cmd = [
            "mfa", "align",
            temp_dir,
            "mandarin_mfa",  # 中文词典
            "mandarin_mfa",  # 中文声学模型
            output_dir,
            "--clean"
        ]
        
        try:
            print("   运行MFA对齐...")
            result = subprocess.run(mfa_cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                print(f"❌ MFA对齐失败:")
                print(f"   stderr: {result.stderr}")
                return None
            
            # 检查生成的TextGrid文件
            textgrid_path = os.path.join(output_dir, "temp.TextGrid")
            if os.path.exists(textgrid_path):
                return self.extract_phonemes_from_textgrid(textgrid_path)
            else:
                print("❌ 未生成TextGrid文件")
                return None
                
        except subprocess.TimeoutExpired:
            print("❌ MFA对齐超时")
            return None
        except Exception as e:
            print(f"❌ MFA对齐异常: {e}")
            return None

    def extract_phonemes_from_textgrid(self, textgrid_path):
        """从TextGrid提取IPA音素，与预处理器完全相同的逻辑"""
        
        try:
            tg = textgrid.TextGrid.fromFile(textgrid_path)
            
            # 查找phones层
            phone_tier = None
            for tier in tg.tiers:
                if tier.name.lower() in ['phones', 'phone']:
                    phone_tier = tier
                    break
            
            if phone_tier is None:
                print("❌ TextGrid中未找到phones层")
                return None
            
            # 提取音素，使用与preprocessor.py相同的逻辑
            sil_phones = ["sil", "sp", "spn"]
            phones = []
            
            for interval in phone_tier:
                phone = interval.mark.strip()
                if phone and phone != '':
                    phones.append(phone)
            
            # 过滤开头和结尾的静音（与preprocessor逻辑一致）
            # 找到第一个非静音音素
            start_idx = 0
            for i, p in enumerate(phones):
                if p not in sil_phones:
                    start_idx = i
                    break
            
            # 找到最后一个非静音音素
            end_idx = len(phones)
            for i in range(len(phones)-1, -1, -1):
                if phones[i] not in sil_phones:
                    end_idx = i + 1
                    break
            
            # 提取有效音素
            valid_phones = phones[start_idx:end_idx]
            
            print(f"✅ 提取到音素: {valid_phones}")
            print(f"   音素数量: {len(valid_phones)}")
            
            return valid_phones
            
        except Exception as e:
            print(f"❌ 提取音素失败: {e}")
            return None

    def phonemes_to_id_sequence(self, phonemes):
        """将IPA音素转换为ID序列，使用训练时的相同方法"""
        
        # 构造IPA格式字符串
        ipa_text = "{" + " ".join(phonemes) + "}"
        print(f"🔢 IPA音素字符串: {ipa_text}")
        
        # 使用训练时的转换方法
        try:
            sequence = text_to_sequence_ipa(ipa_text)
            print(f"✅ ID序列: {sequence}")
            print(f"   序列长度: {len(sequence)}")
            return np.array(sequence, dtype=np.int64)
        except Exception as e:
            print(f"❌ 音素转ID失败: {e}")
            return None

    def synthesize_with_correct_flow(self, text, speaker_id, emotion_id):
        """使用正确流程进行合成"""
        
        print(f"🎤 正确流程推理")
        print(f"文本: {text}")
        print(f"说话人: {speaker_id}")
        print(f"情感: {emotion_id}")
        print("=" * 50)
        
        # 步骤1：MFA对齐
        with tempfile.TemporaryDirectory() as temp_dir:
            phonemes = self.run_mfa_alignment(text, temp_dir)
            
            if phonemes is None:
                # 备用方案：从训练数据中查找相同文本
                print("🔄 使用备用方案：从训练数据查找...")
                phonemes = self.find_from_training_data(text)
                
                if phonemes is None:
                    print("❌ 无法获取音素序列")
                    return None
        
        # 步骤2：转换为ID序列
        id_sequence = self.phonemes_to_id_sequence(phonemes)
        if id_sequence is None:
            return None
        
        # 步骤3：准备模型输入
        return self.prepare_model_input(text, speaker_id, emotion_id, id_sequence)

    def find_from_training_data(self, target_text):
        """从训练数据中查找相同文本的音素（备用方案）"""
        
        data_files = [
            os.path.join(self.preprocess_config["path"]["preprocessed_path"], "train_ipa.txt"),
            os.path.join(self.preprocess_config["path"]["preprocessed_path"], "val_ipa.txt")
        ]
        
        for data_file in data_files:
            if not os.path.exists(data_file):
                continue
                
            with open(data_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("|")
                    if len(parts) >= 4:
                        phoneme_text = parts[2]  # IPA音素字段
                        raw_text = parts[3]     # 原文
                        
                        if raw_text.strip() == target_text.strip():
                            print(f"✅ 在训练数据中找到匹配文本")
                            # 提取音素
                            if phoneme_text.startswith('{') and phoneme_text.endswith('}'):
                                phonemes = phoneme_text[1:-1].split()
                                print(f"   音素: {phonemes}")
                                return phonemes
        
        print(f"⚠️  训练数据中无此文本: {target_text}")
        return None

    def prepare_model_input(self, text, speaker_id, emotion_id, id_sequence):
        """准备模型输入"""
        
        # 加载映射
        with open(os.path.join(self.preprocess_config["path"]["preprocessed_path"], "speakers.json")) as f:
            speaker_map = json.load(f)
        
        with open(os.path.join(self.preprocess_config["path"]["preprocessed_path"], "emotions.json")) as f:
            emotion_data = json.load(f)
            emotion_map = emotion_data["emotion_dict"]
            arousal_map = emotion_data["arousal_dict"]
            valence_map = emotion_data["valence_dict"]
        
        # 准备数据
        ids = [f"{speaker_id}_correct"]
        raw_texts = [text]
        speakers = np.array([speaker_map[speaker_id]], dtype=np.int64)
        emotions = np.array([emotion_map[emotion_id]], dtype=np.int64)
        arousals = np.array([arousal_map[emotion_id]], dtype=np.float32)
        valences = np.array([valence_map[emotion_id]], dtype=np.float32)
        
        text_lens = np.array([len(id_sequence)], dtype=np.int64)
        
        # Padding
        from utils.tools import pad_1D
        texts = pad_1D([id_sequence])
        
        batch = (ids, raw_texts, speakers, emotions, arousals, valences, texts, text_lens, max(text_lens))
        
        return batch

def synthesize(model, configs, vocoder, batchs, control_values):
    """合成函数"""
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
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
    """获取最新checkpoint"""
    checkpoint_dir = "output/ckpt/ESD-Chinese"
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth.tar')]
    if not checkpoints:
        return None
    
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('.')[0]))
    return int(latest_checkpoint.split('.')[0])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=None)
    parser.add_argument("--text", type=str, required=True, help="要合成的中文文本")
    parser.add_argument("--speaker_id", type=str, default="0001", help="说话人ID")
    parser.add_argument("--emotion_id", type=str, default="中立", help="情感ID")
    parser.add_argument("--pitch_control", type=float, default=1.0)
    parser.add_argument("--energy_control", type=float, default=1.0)
    parser.add_argument("--duration_control", type=float, default=1.0)
    args = parser.parse_args()

    # 获取checkpoint
    if args.restore_step is None:
        args.restore_step = get_latest_checkpoint()
        if args.restore_step is None:
            print("❌ 未找到checkpoint文件")
            return

    # 读取配置
    preprocess_config = yaml.load(open("config/ESD-Chinese/preprocess.yaml", "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open("config/ESD-Chinese/model.yaml", "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open("config/ESD-Chinese/train.yaml", "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # 加载模型
    model = get_model(args, configs, device, train=False)
    vocoder = get_vocoder(model_config, device)

    # 创建推理器
    inferencer = CorrectChineseInference(configs)
    
    # 执行正确流程的推理
    batch = inferencer.synthesize_with_correct_flow(args.text, args.speaker_id, args.emotion_id)
    
    if batch is not None:
        control_values = args.pitch_control, args.energy_control, args.duration_control
        
        print(f"\n🎵 开始合成...")
        synthesize(model, configs, vocoder, [batch], control_values)
        print(f"✅ 合成完成！")
        print(f"📁 输出目录: {train_config['path']['result_path']}")
        print(f"🎧 音频文件: {args.speaker_id}_correct.wav")
    else:
        print("❌ 推理失败")

if __name__ == "__main__":
    main() 