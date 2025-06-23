 #!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import numpy as np
import tempfile
import subprocess
import shutil
from pypinyin import lazy_pinyin, Style
import textgrid

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import yaml
from model import FastSpeech2, ScheduledOptim
from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from text import text_to_sequence

class PinyinBasedInference:
    """基于拼音的中文TTS推理器"""
    
    def __init__(self, configs):
        self.preprocess_config, self.model_config, self.train_config = configs
        
        print("🚀 初始化基于拼音的中文TTS推理器")
        print("✅ 配置加载完成")
    
    def chinese_to_pinyin(self, text):
        """将中文转换为拼音（带声调数字）"""
        print(f"🔤 中文转拼音: {text}")
        
        # 使用pypinyin转换为带声调数字的拼音
        pinyin_list = lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True)
        
        # 过滤掉标点符号
        filtered_pinyin = []
        for py in pinyin_list:
            # 只保留字母和数字（拼音格式）
            if py.isalnum() or any(char.isalpha() for char in py):
                filtered_pinyin.append(py)
        
        pinyin_text = " ".join(filtered_pinyin)
        print(f"📝 拼音结果: {pinyin_text}")
        
        return pinyin_text
    
    def create_temp_files_for_mfa(self, pinyin_text, temp_dir):
        """为MFA创建临时文件"""
        
        # 创建临时音频文件（MFA需要，但我们只需要文本对齐）
        wav_path = os.path.join(temp_dir, "temp.wav")
        
        # 创建一个1秒的静音音频（MFA需要音频文件）
        import numpy as np
        from scipy.io import wavfile
        sample_rate = 22050
        duration = max(1.0, len(pinyin_text.split()) * 0.2)  # 根据拼音数量估算时长
        samples = np.zeros(int(sample_rate * duration), dtype=np.float32)
        wavfile.write(wav_path, sample_rate, samples)
        
        # 创建文本文件
        lab_path = os.path.join(temp_dir, "temp.lab")
        with open(lab_path, 'w', encoding='utf-8') as f:
            f.write(pinyin_text)
        
        print(f"✅ 创建临时文件:")
        print(f"   音频: {wav_path}")
        print(f"   文本: {lab_path}")
        
        return wav_path, lab_path
    
    def run_mfa_alignment(self, pinyin_text, temp_dir):
        """运行MFA对齐获取拼音音素"""
        
        print(f"🔧 准备MFA对齐...")
        
        # 创建临时文件
        wav_path, lab_path = self.create_temp_files_for_mfa(pinyin_text, temp_dir)
        
        # 创建输出目录
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # MFA命令（使用中文拼音模型）
        mfa_cmd = [
            "mfa", "align",
            temp_dir,
            "mandarin_pinyin",  # 使用中文拼音词典
            "mandarin_mfa",     # 使用中文声学模型
            output_dir,
            "--clean"
        ]
        
        try:
            print("🔄 运行MFA对齐...")
            
            # 在aligner环境中运行MFA
            conda_cmd = [
                "conda", "run", "-n", "aligner"
            ] + mfa_cmd
            
            print(f"   命令: {' '.join(conda_cmd)}")
            
            result = subprocess.run(conda_cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                print(f"❌ MFA对齐失败:")
                print(f"   stderr: {result.stderr}")
                return None
            
            # 检查生成的TextGrid文件
            textgrid_path = os.path.join(output_dir, "temp.TextGrid")
            if os.path.exists(textgrid_path):
                print("✅ MFA对齐成功，提取音素...")
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
        """从TextGrid提取音素"""
        
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
            
            # 提取音素
            sil_phones = ["sil", "sp", "spn", ""]
            phones = []
            
            for interval in phone_tier:
                phone = interval.mark.strip()
                if phone and phone not in sil_phones:
                    phones.append(phone)
            
            print(f"✅ 提取到音素: {phones}")
            print(f"   音素数量: {len(phones)}")
            
            return phones
            
        except Exception as e:
            print(f"❌ 提取音素失败: {e}")
            return None
    
    def phonemes_to_id_sequence(self, phonemes):
        """将音素转换为ID序列"""
        
        # 构造IPA格式字符串
        ipa_text = "{" + " ".join(phonemes) + "}"
        print(f"🔢 IPA音素字符串: {ipa_text}")
        
        try:
            # 使用IPA清理器
            sequence = text_to_sequence(ipa_text, ["basic_cleaners"])
            print(f"✅ ID序列: {sequence}")
            print(f"   序列长度: {len(sequence)}")
            return np.array(sequence, dtype=np.int64)
        except Exception as e:
            print(f"❌ 音素转ID失败: {e}")
            # 尝试备用方案：从训练数据中查找相似拼音
            print("🔄 尝试备用方案...")
            return self.fallback_pinyin_to_ids(phonemes)
    
    def fallback_pinyin_to_ids(self, phonemes):
        """备用方案：简单的拼音到ID映射"""
        
        # 这里可以实现一个简单的拼音音素映射
        # 或者从训练数据中查找相似的拼音序列
        print("⚠️  使用备用音素映射方案")
        
        # 简单地将每个音素映射为一个默认ID
        # 实际应用中需要更复杂的映射逻辑
        default_ids = [1] * len(phonemes)  # 使用ID=1作为默认
        return np.array(default_ids, dtype=np.int64)
    
    def synthesize_from_chinese(self, text, speaker_id, emotion_id):
        """从中文文本合成语音"""
        
        print(f"🎤 基于拼音+MFA的中文TTS推理")
        print(f"文本: {text}")
        print(f"说话人: {speaker_id}")
        print(f"情感: {emotion_id}")
        print("=" * 50)
        
        # 步骤1：中文转拼音
        pinyin_text = self.chinese_to_pinyin(text)
        if not pinyin_text:
            print("❌ 中文转拼音失败")
            return None
        
        # 步骤2：用拼音进行MFA对齐
        with tempfile.TemporaryDirectory() as temp_dir:
            phonemes = self.run_mfa_alignment(pinyin_text, temp_dir)
            
        if phonemes is None:
            print("❌ 无法获取音素序列")
            return None
        
        # 步骤3：转换为ID序列
        id_sequence = self.phonemes_to_id_sequence(phonemes)
        if id_sequence is None:
            print("❌ 无法转换为ID序列")
            return None
        
        # 步骤4：准备模型输入
        return self.prepare_model_input(text, speaker_id, emotion_id, id_sequence)
    
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
        ids = [f"{speaker_id}_pinyin"]
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
            # Forward
            output = model(*(batch[2:]), p_control=pitch_control, e_control=energy_control, d_control=duration_control)
            
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
            )

def get_latest_checkpoint():
    """获取最新的检查点"""
    ckpt_dir = "./output/ckpt/ESD-Chinese"
    if not os.path.exists(ckpt_dir):
        print(f"检查点目录不存在: {ckpt_dir}")
        return None
    
    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth.tar')]
    if not ckpt_files:
        print("未找到检查点文件")
        return None
    
    # 按步数排序，取最新的
    ckpt_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    latest_ckpt = os.path.join(ckpt_dir, ckpt_files[-1])
    print(f"使用检查点: {latest_ckpt}")
    return latest_ckpt

def main():
    print("🚀 启动基于拼音的中文TTS推理器")
    
    # 加载配置
    preprocess_config = yaml.load(open("./config/ESD-Chinese/preprocess.yaml", "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open("./config/ESD-Chinese/model.yaml", "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open("./config/ESD-Chinese/train.yaml", "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)
    
    # 设置设备
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    checkpoint_path = get_latest_checkpoint()
    if checkpoint_path is None:
        print("❌ 无法找到模型检查点")
        return
    
    # 创建args对象
    import argparse
    args = argparse.Namespace()
    args.restore_step = int(os.path.basename(checkpoint_path).split('.')[0])
    
    model = get_model(args, configs, device, train=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False)["model"])
    model.eval()
    print("✅ 模型加载完成")
    
    # 加载声码器
    vocoder = get_vocoder(model_config, device)
    print("✅ 声码器加载完成")
    
    # 创建推理器
    inference = PinyinBasedInference(configs)
    
    # 测试推理
    test_cases = [
        ("他对谁都那么友好。", "0008", "惊讶"),
        ("今天天气真不错。", "0009", "开心"),
        ("我很高兴见到你。", "0010", "开心"),
    ]
    
    print("\n" + "="*50)
    print("开始测试推理...")
    print("="*50)
    
    for text, speaker_id, emotion in test_cases:
        print(f"\n🎯 测试用例:")
        print(f"   文本: {text}")
        print(f"   说话人: {speaker_id}")
        print(f"   情感: {emotion}")
        
        # 合成
        batch = inference.synthesize_from_chinese(text, speaker_id, emotion)
        if batch is not None:
            print("🎵 开始合成...")
            control_values = (1.0, 1.0, 1.0)  # pitch, energy, duration
            try:
                synthesize(model, configs, vocoder, [batch], control_values)
                print("✅ 合成完成！")
            except Exception as e:
                print(f"❌ 合成失败: {e}")
        else:
            print("❌ 无法创建输入批次")
        
        print("-" * 30)

if __name__ == "__main__":
    main()