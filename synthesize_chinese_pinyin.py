#!/usr/bin/env python3
"""
中文情感语音合成推理脚本
支持拼音音素输入和情感控制
"""

import argparse
import os
import json
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
import pypinyin
from pypinyin import Style

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from dataset_chinese import TextDataset
from text.symbols_pinyin import _symbol_to_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def chinese_to_pinyin_phonemes(text):
    """
    将中文文本转换为拼音音素序列
    """
    # 转换为拼音（不带声调）
    pinyin_list = pypinyin.lazy_pinyin(text, style=Style.NORMAL)
    
    print(f"拼音列表: {pinyin_list}")
    
    # 拼音到音素的映射规则
    def pinyin_to_phonemes(pinyin):
        """将单个拼音转换为音素序列"""
        # 声母映射
        initials = {
            'b': 'b', 'p': 'p', 'm': 'm', 'f': 'f',
            'd': 'd', 't': 't', 'n': 'n', 'l': 'l',
            'g': 'g', 'k': 'k', 'h': 'h',
            'j': 'j', 'q': 'q', 'x': 'x',
            'zh': 'zh', 'ch': 'ch', 'sh': 'sh', 'r': 'r',
            'z': 'z', 'c': 'c', 's': 's',
            'y': 'y', 'w': 'w'
        }
        
        # 韵母映射
        finals = {
            'a': 'a', 'o': 'o', 'e': 'e', 'i': 'i', 'u': 'u', 'v': 'y',
            'ai': 'ai', 'ei': 'ei', 'ui': 'ui', 'ao': 'ao', 'ou': 'ou',
            'iu': 'iu', 'ie': 'ie', 'ue': 'ue', 've': 'ue',
            'an': 'a n', 'en': 'e n', 'in': 'i n', 'un': 'u n', 'vn': 'y n',
            'ang': 'a ng', 'eng': 'e ng', 'ing': 'i ng', 'ong': 'o ng',
            'er': 'er', 'iao': 'iao', 'ian': 'ia n', 'iang': 'ia ng',
            'iong': 'io ng', 'uai': 'uai', 'uan': 'ua n', 'uang': 'ua ng'
        }
        
        # 分离声母和韵母
        initial = ''
        final = pinyin
        
        # 检查双字符声母
        for init in ['zh', 'ch', 'sh']:
            if pinyin.startswith(init):
                initial = init
                final = pinyin[len(init):]
                break
        
        # 检查单字符声母
        if not initial:
            for init in ['b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'j', 'q', 'x', 'r', 'z', 'c', 's', 'y', 'w']:
                if pinyin.startswith(init):
                    initial = init
                    final = pinyin[len(init):]
                    break
        
        # 构建音素序列
        phonemes = []
        
        # 添加声母
        if initial and initial in initials:
            phonemes.append(initials[initial])
        
        # 添加韵母
        if final:
            if final in finals:
                phonemes.extend(finals[final].split())
            else:
                # 如果韵母不在映射中，尝试逐字符处理
                for char in final:
                    if char in finals:
                        phonemes.extend(finals[char].split())
                    else:
                        phonemes.append(char)
        
        return phonemes
    
    # 转换所有拼音
    all_phonemes = []
    for py in pinyin_list:
        phonemes = pinyin_to_phonemes(py)
        all_phonemes.extend(phonemes)
    
    return all_phonemes

def preprocess_chinese_text(text, preprocess_config):
    """
    处理中文文本，转换为音素序列
    """
    if text.startswith('{') and text.endswith('}'):
        # 已经是音素格式
        phonemes = text[1:-1].split()
    else:
        # 中文文本，需要转换
        phonemes = chinese_to_pinyin_phonemes(text)
    
    # 转换为数字序列
    phoneme_ids = []
    for phone in phonemes:
        if phone in _symbol_to_id:
            phoneme_ids.append(_symbol_to_id[phone])
        else:
            print(f"Warning: Unknown phoneme '{phone}', using padding token")
            phoneme_ids.append(_symbol_to_id['_'])  # padding token
    
    print("Raw Text: {}".format(text))
    print("Phonemes: {}".format(phonemes))
    print("Phoneme IDs: {}".format(phoneme_ids))
    
    return np.array(phoneme_ids)

def synthesize(model, step, configs, vocoder, batchs, control_values, tag):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True, help="训练步数，例如 10000")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        required=True,
        help="批量合成或单句合成",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="批量模式：源文件路径（格式如train.txt）",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="单句模式：要合成的中文文本或音素序列",
    )
    parser.add_argument(
        "--speaker_id",
        type=str,
        default="0001",
        help="说话人ID (0001-0010)",
    )
    parser.add_argument(
        "--emotion",
        type=str,
        default="Neutral",
        choices=["Angry", "Happy", "Neutral", "Sad", "Surprise"],
        help="情感类型",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="自定义输出文件名（不包含扩展名）",
    )
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="预处理配置文件路径",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="模型配置文件路径"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="训练配置文件路径"
    )
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="音调控制 (0.5-2.0，越大音调越高)",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="能量控制 (0.5-2.0，越大音量越大)",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="语速控制 (0.5-2.0，越大语速越慢)",
    )
    args = parser.parse_args()

    # 检查参数
    if args.mode == "batch":
        assert args.source is not None and args.text is None
    if args.mode == "single":
        assert args.source is None and args.text is not None

    # 读取配置
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # 加载模型
    model = get_model(args, configs, device, train=False)

    # 加载声码器
    vocoder = get_vocoder(model_config, device)

    # 预处理文本
    if args.mode == "batch":
        # 批量模式
        dataset = TextDataset(args.source, preprocess_config, model_config)
        batchs = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=dataset.collate_fn,
        )
        tag = "batch"
    else:
        # 单句模式
        if args.output_name:
            ids = [args.output_name]
        else:
            ids = [f"synthesis_{args.speaker_id}_{args.emotion}"]
        raw_texts = [args.text]
        
        # 加载说话人映射
        with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json")) as f:
            speaker_map = json.load(f)
        speakers = np.array([speaker_map[args.speaker_id]])
        
        # 加载情感映射
        with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "emotions.json")) as f:
            json_raw = json.load(f)
            emotion_map = json_raw["emotion_dict"]
            arousal_map = json_raw["arousal_dict"] 
            valence_map = json_raw["valence_dict"]
        
        # 情感到arousal/valence的映射
        emotion_to_arousal_valence = {
            "Angry": ("0.9", "0.1"),
            "Happy": ("0.8", "0.8"), 
            "Neutral": ("0.5", "0.5"),
            "Sad": ("0.3", "0.2"),
            "Surprise": ("0.8", "0.6")
        }
        
        arousal_str, valence_str = emotion_to_arousal_valence[args.emotion]
        
        emotions = np.array([emotion_map[args.emotion]])
        arousals = np.array([arousal_map[arousal_str]])
        valences = np.array([valence_map[valence_str]])
        
        # 处理文本
        text_sequence = preprocess_chinese_text(args.text, preprocess_config)
        texts = np.array([text_sequence])
        text_lens = np.array([len(text_sequence)])
        
        batchs = [(ids, raw_texts, speakers, emotions, arousals, valences, texts, text_lens, max(text_lens))]
        tag = f"{args.speaker_id}_{args.emotion}"

    # 控制参数
    control_values = args.pitch_control, args.energy_control, args.duration_control

    # 合成
    synthesize(model, args.restore_step, configs, vocoder, batchs, control_values, tag)
    
    print(f"合成完成！输出文件保存在: {train_config['path']['result_path']}") 