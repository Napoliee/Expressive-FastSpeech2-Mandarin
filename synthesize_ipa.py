import argparse
import os
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from dataset_ipa import TextDataset
from text.ipa_processor import text_to_sequence_ipa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon

def preprocess_chinese_ipa(text, speaker="0001"):
    """
    使用IPA音素预处理中文文本
    注意：这里需要将中文文本转换为IPA音素
    简化版本：返回一些常见的IPA音素作为示例
    实际使用时需要接入中文转IPA的工具
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
    }
    
    # 简化处理：如果找到映射就使用，否则使用默认音素
    if text in chinese_to_ipa:
        ipa_phones = chinese_to_ipa[text]
    else:
        # 默认音素序列（静音）
        ipa_phones = ["spn"]
    
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument("--mode", type=str, choices=["batch", "single"], required=True, help="Synthesize a whole dataset or a single sentence")
    parser.add_argument("--source", type=str, default=None, help="path to a source file with format like train.txt and val.txt, for batch mode only")
    parser.add_argument("--text", type=str, default=None, help="raw text to synthesize, for single mode only")
    parser.add_argument("--speaker_id", type=str, default="0001", help="speaker ID for single mode")
    parser.add_argument("--emotion", type=str, default="开心", help="emotion: 开心/伤心/惊讶/愤怒/中立")
    parser.add_argument("--pitch_control", type=float, default=1.0, help="control the pitch of the whole utterance, larger value for higher pitch")
    parser.add_argument("--energy_control", type=float, default=1.0, help="control the energy of the whole utterance, larger value for larger volume")
    parser.add_argument("--duration_control", type=float, default=1.0, help="control the speed of the whole utterance, larger value for slower speaking rate")
    args = parser.parse_args()

    # Check source texts
    if args.mode == "batch":
        assert args.source is not None and args.text is None
    if args.mode == "single":
        assert args.source is None and args.text is not None

    # Read Config
    preprocess_config = yaml.load(open("config/ESD-Chinese/preprocess.yaml", "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open("config/ESD-Chinese/model.yaml", "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open("config/ESD-Chinese/train.yaml", "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Keep original max_seq_len to match checkpoint
    # model_config["max_seq_len"] = 10000

    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Preprocess texts
    if args.mode == "batch":
        # Batch mode
        dataset = TextDataset(args.source, preprocess_config, model_config)
        batchs = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=dataset.collate_fn,
        )
    if args.mode == "single":
        # Single mode
        ids = [args.speaker_id + "_single"]
        raw_texts = [args.text]
        speakers = np.array([int(args.speaker_id)])
        
        # 情感映射
        emotion_map = {"开心": 0, "伤心": 1, "惊讶": 2, "愤怒": 3, "中立": 4}
        emotions = np.array([emotion_map.get(args.emotion, 0)])
        arousals = np.array([0.5])  # 默认arousal
        valences = np.array([0.5])  # 默认valence
        
        # 转换为IPA音素
        ipa_text = preprocess_chinese_ipa(args.text, args.speaker_id)
        print(f"原文: {args.text}")
        print(f"IPA音素: {ipa_text}")
        
        # 转换为序列
        texts = [np.array(text_to_sequence_ipa(ipa_text))]
        text_lens = np.array([len(texts[0])])
        
        batchs = [(ids, raw_texts, speakers, emotions, arousals, valences, texts, text_lens, max(text_lens))]

    control_values = args.pitch_control, args.energy_control, args.duration_control

    synthesize(model, configs, vocoder, batchs, control_values)

if __name__ == "__main__":
    main() 