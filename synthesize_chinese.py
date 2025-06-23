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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_chinese_phonemes(phoneme_text, preprocess_config):
    """
    处理中文音素序列
    输入: 音素字符串，例如 "t w ej˥˩ ʂ ej˧˥ spn n a˥˩"
    输出: 数字编码序列
    """
    # 加载音素映射表
    phoneme_mapping_path = os.path.join(
        preprocess_config["path"]["preprocessed_path"], 
        "phoneme_mapping.json"
    )
    with open(phoneme_mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    
    phoneme_to_id = mapping["phoneme_to_id"]
    
    # 分割音素
    if phoneme_text.startswith('{') and phoneme_text.endswith('}'):
        phonemes = phoneme_text[1:-1].split()
    else:
        phonemes = phoneme_text.split()
    
    # 转换为数字序列
    phoneme_ids = [phoneme_to_id.get(p, 1) for p in phonemes]  # 1是_UNK_的ID
    
    print("Raw Phoneme Sequence: {}".format(phoneme_text))
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
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        required=True,
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        "--phonemes",
        type=str,
        default=None,
        help="phoneme sequence to synthesize, for single-sentence mode only (e.g., 't w ej˥˩ ʂ ej˧˥')",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="display text for the output file name",
    )
    parser.add_argument(
        "--speaker_id",
        type=str,
        default="0001",
        help="speaker ID for multi-speaker synthesis (e.g., '0001', '0005')",
    )
    parser.add_argument(
        "--emotion_id",
        type=str,
        default="中立",
        help="emotion ID for synthesis (开心/伤心/惊讶/愤怒/中立)",
    )
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
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
        assert args.source is not None and args.phonemes is None
    if args.mode == "single":
        assert args.source is None and args.phonemes is not None

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

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
        # 准备单句推理
        ids = [args.text[:50] if args.text else "sample"]
        raw_texts = [args.text if args.text else "Generated speech"]
        
        # 加载说话人映射
        with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json")) as f:
            speaker_map = json.load(f)
        speakers = np.array([speaker_map[args.speaker_id]])
        
        # 加载情感映射
        emotions = arousals = valences = None
        if model_config["multi_emotion"]:
            with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "emotions.json")) as f:
                json_raw = json.load(f)
                emotion_map = json_raw["emotion_dict"]
                arousal_map = json_raw["arousal_dict"] 
                valence_map = json_raw["valence_dict"]
            emotions = np.array([emotion_map[args.emotion_id]])
            arousals = np.array([arousal_map[args.emotion_id]])
            valences = np.array([valence_map[args.emotion_id]])
        
        # 处理音素序列
        texts = np.array([preprocess_chinese_phonemes(args.phonemes, preprocess_config)])
        text_lens = np.array([len(texts[0])])
        
        batchs = [(ids, raw_texts, speakers, emotions, arousals, valences, texts, text_lens, max(text_lens))]
        tag = f"{args.speaker_id}_{args.emotion_id}"

    control_values = args.pitch_control, args.energy_control, args.duration_control

    synthesize(model, args.restore_step, configs, vocoder, batchs, control_values, tag)
    
    print(f"\n合成完成！输出文件保存在: {train_config['path']['result_path']}") 