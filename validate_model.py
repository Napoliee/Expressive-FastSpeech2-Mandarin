#!/usr/bin/env python3
"""
验证模型训练效果的脚本
在训练集样本上测试重建能力
"""

import argparse
import os
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_one_sample
from dataset_ipa_fixed import Dataset
import soundfile as sf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def validate_model_on_training_data():
    """在训练数据上验证模型效果"""
    
    # Read Config
    preprocess_config = yaml.load(open("config/ESD-Chinese/preprocess.yaml", "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open("config/ESD-Chinese/model.yaml", "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open("config/ESD-Chinese/train.yaml", "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # 获取最新checkpoint
    checkpoint_dir = "output/ckpt/ESD-Chinese"
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth.tar')]
    if not checkpoints:
        print("未找到checkpoint文件")
        return
    
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('.')[0]))
    restore_step = int(latest_checkpoint.split('.')[0])
    
    print(f"使用checkpoint: {latest_checkpoint} (step {restore_step})")

    # 创建一个假的args对象
    class Args:
        def __init__(self):
            self.restore_step = restore_step
    
    args = Args()

    # Get model
    model = get_model(args, configs, device, train=False)
    model.eval()

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # 加载验证数据集（少量样本）
    val_dataset = Dataset(
        "val_ipa.txt", 
        preprocess_config, 
        model_config, 
        train_config,
        sort=False,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        collate_fn=val_dataset.collate_fn,
    )

    print(f"验证集样本数: {len(val_dataset)}")
    
    # 测试前5个样本
    test_samples = []
    
    with torch.no_grad():
        for i, batchs in enumerate(val_loader):
            if i >= 5:  # 只测试前5个样本
                break
            
            # val_loader返回的是batch列表，需要取第一个batch
            if not batchs or len(batchs) == 0:
                print(f"第{i}个batch为空，跳过")
                continue
                
            batch = batchs[0]  # 取第一个batch
            print(f"Batch结构长度: {len(batch)}")
            print(f"Batch元素类型: {[type(x) for x in batch]}")
            
            batch = to_device(batch, device)
            
            # Forward pass
            try:
                output = model(*(batch[2:]))
            except Exception as e:
                print(f"模型推理失败: {e}")
                print(f"Batch[2:]长度: {len(batch[2:])}")
                for j, item in enumerate(batch[2:]):
                    print(f"  Batch[{j+2}]类型: {type(item)}, 形状: {getattr(item, 'shape', 'N/A')}")
                continue
            
            # 获取样本信息
            basename = batch[0][0]
            speaker_id = batch[2][0].item()
            text_sequence = batch[6][0].cpu().numpy()
            raw_text = batch[1][0]
            
            print(f"\n样本 {i+1}: {basename}")
            print(f"原文: {raw_text}")
            print(f"说话人: {speaker_id}")
            print(f"文本序列长度: {len(text_sequence)}")
            
            # 获取mel-spectrogram预测和真实值
            mel_prediction = output[1][0].detach().cpu().numpy()
            mel_target = batch[9][0].detach().cpu().numpy()
            
            print(f"预测mel形状: {mel_prediction.shape}")
            print(f"真实mel形状: {mel_target.shape}")
            
            # 计算重建loss
            mel_loss = np.mean((mel_prediction - mel_target) ** 2)
            print(f"Mel重建MSE Loss: {mel_loss:.4f}")
            
            # 使用vocoder生成音频
            if vocoder is not None:
                try:
                    # 重建音频（使用真实mel）
                    from utils.model import vocoder_infer
                    
                    mel_target_tensor = torch.FloatTensor(mel_target).unsqueeze(0).transpose(1, 2).to(device)
                    wav_reconstruction = vocoder_infer(
                        mel_target_tensor,
                        vocoder,
                        model_config,
                        preprocess_config,
                    )[0]
                    
                    # 预测音频
                    mel_prediction_tensor = torch.FloatTensor(mel_prediction).unsqueeze(0).transpose(1, 2).to(device)
                    wav_prediction = vocoder_infer(
                        mel_prediction_tensor,
                        vocoder,
                        model_config,
                        preprocess_config,
                    )[0]
                    
                    # 保存音频
                    output_dir = "validation_outputs"
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # 保存重建音频（Ground Truth Mel -> Audio）
                    sf.write(
                        os.path.join(output_dir, f"{basename}_reconstruction.wav"),
                        wav_reconstruction,
                        22050
                    )
                    
                    # 保存预测音频（Predicted Mel -> Audio）
                    sf.write(
                        os.path.join(output_dir, f"{basename}_prediction.wav"),
                        wav_prediction,
                        22050
                    )
                    
                    print(f"✅ 音频已保存到 {output_dir}/")
                    print(f"   - {basename}_reconstruction.wav (真实mel重建)")
                    print(f"   - {basename}_prediction.wav (模型预测)")
                    
                    # 检查音频质量指标
                    print(f"重建音频长度: {len(wav_reconstruction)/22050:.2f}秒")
                    print(f"预测音频长度: {len(wav_prediction)/22050:.2f}秒")
                    
                    # 检查音频幅度
                    print(f"重建音频RMS: {np.sqrt(np.mean(wav_reconstruction**2)):.4f}")
                    print(f"预测音频RMS: {np.sqrt(np.mean(wav_prediction**2)):.4f}")
                    
                except Exception as e:
                    print(f"❌ Vocoder合成失败: {e}")
            
            test_samples.append({
                'basename': basename,
                'mel_loss': mel_loss,
                'mel_shape': mel_prediction.shape,
                'text_len': len(text_sequence)
            })
    
    # 总结
    print(f"\n{'='*50}")
    print("验证总结")
    print(f"{'='*50}")
    
    avg_mel_loss = np.mean([s['mel_loss'] for s in test_samples])
    print(f"平均Mel重建Loss: {avg_mel_loss:.4f}")
    
    mel_shapes = [s['mel_shape'] for s in test_samples]
    print(f"Mel频谱形状范围: {mel_shapes}")
    
    text_lens = [s['text_len'] for s in test_samples]
    print(f"文本序列长度范围: {min(text_lens)} - {max(text_lens)}")
    
    # 判断训练质量
    if avg_mel_loss < 0.1:
        print("✅ 训练质量良好 - Mel重建Loss较低")
    elif avg_mel_loss < 0.5:
        print("⚠️  训练质量一般 - Mel重建Loss中等")
    else:
        print("❌ 训练质量差 - Mel重建Loss过高")
    
    print(f"\n📁 验证音频保存在: validation_outputs/")
    print("请听这些音频文件来判断训练效果:")
    for sample in test_samples:
        print(f"  - {sample['basename']}_reconstruction.wav (应该清晰)")
        print(f"  - {sample['basename']}_prediction.wav (检查模型预测质量)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5, help="测试样本数量")
    args = parser.parse_args()
    
    print("🔍 开始验证模型训练效果...")
    validate_model_on_training_data()

if __name__ == "__main__":
    main() 