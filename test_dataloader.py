import os
import yaml
import torch
from torch.utils.data import DataLoader
from dataset import Dataset

def test_dataloader():
    # 加载配置
    preprocess_config = yaml.load(open("config/ESD-Chinese/preprocess.yaml", "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open("config/ESD-Chinese/model.yaml", "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open("config/ESD-Chinese/train.yaml", "r"), Loader=yaml.FullLoader)
    
    # 创建数据集
    dataset = Dataset(
        "train.txt", 
        preprocess_config, 
        model_config, 
        train_config, 
        sort=True, 
        drop_last=True
    )
    
    # 创建数据加载器
    loader = DataLoader(
        dataset,
        batch_size=train_config["optimizer"]["batch_size"],
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )
    
    print(f"数据集大小: {len(dataset)}")
    print(f"batch_size: {train_config['optimizer']['batch_size']}")
    
    # 测试第一个batch
    print("测试第一个batch...")
    for i, batchs in enumerate(loader):
        print(f"Batch {i}:")
        for j, batch in enumerate(batchs):
            print(f"  Sub-batch {j}:")
            ids, raw_texts, speakers, emotions, arousals, valences, texts, text_lens, max_text_len, mels, mel_lens, max_mel_len, pitches, energies, durations = batch
            
            print(f"    texts shape: {texts.shape}")
            print(f"    pitches shape: {pitches.shape}")
            print(f"    energies shape: {energies.shape}")
            print(f"    durations shape: {durations.shape}")
            print(f"    mels shape: {mels.shape}")
            
            # 检查第一个样本的维度
            print(f"    Sample 0:")
            print(f"      text_len: {text_lens[0]}")
            print(f"      mel_len: {mel_lens[0]}")
            print(f"      duration_sum: {durations[0][:text_lens[0]].sum()}")
            
            if texts.shape[1] != pitches.shape[1]:
                print(f"    维度不匹配: texts({texts.shape[1]}) != pitches({pitches.shape[1]})")
                
        if i >= 2:  # 只测试前3个batch
            break
    
    print("测试完成!")

if __name__ == "__main__":
    test_dataloader() 