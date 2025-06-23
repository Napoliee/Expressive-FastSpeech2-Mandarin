#!/usr/bin/env python3
"""
生成5种情感各5句话的语音样本
总共25个语音文件
"""

import os
import subprocess
import sys

def run_synthesis(step, text, speaker_id, emotion, output_name):
    """运行语音合成"""
    cmd = [
        "python", "synthesize_chinese_pinyin.py",
        "--restore_step", str(step),
        "--mode", "single",
        "--text", text,
        "--speaker_id", speaker_id,
        "--emotion", emotion,
        "--output_name", output_name,
        "-p", "config/ESD-Chinese-Singing-MFA/preprocess.yaml",
        "-m", "config/ESD-Chinese-Singing-MFA/model.yaml", 
        "-t", "config/ESD-Chinese-Singing-MFA/train.yaml"
    ]
    
    print(f"正在合成: {emotion} - {text} -> {output_name}.wav")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print(f"✓ 成功: {output_name}.wav")
            return True
        else:
            print(f"✗ 失败: {output_name}.wav")
            print(f"错误: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"✗ 超时: {output_name}.wav")
        return False
    except Exception as e:
        print(f"✗ 异常: {output_name}.wav - {e}")
        return False

def main():
    # 检查参数
    if len(sys.argv) != 2:
        print("用法: python generate_emotion_samples.py <restore_step>")
        print("例如: python generate_emotion_samples.py 10000")
        sys.exit(1)
    
    step = int(sys.argv[1])
    speaker_id = "0001"  # 使用说话人0001
    
    # 检查模型文件是否存在
    model_file = f"output/ckpt/ESD-Chinese-Singing-MFA/{step}.pth.tar"
    if not os.path.exists(model_file):
        print(f"错误: 找不到模型文件 {model_file}")
        print("可用的检查点:")
        ckpt_dir = "output/ckpt/ESD-Chinese-Singing-MFA/"
        if os.path.exists(ckpt_dir):
            files = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth.tar')]
            for f in sorted(files):
                print(f"  {f}")
        sys.exit(1)
    
    # 创建输出目录
    os.makedirs("output/result/ESD-Chinese-Singing-MFA/", exist_ok=True)
    
    print(f"=== 开始生成情感语音样本 ===")
    print(f"模型步数: {step}")
    print(f"说话人: {speaker_id}")
    print(f"总共将生成: 5种情感 × 5句话 = 25个音频文件")
    print()
    
    # 定义每种情感的5句话
    emotion_texts = {
        "Happy": [
            "今天天气真好",
            "我很开心见到你",
            "这个消息太棒了",
            "生活充满了希望",
            "让我们一起庆祝吧"
        ],
        "Sad": [
            "我感到很难过",
            "这真是个坏消息",
            "我想念那些美好时光",
            "心情有些沉重",
            "希望明天会更好"
        ],
        "Angry": [
            "这太让人生气了",
            "我无法忍受这种行为",
            "这完全不公平",
            "我对此非常愤怒",
            "必须要改变这种情况"
        ],
        "Surprise": [
            "这真是太意外了",
            "我没想到会这样",
            "真的吗这太神奇了",
            "这个结果令人震惊",
            "完全出乎我的意料"
        ],
        "Neutral": [
            "今天是星期一",
            "请问现在几点了",
            "我需要去买些东西",
            "会议安排在下午",
            "这是一个普通的日子"
        ]
    }
    
    # 统计
    total_count = 0
    success_count = 0
    
    # 为每种情感生成语音
    for emotion, texts in emotion_texts.items():
        print(f"\n--- 正在生成 {emotion} 情感语音 ---")
        
        for i, text in enumerate(texts, 1):
            total_count += 1
            # 创建唯一的输出文件名：情感_序号_说话人_步数
            output_name = f"{emotion}_{i:02d}_{speaker_id}_{step}"
            success = run_synthesis(step, text, speaker_id, emotion, output_name)
            if success:
                success_count += 1
            
            # 添加小延迟避免过快调用
            import time
            time.sleep(1)
    
    print(f"\n=== 生成完成 ===")
    print(f"总计: {total_count} 个文件")
    print(f"成功: {success_count} 个文件")
    print(f"失败: {total_count - success_count} 个文件")
    print(f"成功率: {success_count/total_count*100:.1f}%")
    
    # 显示输出文件
    print(f"\n输出文件位置: output/result/ESD-Chinese-Singing-MFA/")
    output_dir = "output/result/ESD-Chinese-Singing-MFA/"
    if os.path.exists(output_dir):
        wav_files = [f for f in os.listdir(output_dir) if f.endswith('.wav')]
        if wav_files:
            print(f"生成的音频文件 ({len(wav_files)} 个):")
            # 按情感分组显示
            emotions = ["Happy", "Sad", "Angry", "Surprise", "Neutral"]
            for emotion in emotions:
                emotion_files = [f for f in wav_files if f.startswith(emotion)]
                if emotion_files:
                    print(f"  {emotion}: {len(emotion_files)} 个文件")
                    for f in sorted(emotion_files):
                        print(f"    {f}")
        else:
            print("没有找到生成的音频文件")

if __name__ == "__main__":
    main() 