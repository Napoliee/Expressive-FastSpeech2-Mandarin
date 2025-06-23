#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
快速测试MFA拼音对齐 - 使用小批次
"""

import os
import shutil
import subprocess
import time
from pathlib import Path

def create_small_test_batch(source_dir, batch_size=50):
    """创建一个小测试批次"""
    
    source_path = Path(source_dir)
    test_dir = Path("./raw_data/ESD-Chinese-Pinyin-QuickTest")
    
    # 清理旧的测试目录
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集前batch_size个文件
    collected_files = []
    for speaker_dir in source_path.iterdir():
        if speaker_dir.is_dir():
            speaker_files = list(speaker_dir.glob("*.wav"))[:batch_size//10]  # 每个说话人取几个文件
            for wav_file in speaker_files:
                lab_file = wav_file.with_suffix(".lab")
                if lab_file.exists():
                    collected_files.append((wav_file, lab_file, speaker_dir.name))
                    if len(collected_files) >= batch_size:
                        break
        if len(collected_files) >= batch_size:
            break
    
    print(f"📊 收集了 {len(collected_files)} 个文件进行快速测试")
    
    # 复制文件到测试目录
    speakers_in_test = set()
    for wav_file, lab_file, speaker_id in collected_files:
        speaker_test_dir = test_dir / speaker_id
        speaker_test_dir.mkdir(exist_ok=True)
        
        # 复制文件
        shutil.copy2(wav_file, speaker_test_dir / wav_file.name)
        shutil.copy2(lab_file, speaker_test_dir / lab_file.name)
        
        speakers_in_test.add(speaker_id)
    
    print(f"📦 测试批次包含 {len(speakers_in_test)} 个说话人")
    
    return test_dir, len(collected_files)

def run_quick_mfa_test(test_dir, file_count):
    """运行快速MFA测试"""
    
    output_dir = Path("./preprocessed_data/ESD-Chinese-Pinyin/TextGrid_QuickTest")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🚀 开始快速MFA测试")
    print(f"📁 输入目录: {test_dir}")
    print(f"📁 输出目录: {output_dir}")
    print(f"📊 文件数量: {file_count}")
    
    # 构建MFA命令
    cmd = [
        "conda", "run", "-n", "aligner",
        "mfa", "align",
        str(test_dir),
        "mandarin_pinyin",
        "mandarin_mfa", 
        str(output_dir)
    ]
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 运行MFA对齐
        print("⏱️  MFA对齐开始...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30分钟超时
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ 快速测试成功!")
            print(f"⏱️  耗时: {duration:.1f} 秒")
            print(f"📈 平均每文件: {duration/file_count:.2f} 秒")
            
            # 检查输出文件
            textgrid_files = list(output_dir.rglob("*.TextGrid"))
            print(f"📄 生成了 {len(textgrid_files)} 个TextGrid文件")
            
            return True, duration, len(textgrid_files)
        else:
            print(f"❌ 快速测试失败")
            print(f"错误输出: {result.stderr}")
            return False, duration, 0
            
    except subprocess.TimeoutExpired:
        print(f"⏱️ 快速测试超时")
        return False, 1800, 0
    except Exception as e:
        print(f"💥 快速测试出现异常: {e}")
        return False, 0, 0

def estimate_full_processing_time(test_duration, test_files, total_files):
    """估算全量处理时间"""
    
    if test_files == 0:
        return "无法估算"
    
    avg_time_per_file = test_duration / test_files
    estimated_total_time = avg_time_per_file * total_files
    
    # 考虑批次开销
    batch_overhead = 1.2  # 20%的额外开销
    estimated_total_time *= batch_overhead
    
    hours = estimated_total_time / 3600
    
    return f"{hours:.1f} 小时 ({estimated_total_time/60:.0f} 分钟)"

def main():
    """主函数"""
    
    source_dir = "./raw_data/ESD-Chinese-Pinyin"
    test_batch_size = 50  # 快速测试只用50个文件
    
    print("🎯 快速MFA拼音对齐测试")
    print(f"📂 源目录: {source_dir}")
    print(f"📊 测试批次大小: {test_batch_size}")
    
    # 1. 创建小测试批次
    print("\n📦 第一步: 创建测试批次")
    test_dir, file_count = create_small_test_batch(source_dir, test_batch_size)
    
    if file_count == 0:
        print("❌ 没有找到可用的文件")
        return
    
    # 2. 运行快速测试
    print(f"\n🚀 第二步: 运行快速测试")
    success, duration, output_files = run_quick_mfa_test(test_dir, file_count)
    
    # 3. 估算全量处理时间
    if success:
        total_files = 17500  # ESD数据集总文件数
        estimated_time = estimate_full_processing_time(duration, file_count, total_files)
        
        print(f"\n📊 性能分析:")
        print(f"   测试文件数: {file_count}")
        print(f"   测试耗时: {duration:.1f} 秒")
        print(f"   平均每文件: {duration/file_count:.2f} 秒")
        print(f"   生成TextGrid: {output_files}")
        print(f"\n⏱️  全量处理时间估算:")
        print(f"   总文件数: {total_files}")
        print(f"   预计耗时: {estimated_time}")
        
        # 给出建议
        if duration/file_count > 10:  # 每文件超过10秒
            print(f"\n💡 优化建议:")
            print(f"   • 当前速度较慢，建议使用更小的批次(100-200个文件)")
            print(f"   • 可以考虑并行处理多个批次")
            print(f"   • 检查系统资源(CPU/内存/磁盘)")
        else:
            print(f"\n✅ 速度正常，可以继续全量处理")
            
    else:
        print(f"\n❌ 测试失败，需要检查配置")

if __name__ == "__main__":
    main() 