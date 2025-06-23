#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
分批处理MFA拼音对齐
"""

import os
import shutil
import subprocess
from pathlib import Path

def create_batch_directories(source_dir, batch_size=1000):
    """将数据分成小批次"""
    
    source_path = Path(source_dir)
    batches_dir = Path("./raw_data/ESD-Chinese-Pinyin-Batches")
    
    # 清理旧的批次目录
    if batches_dir.exists():
        shutil.rmtree(batches_dir)
    
    batches_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集所有音频文件
    all_files = []
    for speaker_dir in source_path.iterdir():
        if speaker_dir.is_dir():
            speaker_files = list(speaker_dir.glob("*.wav"))
            for wav_file in speaker_files:
                lab_file = wav_file.with_suffix(".lab")
                if lab_file.exists():
                    all_files.append((wav_file, lab_file, speaker_dir.name))
    
    print(f"📊 总共找到 {len(all_files)} 对音频+标注文件")
    
    # 分批处理
    batches = []
    for i in range(0, len(all_files), batch_size):
        batch_num = i // batch_size + 1
        batch_files = all_files[i:i + batch_size]
        
        batch_dir = batches_dir / f"batch_{batch_num:03d}"
        batch_dir.mkdir(exist_ok=True)
        
        # 按说话人组织文件
        speakers_in_batch = {}
        for wav_file, lab_file, speaker_id in batch_files:
            if speaker_id not in speakers_in_batch:
                speakers_in_batch[speaker_id] = []
                speaker_batch_dir = batch_dir / speaker_id
                speaker_batch_dir.mkdir(exist_ok=True)
            
            # 复制文件
            dst_wav = batch_dir / speaker_id / wav_file.name
            dst_lab = batch_dir / speaker_id / lab_file.name
            
            shutil.copy2(wav_file, dst_wav)
            shutil.copy2(lab_file, dst_lab)
            
            speakers_in_batch[speaker_id].append(wav_file.name)
        
        batches.append({
            'batch_num': batch_num,
            'batch_dir': batch_dir,
            'file_count': len(batch_files),
            'speakers': speakers_in_batch
        })
        
        print(f"📦 批次 {batch_num}: {len(batch_files)} 个文件, {len(speakers_in_batch)} 个说话人")
    
    return batches

def run_mfa_align_batch(batch_info):
    """对单个批次运行MFA对齐"""
    
    batch_dir = batch_info['batch_dir']
    batch_num = batch_info['batch_num']
    
    # 输出目录
    output_dir = Path(f"./preprocessed_data/ESD-Chinese-Pinyin/TextGrid_batch_{batch_num:03d}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🚀 开始对齐批次 {batch_num}")
    print(f"📁 输入目录: {batch_dir}")
    print(f"📁 输出目录: {output_dir}")
    
    # 构建MFA命令
    cmd = [
        "conda", "run", "-n", "aligner",
        "mfa", "align",
        str(batch_dir),
        "mandarin_pinyin",
        "mandarin_mfa", 
        str(output_dir)
    ]
    
    try:
        # 运行MFA对齐
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1小时超时
        
        if result.returncode == 0:
            print(f"✅ 批次 {batch_num} 对齐成功")
            return True
        else:
            print(f"❌ 批次 {batch_num} 对齐失败")
            print(f"错误输出: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏱️ 批次 {batch_num} 对齐超时")
        return False
    except Exception as e:
        print(f"💥 批次 {batch_num} 出现异常: {e}")
        return False

def merge_textgrids(batches):
    """合并所有批次的TextGrid文件"""
    
    final_output_dir = Path("./preprocessed_data/ESD-Chinese-Pinyin/TextGrid")
    final_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🔗 开始合并TextGrid文件到 {final_output_dir}")
    
    total_merged = 0
    
    for batch_info in batches:
        batch_num = batch_info['batch_num']
        batch_output_dir = Path(f"./preprocessed_data/ESD-Chinese-Pinyin/TextGrid_batch_{batch_num:03d}")
        
        if not batch_output_dir.exists():
            print(f"⚠️  批次 {batch_num} 的输出目录不存在，跳过")
            continue
        
        # 复制所有TextGrid文件
        for textgrid_file in batch_output_dir.rglob("*.TextGrid"):
            # 保持说话人目录结构
            relative_path = textgrid_file.relative_to(batch_output_dir)
            dst_file = final_output_dir / relative_path
            
            # 创建目标目录
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 复制文件
            shutil.copy2(textgrid_file, dst_file)
            total_merged += 1
    
    print(f"✅ 总共合并了 {total_merged} 个TextGrid文件")
    
    return total_merged

def main():
    """主函数"""
    
    source_dir = "./raw_data/ESD-Chinese-Pinyin"
    batch_size = 1000  # 每批1000个文件
    
    print("🎯 开始分批MFA拼音对齐")
    print(f"📂 源目录: {source_dir}")
    print(f"📊 批次大小: {batch_size}")
    
    # 1. 创建批次目录
    print("\n📦 第一步: 创建批次目录")
    batches = create_batch_directories(source_dir, batch_size)
    
    # 2. 逐个对齐批次
    print(f"\n🚀 第二步: 开始对齐 {len(batches)} 个批次")
    
    successful_batches = []
    failed_batches = []
    
    for batch_info in batches:
        success = run_mfa_align_batch(batch_info)
        if success:
            successful_batches.append(batch_info)
        else:
            failed_batches.append(batch_info)
    
    # 3. 合并结果
    if successful_batches:
        print(f"\n🔗 第三步: 合并成功的批次")
        total_merged = merge_textgrids(successful_batches)
        
        print(f"\n📊 最终统计:")
        print(f"   成功批次: {len(successful_batches)}")
        print(f"   失败批次: {len(failed_batches)}")
        print(f"   合并文件: {total_merged}")
        
        if failed_batches:
            print(f"\n❌ 失败的批次: {[b['batch_num'] for b in failed_batches]}")
        
        if total_merged > 0:
            print(f"\n🎉 拼音对齐完成!")
            print(f"📁 TextGrid文件位置: ./preprocessed_data/ESD-Chinese-Pinyin/TextGrid")
        
    else:
        print("\n💥 所有批次都失败了，请检查配置")

if __name__ == "__main__":
    main() 