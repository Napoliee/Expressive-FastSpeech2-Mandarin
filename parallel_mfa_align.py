#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
并行MFA拼音对齐 - 后台运行版本
"""

import os
import shutil
import subprocess
import time
import json
from pathlib import Path
import concurrent.futures
from datetime import datetime

def create_small_batches(source_dir, batch_size=200):
    """将数据分成小批次（200个文件）"""
    
    source_path = Path(source_dir)
    batches_dir = Path("./raw_data/ESD-Chinese-Pinyin-SmallBatches")
    
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
            'speakers': list(speakers_in_batch.keys())
        })
        
        print(f"📦 批次 {batch_num}: {len(batch_files)} 个文件, {len(speakers_in_batch)} 个说话人")
    
    return batches

def run_single_batch_align(batch_info):
    """对单个批次运行MFA对齐"""
    
    batch_dir = batch_info['batch_dir']
    batch_num = batch_info['batch_num']
    
    # 输出目录
    output_dir = Path(f"./preprocessed_data/ESD-Chinese-Pinyin/TextGrid_batch_{batch_num:03d}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 日志文件
    log_file = Path(f"./logs/mfa_batch_{batch_num:03d}.log")
    log_file.parent.mkdir(exist_ok=True)
    
    start_time = time.time()
    
    print(f"🚀 [{datetime.now().strftime('%H:%M:%S')}] 开始批次 {batch_num}")
    
    # 构建MFA命令
    cmd = [
        "conda", "run", "-n", "aligner",
        "mfa", "align",
        "--clean",  # 清理中间文件
        "--num_jobs", "2",  # 限制并行度
        str(batch_dir),
        "mandarin_pinyin",
        "mandarin_mfa", 
        str(output_dir)
    ]
    
    try:
        # 运行MFA对齐，将输出重定向到日志文件
        with open(log_file, 'w', encoding='utf-8') as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=7200)  # 2小时超时
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            # 检查输出文件
            textgrid_files = list(output_dir.rglob("*.TextGrid"))
            print(f"✅ [{datetime.now().strftime('%H:%M:%S')}] 批次 {batch_num} 完成 - {len(textgrid_files)} 文件 - {duration/60:.1f}分钟")
            return {
                'batch_num': batch_num,
                'success': True,
                'duration': duration,
                'output_files': len(textgrid_files),
                'file_count': batch_info['file_count']
            }
        else:
            print(f"❌ [{datetime.now().strftime('%H:%M:%S')}] 批次 {batch_num} 失败")
            return {
                'batch_num': batch_num,
                'success': False,
                'duration': duration,
                'output_files': 0,
                'file_count': batch_info['file_count']
            }
            
    except subprocess.TimeoutExpired:
        print(f"⏱️ [{datetime.now().strftime('%H:%M:%S')}] 批次 {batch_num} 超时")
        return {
            'batch_num': batch_num,
            'success': False,
            'duration': 7200,
            'output_files': 0,
            'file_count': batch_info['file_count']
        }
    except Exception as e:
        print(f"💥 [{datetime.now().strftime('%H:%M:%S')}] 批次 {batch_num} 异常: {e}")
        return {
            'batch_num': batch_num,
            'success': False,
            'duration': 0,
            'output_files': 0,
            'file_count': batch_info['file_count']
        }

def save_progress(results, progress_file="./logs/mfa_progress.json"):
    """保存进度"""
    Path(progress_file).parent.mkdir(exist_ok=True)
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def main():
    """主函数 - 并行处理"""
    
    source_dir = "./raw_data/ESD-Chinese-Pinyin"
    batch_size = 200  # 每批200个文件
    max_workers = 3   # 最多3个并行进程
    
    print("🎯 并行MFA拼音对齐")
    print(f"📂 源目录: {source_dir}")
    print(f"📊 批次大小: {batch_size}")
    print(f"🔧 并行进程: {max_workers}")
    
    # 1. 创建批次目录
    print(f"\n📦 创建批次目录...")
    batches = create_small_batches(source_dir, batch_size)
    
    if not batches:
        print("❌ 没有找到可处理的文件")
        return
    
    print(f"\n🚀 开始并行处理 {len(batches)} 个批次")
    print(f"⏱️  预计耗时: {len(batches) * 10 / max_workers:.0f} - {len(batches) * 30 / max_workers:.0f} 分钟")
    
    # 2. 并行处理批次
    all_results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_batch = {executor.submit(run_single_batch_align, batch): batch for batch in batches}
        
        # 收集结果
        for future in concurrent.futures.as_completed(future_to_batch):
            result = future.result()
            all_results.append(result)
            
            # 保存进度
            save_progress(all_results)
            
            # 打印当前进度
            completed = len(all_results)
            total = len(batches)
            successful = sum(1 for r in all_results if r['success'])
            
            print(f"\n📊 进度: {completed}/{total} ({completed/total*100:.1f}%) - 成功: {successful}")
    
    # 3. 合并结果
    successful_batches = [r for r in all_results if r['success']]
    failed_batches = [r for r in all_results if not r['success']]
    
    if successful_batches:
        print(f"\n🔗 合并TextGrid文件...")
        merge_textgrids_from_results(successful_batches)
    
    # 4. 最终统计
    total_files = sum(r['file_count'] for r in all_results)
    successful_files = sum(r['output_files'] for r in successful_batches)
    total_time = sum(r['duration'] for r in all_results)
    
    print(f"\n🎉 处理完成!")
    print(f"📊 最终统计:")
    print(f"   总批次: {len(batches)}")
    print(f"   成功批次: {len(successful_batches)}")
    print(f"   失败批次: {len(failed_batches)}")
    print(f"   总文件: {total_files}")
    print(f"   成功文件: {successful_files}")
    print(f"   成功率: {successful_files/total_files*100:.1f}%")
    print(f"   总耗时: {total_time/3600:.1f} 小时")
    
    if failed_batches:
        print(f"\n❌ 失败批次: {[r['batch_num'] for r in failed_batches]}")
    
    print(f"\n📁 TextGrid文件位置: ./preprocessed_data/ESD-Chinese-Pinyin/TextGrid")

def merge_textgrids_from_results(successful_results):
    """从结果中合并TextGrid文件"""
    
    final_output_dir = Path("./preprocessed_data/ESD-Chinese-Pinyin/TextGrid")
    final_output_dir.mkdir(parents=True, exist_ok=True)
    
    total_merged = 0
    
    for result in successful_results:
        batch_num = result['batch_num']
        batch_output_dir = Path(f"./preprocessed_data/ESD-Chinese-Pinyin/TextGrid_batch_{batch_num:03d}")
        
        if not batch_output_dir.exists():
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

if __name__ == "__main__":
    main() 