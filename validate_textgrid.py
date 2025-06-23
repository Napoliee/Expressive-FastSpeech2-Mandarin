#!/usr/bin/env python3
"""
验证MFA生成的TextGrid质量
"""

import os
import json
import numpy as np
import librosa
import textgrid
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm

def validate_single_textgrid(tg_path, wav_path, text_content=None):
    """验证单个TextGrid文件"""
    
    print(f"\n=== 验证文件: {os.path.basename(tg_path)} ===")
    
    # 1. 读取音频时长
    try:
        audio_duration = librosa.get_duration(filename=wav_path)
        print(f"音频时长: {audio_duration:.3f}秒")
    except Exception as e:
        print(f"无法读取音频: {e}")
        return None
    
    # 2. 读取TextGrid
    try:
        tg = textgrid.TextGrid.fromFile(tg_path)
        print(f"TextGrid时长: {tg.maxTime:.3f}秒")
        print(f"层数: {len(tg.tiers)}")
        
        for i, tier in enumerate(tg.tiers):
            print(f"  层{i}: {tier.name} ({len(tier)} intervals)")
    except Exception as e:
        print(f"无法读取TextGrid: {e}")
        return None
    
    # 3. 检查phones层
    phone_tier = None
    for tier in tg.tiers:
        if tier.name.lower() in ['phones', 'phone']:
            phone_tier = tier
            break
    
    if phone_tier is None:
        print("❌ 未找到phones层")
        return None
    
    print(f"\n📱 Phones层分析:")
    print(f"音素数量: {len(phone_tier)}")
    
    # 4. 音素统计
    phonemes = []
    durations = []
    empty_count = 0
    
    for interval in phone_tier:
        phone = interval.mark.strip()
        duration = interval.maxTime - interval.minTime
        
        if phone == '' or phone is None:
            empty_count += 1
        else:
            phonemes.append(phone)
            durations.append(duration)
    
    print(f"有效音素: {len(phonemes)}")
    print(f"空音素: {empty_count}")
    
    if len(durations) > 0:
        print(f"平均音素时长: {np.mean(durations):.3f}秒")
        print(f"最短音素: {np.min(durations):.3f}秒")
        print(f"最长音素: {np.max(durations):.3f}秒")
        
        # 检查异常短的音素
        short_phonemes = [(p, d) for p, d in zip(phonemes, durations) if d < 0.01]
        if short_phonemes:
            print(f"⚠️  异常短音素 (<0.01s): {len(short_phonemes)}个")
            for p, d in short_phonemes[:5]:
                print(f"   {p}: {d:.4f}s")
    
    # 5. 音素类型分析
    phoneme_counter = Counter(phonemes)
    print(f"\n🔤 音素类型分析:")
    print(f"唯一音素数: {len(phoneme_counter)}")
    print("最常见音素:")
    for phone, count in phoneme_counter.most_common(10):
        print(f"  {phone}: {count}次")
    
    # 6. 时间连续性检查
    print(f"\n⏰ 时间连续性检查:")
    gaps = []
    overlaps = []
    
    for i in range(len(phone_tier) - 1):
        current_end = phone_tier[i].maxTime
        next_start = phone_tier[i+1].minTime
        diff = next_start - current_end
        
        if abs(diff) > 0.001:  # 1ms容差
            if diff > 0:
                gaps.append((i, diff))
            else:
                overlaps.append((i, -diff))
    
    print(f"时间间隙: {len(gaps)}个")
    print(f"时间重叠: {len(overlaps)}个")
    
    if gaps:
        print("前5个间隙:")
        for i, gap in gaps[:5]:
            print(f"  位置{i}: {gap:.4f}s")
    
    # 7. 边界检查
    first_start = phone_tier[0].minTime
    last_end = phone_tier[-1].maxTime
    
    print(f"\n📏 边界检查:")
    print(f"首音素开始: {first_start:.3f}s")
    print(f"末音素结束: {last_end:.3f}s")
    print(f"覆盖范围: {last_end - first_start:.3f}s ({(last_end - first_start)/audio_duration*100:.1f}%)")
    
    # 8. 文本对应分析（如果提供）
    if text_content:
        print(f"\n📝 文本对应分析:")
        print(f"原文: {text_content}")
        print(f"音素序列: {' '.join(phonemes)}")
        
        # 简单的中文字符vs音素比例检查
        chinese_chars = len([c for c in text_content if '\u4e00' <= c <= '\u9fff'])
        if chinese_chars > 0:
            ratio = len(phonemes) / chinese_chars
            print(f"音素/汉字比例: {ratio:.2f} (通常在1.5-3.0之间)")
            if ratio < 1.0:
                print("⚠️  音素数量可能过少")
            elif ratio > 5.0:
                print("⚠️  音素数量可能过多")
    
    return {
        'audio_duration': audio_duration,
        'textgrid_duration': tg.maxTime,
        'phoneme_count': len(phonemes),
        'empty_count': empty_count,
        'unique_phonemes': len(phoneme_counter),
        'avg_duration': np.mean(durations) if durations else 0,
        'gaps': len(gaps),
        'overlaps': len(overlaps),
        'coverage': (last_end - first_start) / audio_duration if audio_duration > 0 else 0
    }

def batch_validate_textgrids():
    """批量验证TextGrid文件"""
    
    print("=== 批量验证TextGrid质量 ===")
    
    textgrid_dir = "preprocessed_data/ESD-Chinese/TextGrid"
    raw_data_dir = "raw_data/ESD-Chinese"
    
    results = []
    phoneme_global_counter = Counter()
    
    # 选择一些样本进行检查
    sample_files = []
    for speaker in ['0001', '0003', '0008']:
        speaker_tg_dir = os.path.join(textgrid_dir, speaker)
        if os.path.exists(speaker_tg_dir):
            tg_files = [f for f in os.listdir(speaker_tg_dir) if f.endswith('.TextGrid')]
            # 每个说话人取前5个文件
            sample_files.extend([(speaker, f) for f in tg_files[:5]])
    
    print(f"将检查 {len(sample_files)} 个样本文件")
    
    for speaker, tg_file in tqdm(sample_files):
        basename = tg_file.replace('.TextGrid', '')
        
        tg_path = os.path.join(textgrid_dir, speaker, tg_file)
        wav_path = os.path.join(raw_data_dir, speaker, f"{basename}.wav")
        lab_path = os.path.join(raw_data_dir, speaker, f"{basename}.lab")
        
        # 读取文本内容
        text_content = None
        if os.path.exists(lab_path):
            with open(lab_path, 'r', encoding='utf-8') as f:
                text_content = f.read().strip()
        
        result = validate_single_textgrid(tg_path, wav_path, text_content)
        if result:
            result['speaker'] = speaker
            result['basename'] = basename
            results.append(result)
            
            # 收集音素统计
            try:
                tg = textgrid.TextGrid.fromFile(tg_path)
                phone_tier = None
                for tier in tg.tiers:
                    if tier.name.lower() in ['phones', 'phone']:
                        phone_tier = tier
                        break
                        
                if phone_tier:
                    for interval in phone_tier:
                        phone = interval.mark.strip()
                        if phone and phone != '':
                            phoneme_global_counter[phone] += 1
            except:
                pass
    
    # 7. 总体统计
    print(f"\n{'='*50}")
    print("📊 总体质量统计")
    print(f"{'='*50}")
    
    if results:
        coverage_values = [r['coverage'] for r in results]
        phoneme_counts = [r['phoneme_count'] for r in results]
        durations = [r['avg_duration'] for r in results]
        
        print(f"平均覆盖率: {np.mean(coverage_values):.3f} ± {np.std(coverage_values):.3f}")
        print(f"平均音素数: {np.mean(phoneme_counts):.1f} ± {np.std(phoneme_counts):.1f}")
        print(f"平均音素时长: {np.mean(durations):.3f} ± {np.std(durations):.3f}秒")
        
        # 问题检测
        low_coverage = [r for r in results if r['coverage'] < 0.8]
        high_gaps = [r for r in results if r['gaps'] > 5]
        
        print(f"\n⚠️  潜在问题:")
        print(f"低覆盖率文件 (<80%): {len(low_coverage)}")
        print(f"高间隙文件 (>5个): {len(high_gaps)}")
        
        if low_coverage:
            print("低覆盖率文件:")
            for r in low_coverage[:3]:
                print(f"  {r['speaker']}_{r['basename']}: {r['coverage']:.2f}")
    
    # 8. 全局音素统计
    print(f"\n🌍 全局音素统计:")
    print(f"总音素类型: {len(phoneme_global_counter)}")
    print("最常见音素:")
    for phone, count in phoneme_global_counter.most_common(15):
        print(f"  {phone}: {count}")
    
    # 9. 生成质量报告
    generate_quality_report(results, phoneme_global_counter)

def generate_quality_report(results, phoneme_counter):
    """生成质量报告"""
    
    report = {
        'summary': {
            'total_files': len(results),
            'avg_coverage': np.mean([r['coverage'] for r in results]),
            'avg_phonemes': np.mean([r['phoneme_count'] for r in results]),
            'total_phoneme_types': len(phoneme_counter)
        },
        'phoneme_distribution': dict(phoneme_counter.most_common(20)),
        'quality_issues': {
            'low_coverage': [r for r in results if r['coverage'] < 0.8],
            'high_gaps': [r for r in results if r['gaps'] > 5],
            'short_files': [r for r in results if r['phoneme_count'] < 5]
        }
    }
    
    with open('textgrid_quality_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n📄 质量报告已保存到: textgrid_quality_report.json")

def visualize_sample_textgrid(speaker="0001", file_index=0):
    """可视化单个TextGrid文件"""
    
    textgrid_dir = "preprocessed_data/ESD-Chinese/TextGrid"
    raw_data_dir = "raw_data/ESD-Chinese"
    
    speaker_dir = os.path.join(textgrid_dir, speaker)
    tg_files = [f for f in os.listdir(speaker_dir) if f.endswith('.TextGrid')]
    
    if file_index >= len(tg_files):
        print(f"文件索引超出范围，最大索引: {len(tg_files)-1}")
        return
    
    tg_file = tg_files[file_index]
    basename = tg_file.replace('.TextGrid', '')
    
    tg_path = os.path.join(speaker_dir, tg_file)
    wav_path = os.path.join(raw_data_dir, speaker, f"{basename}.wav")
    
    print(f"可视化文件: {speaker}_{basename}")
    
    # 读取音频
    y, sr = librosa.load(wav_path)
    
    # 读取TextGrid
    tg = textgrid.TextGrid.fromFile(tg_path)
    phone_tier = None
    for tier in tg.tiers:
        if tier.name.lower() in ['phones', 'phone']:
            phone_tier = tier
            break
    
    if phone_tier is None:
        print("未找到phones层")
        return
    
    # 绘制波形和音素标注
    plt.figure(figsize=(15, 8))
    
    # 上部分：波形
    plt.subplot(2, 1, 1)
    time_axis = np.linspace(0, len(y)/sr, len(y))
    plt.plot(time_axis, y)
    plt.title(f'Audio Waveform: {speaker}_{basename}')
    plt.ylabel('Amplitude')
    
    # 添加音素边界线
    for interval in phone_tier:
        plt.axvline(x=interval.minTime, color='red', alpha=0.5, linestyle='--')
        plt.axvline(x=interval.maxTime, color='red', alpha=0.5, linestyle='--')
    
    # 下部分：音素标注
    plt.subplot(2, 1, 2)
    for i, interval in enumerate(phone_tier):
        start_time = interval.minTime
        end_time = interval.maxTime
        phone = interval.mark.strip()
        
        # 绘制音素区间
        plt.barh(0, end_time - start_time, left=start_time, height=0.8, 
                alpha=0.7, color=plt.cm.Set3(i % 12))
        
        # 添加音素标签
        mid_time = (start_time + end_time) / 2
        if phone and phone != '':
            plt.text(mid_time, 0, phone, ha='center', va='center', 
                    fontsize=8, rotation=90)
    
    plt.xlim(0, len(y)/sr)
    plt.ylim(-0.5, 0.5)
    plt.ylabel('Phonemes')
    plt.xlabel('Time (s)')
    plt.title('Phoneme Alignment')
    
    plt.tight_layout()
    plt.savefig(f'textgrid_visualization_{speaker}_{basename}.png', dpi=150)
    plt.show()
    
    print(f"可视化图片已保存: textgrid_visualization_{speaker}_{basename}.png")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "batch", "visualize"], 
                       default="batch", help="验证模式")
    parser.add_argument("--speaker", default="0001", help="说话人ID")
    parser.add_argument("--file_index", type=int, default=0, help="文件索引")
    args = parser.parse_args()
    
    if args.mode == "batch":
        batch_validate_textgrids()
    elif args.mode == "visualize":
        visualize_sample_textgrid(args.speaker, args.file_index)
    else:
        # Single mode - 这里可以添加单文件验证逻辑
        print("单文件验证模式") 