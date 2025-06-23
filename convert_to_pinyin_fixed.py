#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
改进版中文转拼音脚本
- 增加错误处理
- 增加日志记录
- 支持多种拼音格式
"""

import os
import shutil
import logging
from tqdm import tqdm
from pypinyin import lazy_pinyin, Style, load_phrases_dict
from config.pinyin_config import PinyinConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/pinyin_conversion.log'),
        logging.StreamHandler()
    ]
)

def setup_custom_phrases():
    """设置自定义词组的拼音"""
    custom_phrases = {
        '嗯': [['en']],
        '呣': [['m']],
        '唔': [['wu']],
        # 添加更多自定义词组
    }
    load_phrases_dict(custom_phrases)

def chinese_to_pinyin(text, config=PinyinConfig):
    """将中文转换为拼音
    
    Args:
        text: 输入的中文文本
        config: 拼音配置对象
    
    Returns:
        str: 转换后的拼音字符串
    """
    try:
        # 移除标点符号，只保留中文字符
        chinese_chars = ''.join([char for char in text if '\u4e00' <= char <= '\u9fff'])
        
        if not chinese_chars:
            logging.warning(f"文本不包含中文字符: {text}")
            return ""
        
        # 设置拼音风格
        style = Style.TONE3 if config.TONE_NUMBERS else Style.TONE
        
        # 转换为拼音
        pinyin_list = lazy_pinyin(
            chinese_chars,
            style=style,
            neutral_tone_with_five=config.NEUTRAL_TONE_WITH_FIVE
        )
        
        # 用空格连接拼音
        pinyin_text = " ".join(pinyin_list)
        return pinyin_text
        
    except Exception as e:
        logging.error(f"拼音转换失败: {text} - {str(e)}")
        return ""

def convert_esd_to_pinyin(config=PinyinConfig):
    """转换ESD数据集为拼音版本
    
    Args:
        config: 拼音配置对象
    
    Returns:
        str: 目标目录路径
    """
    logging.info("开始批量转换中文到拼音")
    logging.info(f"源目录: {config.RAW_DATA_DIR}")
    logging.info(f"目标目录: {config.PINYIN_DATA_DIR}")
    
    if not os.path.exists(config.RAW_DATA_DIR):
        logging.error(f"源目录不存在: {config.RAW_DATA_DIR}")
        return None
    
    # 创建目标目录
    os.makedirs(config.PINYIN_DATA_DIR, exist_ok=True)
    
    # 读取filelist
    filelist_path = os.path.join(config.RAW_DATA_DIR, "filelist.txt")
    if not os.path.exists(filelist_path):
        logging.error(f"找不到filelist: {filelist_path}")
        return None
    
    # 统计信息
    stats = {
        'total_files': 0,
        'converted_files': 0,
        'failed_files': 0,
        'empty_pinyin': 0
    }
    
    pinyin_entries = []
    failed_entries = []
    
    logging.info("读取原始数据...")
    
    with open(filelist_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    logging.info(f"找到 {len(lines)} 条数据")
    
    # 设置自定义词组
    setup_custom_phrases()
    
    for line in tqdm(lines, desc="转换进度"):
        line = line.strip()
        if not line:
            continue
            
        try:
            parts = line.split('|')
            if len(parts) < 6:
                logging.warning(f"无效的文件列表条目: {line}")
                continue
                
            wav_path, text, speaker_id = parts[0], parts[1], parts[2]
            emotion = parts[5]  # 第6个字段是情感
            stats['total_files'] += 1
            
            # 转换中文为拼音
            pinyin_text = chinese_to_pinyin(text, config)
            
            if not pinyin_text:
                stats['empty_pinyin'] += 1
                failed_entries.append(f"{line} -> 空拼音")
                continue
            
            # 创建说话人目录
            speaker_target_dir = os.path.join(config.PINYIN_DATA_DIR, speaker_id)
            os.makedirs(speaker_target_dir, exist_ok=True)
            
            # 文件名
            basename = os.path.splitext(os.path.basename(wav_path))[0]
            
            # 实际音频文件路径
            actual_source_wav = os.path.join(config.RAW_DATA_DIR, speaker_id, f"{basename}.wav")
            target_wav = os.path.join(speaker_target_dir, f"{basename}.wav")
            
            if os.path.exists(actual_source_wav):
                if not os.path.exists(target_wav):
                    shutil.copy2(actual_source_wav, target_wav)
                
                # 创建拼音lab文件
                lab_path = os.path.join(speaker_target_dir, f"{basename}.lab")
                with open(lab_path, 'w', encoding='utf-8') as f:
                    f.write(pinyin_text)
                
                # 记录拼音版本的filelist条目
                pinyin_wav_path = os.path.join(speaker_id, f"{basename}.wav")
                pinyin_entries.append(f"{pinyin_wav_path}|{pinyin_text}|{speaker_id}|{emotion}")
                
                stats['converted_files'] += 1
                
            else:
                logging.warning(f"找不到音频文件: {actual_source_wav}")
                failed_entries.append(f"{line} -> 音频文件不存在")
                stats['failed_files'] += 1
                
        except Exception as e:
            logging.error(f"处理失败: {line} - {str(e)}")
            failed_entries.append(f"{line} -> {str(e)}")
            stats['failed_files'] += 1
    
    # 保存拼音版本的filelist
    pinyin_filelist_path = os.path.join(config.PINYIN_DATA_DIR, "filelist.txt")
    with open(pinyin_filelist_path, 'w', encoding='utf-8') as f:
        for entry in pinyin_entries:
            f.write(entry + '\n')
    
    # 保存失败记录
    if failed_entries:
        failed_log_path = os.path.join("./logs", "failed_conversions.txt")
        with open(failed_log_path, 'w', encoding='utf-8') as f:
            for entry in failed_entries:
                f.write(entry + '\n')
    
    # 输出统计信息
    logging.info("\n转换完成!")
    logging.info("统计信息:")
    logging.info(f"  总文件数: {stats['total_files']}")
    logging.info(f"  成功转换: {stats['converted_files']}")
    logging.info(f"  转换失败: {stats['failed_files']}")
    logging.info(f"  空拼音数: {stats['empty_pinyin']}")
    logging.info(f"  转换率: {stats['converted_files']/stats['total_files']*100:.1f}%")
    logging.info(f"拼音数据目录: {config.PINYIN_DATA_DIR}")
    logging.info(f"拼音filelist: {pinyin_filelist_path}")
    
    if failed_entries:
        logging.warning(f"失败记录已保存到: {failed_log_path}")
    
    return config.PINYIN_DATA_DIR

def test_pinyin_conversion():
    """测试拼音转换功能"""
    test_sentences = [
        "他对谁都那么友好。",
        "今天天气真不错。",
        "我很高兴见到你。",
        "这是一个测试句子。",
        "语音合成技术很有趣。",
        "嗯，这个音调不太对。",  # 测试自定义词组
        "唔，让我想想。",      # 测试自定义词组
    ]
    
    logging.info("测试拼音转换:")
    for sentence in test_sentences:
        pinyin = chinese_to_pinyin(sentence)
        logging.info(f"   {sentence} → {pinyin}")

if __name__ == "__main__":
    # 先测试转换功能
    test_pinyin_conversion()
    
    print("\n" + "="*50 + "\n")
    
    # 批量转换
    pinyin_dir = convert_esd_to_pinyin()
    
    if pinyin_dir:
        print(f"\n下一步: 运行MFA拼音对齐")
        print(f"命令: {' '.join(PinyinConfig.get_mfa_align_command(pinyin_dir, PinyinConfig.TEXTGRID_OUTPUT_DIR))}") 