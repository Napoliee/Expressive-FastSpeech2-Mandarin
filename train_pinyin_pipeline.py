#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
拼音训练流程脚本
包含：
1. 数据预处理
2. MFA对齐
3. 模型训练
"""

import os
import logging
import subprocess
from pathlib import Path
from config.pinyin_config import PinyinConfig
from convert_to_pinyin_fixed import convert_esd_to_pinyin
from parallel_mfa_align import create_small_batches, run_single_batch_align

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/training_pipeline.log'),
        logging.StreamHandler()
    ]
)

def check_environment():
    """检查环境配置"""
    try:
        # 检查MFA环境
        result = subprocess.run(
            ["conda", "run", "-n", "aligner", "mfa", "--version"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            logging.error("MFA环境检查失败，请确保已安装MFA并创建aligner环境")
            return False
            
        # 检查词典文件
        if not os.path.exists(PinyinConfig.MFA_DICT):
            logging.error(f"找不到拼音词典文件: {PinyinConfig.MFA_DICT}")
            return False
            
        return True
        
    except Exception as e:
        logging.error(f"环境检查失败: {str(e)}")
        return False

def prepare_data():
    """准备训练数据"""
    logging.info("开始准备训练数据...")
    
    # 1. 转换为拼音
    pinyin_dir = convert_esd_to_pinyin(PinyinConfig)
    if not pinyin_dir:
        logging.error("拼音转换失败")
        return False
        
    # 2. 创建输出目录
    os.makedirs(PinyinConfig.TEXTGRID_OUTPUT_DIR, exist_ok=True)
    
    return True

def run_mfa_alignment():
    """运行MFA对齐"""
    logging.info("开始MFA对齐...")
    
    # 1. 创建批次
    batches = create_small_batches(
        PinyinConfig.PINYIN_DATA_DIR,
        PinyinConfig.MFA_BATCH_SIZE
    )
    
    if not batches:
        logging.error("创建批次失败")
        return False
    
    # 2. 运行对齐
    success_count = 0
    for batch in batches:
        result = run_single_batch_align(batch)
        if result['success']:
            success_count += 1
    
    success_rate = success_count / len(batches) * 100
    logging.info(f"MFA对齐完成: {success_count}/{len(batches)} 批次成功 ({success_rate:.1f}%)")
    
    return success_rate > 80  # 要求至少80%的批次成功

def train_model():
    """训练模型"""
    logging.info("开始模型训练...")
    
    try:
        # 构建训练命令
        cmd = [
            "python", "train.py",
            "--config", "config/pinyin_train.json",
            "--input_dir", PinyinConfig.TEXTGRID_OUTPUT_DIR,
            "--checkpoint_dir", "./checkpoints/pinyin_model",
            "--log_dir", "./logs/pinyin_training"
        ]
        
        # 运行训练
        result = subprocess.run(cmd, check=True)
        
        if result.returncode == 0:
            logging.info("模型训练完成")
            return True
        else:
            logging.error("模型训练失败")
            return False
            
    except Exception as e:
        logging.error(f"训练过程出错: {str(e)}")
        return False

def main():
    """主函数"""
    logging.info("开始拼音训练流程")
    
    # 1. 检查环境
    if not check_environment():
        logging.error("环境检查失败，终止流程")
        return
    
    # 2. 准备数据
    if not prepare_data():
        logging.error("数据准备失败，终止流程")
        return
    
    # 3. MFA对齐
    if not run_mfa_alignment():
        logging.error("MFA对齐失败，终止流程")
        return
    
    # 4. 训练模型
    if not train_model():
        logging.error("模型训练失败")
        return
    
    logging.info("拼音训练流程完成!")

if __name__ == "__main__":
    main() 