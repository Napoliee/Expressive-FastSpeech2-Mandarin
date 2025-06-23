#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
拼音处理相关配置
"""

class PinyinConfig:
    # 数据目录
    RAW_DATA_DIR = "./raw_data/ESD-Chinese"
    PINYIN_DATA_DIR = "./raw_data/ESD-Chinese-Pinyin"
    TEXTGRID_OUTPUT_DIR = "./preprocessed_data/ESD-Chinese-Pinyin/TextGrid"
    
    # MFA配置
    MFA_DICT = "./lexicon/mandarin_pinyin.dict"
    MFA_MODEL = "mandarin_mfa"
    MFA_BATCH_SIZE = 200
    MFA_NUM_JOBS = 2
    MFA_MAX_WORKERS = 3
    
    # 拼音处理配置
    TONE_NUMBERS = True  # 使用数字声调（如: ma1 vs mā）
    NEUTRAL_TONE_WITH_FIVE = True  # 轻声用数字5表示
    SEPARATE_SYLLABLE = True  # 是否分离音节（如: "zhong1 guo2" vs "zh ong1 g uo2"）
    
    # 预处理配置
    SAMPLING_RATE = 22050
    MAX_WAV_VALUE = 32768.0
    
    # 训练配置
    BATCH_SIZE = 16
    EPOCHS = 200
    LEARNING_RATE = 0.001
    
    @classmethod
    def get_mfa_align_command(cls, input_dir, output_dir):
        """获取MFA对齐命令"""
        return [
            "conda", "run", "-n", "aligner",
            "mfa", "align",
            "--clean",
            f"--num_jobs={cls.MFA_NUM_JOBS}",
            str(input_dir),
            cls.MFA_DICT,
            cls.MFA_MODEL,
            str(output_dir)
        ] 