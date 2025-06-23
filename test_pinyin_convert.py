#!/usr/bin/env python3

import yaml
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessor.esd_pinyin import prepare_align_pinyin

def main():
    print("🚀 开始中文到拼音数据转换")
    
    # 加载配置
    config = yaml.load(open('config/ESD-Chinese/preprocess.yaml', 'r'), Loader=yaml.FullLoader)
    
    # 执行拼音转换
    pinyin_path = prepare_align_pinyin(config)
    
    if pinyin_path:
        print(f"✅ 拼音数据转换完成")
        print(f"📁 拼音数据路径: {pinyin_path}")
        print("🔄 下一步: 运行MFA拼音对齐")
    else:
        print("❌ 拼音数据转换失败")

if __name__ == "__main__":
    main() 