#!/bin/bash

# 修复MFA权限问题的完整脚本

echo "🔧 修复MFA权限问题..."

# 1. 创建MFA工作目录并设置权限
MFA_WORK_DIR="/tmp/mfa_work"
mkdir -p "$MFA_WORK_DIR"
chmod 777 "$MFA_WORK_DIR"

# 2. 创建MFA输出目录并设置权限
mkdir -p /root/Documents/MFA
chmod 777 /root/Documents/MFA
mkdir -p /root/.local/share/montreal-forced-aligner
chmod 777 /root/.local/share/montreal-forced-aligner

# 3. 设置环境变量
export MFA_ROOT_DIR="$MFA_WORK_DIR"
export TMPDIR="$MFA_WORK_DIR"

# 4. 创建项目特定的输出目录
OUTPUT_DIR="/home/taopeng/shuiwen/Expressive-FastSpeech2-continuous/preprocessed_data/ESD-Chinese-Pinyin/TextGrid"
mkdir -p "$OUTPUT_DIR"
chmod 777 "$OUTPUT_DIR"

# 5. 创建MFA内部工作目录
mkdir -p "/root/Documents/MFA/ESD-Chinese-Singing-MFA/alignment"
chmod 777 "/root/Documents/MFA/ESD-Chinese-Singing-MFA/alignment"

echo "✅ 权限设置完成"

# 6. 首先尝试对齐
echo "🚀 开始MFA对齐..."

conda run -n aligner mfa align \
    /home/taopeng/shuiwen/Expressive-FastSpeech2-continuous/raw_data/ESD-Chinese-Singing-MFA \
    /home/taopeng/shuiwen/Expressive-FastSpeech2-continuous/dictory.txt \
    mandarin_mfa \
    "$OUTPUT_DIR" \
    --clean \
    --verbose \
    --num_jobs 10 \
    --single_speaker

echo "📊 对齐状态检查..."
if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A $OUTPUT_DIR)" ]; then
    echo "✅ 对齐成功！生成了 $(find $OUTPUT_DIR -name "*.TextGrid" | wc -l) 个TextGrid文件"
else
    echo "❌ 对齐失败，尝试使用单线程模式..."
    
    # 备用方案：单线程模式
    conda run -n aligner mfa align \
        /home/taopeng/shuiwen/Expressive-FastSpeech2-continuous/raw_data/ESD-Chinese-Singing-MFA \
        /home/taopeng/shuiwen/Expressive-FastSpeech2-continuous/dictory.txt \
        mandarin_mfa \
        "$OUTPUT_DIR" \
        --clean \
        --verbose \
        --num_jobs 1
fi

echo "🎉 MFA对齐完成！" 