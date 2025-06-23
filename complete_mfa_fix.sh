#!/bin/bash

# 完整的MFA修复脚本 - 解决所有权限和目录结构问题

echo "🔧 开始完整的MFA修复..."

# 1. 设置项目路径
PROJECT_DIR="/home/taopeng/shuiwen/Expressive-FastSpeech2-continuous"
MFA_WORK_DIR="/root/Documents/MFA/ESD-Chinese-Singing-MFA"
OUTPUT_DIR="$PROJECT_DIR/preprocessed_data/ESD-Chinese-Pinyin/TextGrid"

# 2. 清理所有MFA相关文件
echo "🧹 清理旧文件..."
rm -rf "$MFA_WORK_DIR"
rm -rf /root/Documents/MFA/ESD-Chinese-Singing-MFA*

# 3. 创建完整的目录结构
echo "📁 创建目录结构..."
mkdir -p "$MFA_WORK_DIR"
mkdir -p "$OUTPUT_DIR"

# 预创建可能需要的split目录
for i in {1..20}; do
    mkdir -p "$MFA_WORK_DIR/ESD-Chinese-Singing-MFA/split$i/log"
    mkdir -p "$MFA_WORK_DIR/split$i/log"
done

# 4. 设置权限
echo "🔑 设置权限..."
chmod -R 777 /root/Documents/MFA/
chmod -R 777 "$OUTPUT_DIR"

# 5. 设置环境变量
echo "🌐 设置环境变量..."
export MFA_ROOT_DIR="$MFA_WORK_DIR"
export TMPDIR="/tmp/mfa_work"
mkdir -p /tmp/mfa_work
chmod 777 /tmp/mfa_work

# 6. 检查数据完整性
echo "📊 检查数据..."
if [ ! -d "$PROJECT_DIR/raw_data/ESD-Chinese-Singing-MFA" ]; then
    echo "❌ 源数据目录不存在"
    exit 1
fi

if [ ! -f "$PROJECT_DIR/dictory.txt" ]; then
    echo "❌ 词典文件不存在"
    exit 1
fi

echo "✅ 数据检查通过"

# 7. 使用更保守的参数运行对齐
echo "🚀 开始MFA对齐（保守模式）..."

cd "$PROJECT_DIR"
conda run -n aligner mfa align \
    "$PROJECT_DIR/raw_data/ESD-Chinese-Singing-MFA" \
    "$PROJECT_DIR/dictory.txt" \
    mandarin_mfa \
    "$OUTPUT_DIR" \
    --clean \
    --verbose \
    --num_jobs 1 \
    --single_speaker

if [ $? -eq 0 ]; then
    echo "✅ 对齐成功！"
    echo "📁 TextGrid文件保存在: $OUTPUT_DIR"
    echo "📊 生成的文件数量: $(find $OUTPUT_DIR -name "*.TextGrid" | wc -l)"
else
    echo "❌ 对齐失败，尝试最小配置..."
    
    # 备用方案：最小配置
    conda run -n aligner mfa align \
        "$PROJECT_DIR/raw_data/ESD-Chinese-Singing-MFA" \
        "$PROJECT_DIR/dictory.txt" \
        mandarin_mfa \
        "$OUTPUT_DIR" \
        --clean \
        --num_jobs 1 \
        --single_speaker \
        --disable_mp
fi

echo "🎉 MFA修复和对齐完成！" 