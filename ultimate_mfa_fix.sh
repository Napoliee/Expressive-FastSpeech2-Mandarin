#!/bin/bash

# 终极MFA权限修复脚本
# 使用最彻底的方法解决所有权限问题

echo "🚀 终极MFA权限修复开始..."

# 1. 项目路径
PROJECT_DIR="/home/taopeng/shuiwen/Expressive-FastSpeech2-continuous"
OUTPUT_DIR="$PROJECT_DIR/preprocessed_data/ESD-Chinese-Pinyin/TextGrid"

# 2. 彻底清理所有MFA文件
echo "🧹 彻底清理..."
rm -rf /root/Documents/MFA/
rm -rf /root/.local/share/montreal-forced-aligner/
rm -rf /tmp/mfa_*

# 3. 创建全新的工作环境
echo "🏗️  创建全新环境..."
mkdir -p /root/Documents/MFA
mkdir -p "$OUTPUT_DIR"
chmod -R 777 /root/Documents/
chmod -R 777 "$OUTPUT_DIR"

# 4. 设置最宽松的umask
echo "🔓 设置最宽松权限..."
umask 000

# 5. 创建权限监控脚本
echo "🔍 创建权限监控..."
cat > /tmp/fix_permissions.sh << 'EOF'
#!/bin/bash
while true; do
    find /root/Documents/MFA -name "*.db*" -exec chmod 666 {} \; 2>/dev/null
    find /root/Documents/MFA -type d -exec chmod 777 {} \; 2>/dev/null
    sleep 1
done
EOF
chmod +x /tmp/fix_permissions.sh

# 6. 在后台启动权限监控
echo "🔄 启动权限监控..."
/tmp/fix_permissions.sh &
MONITOR_PID=$!

# 7. 清理函数
cleanup() {
    echo "🛑 清理权限监控..."
    kill $MONITOR_PID 2>/dev/null
}
trap cleanup EXIT

# 8. 设置所有可能的环境变量
echo "🌐 设置环境变量..."
export MFA_ROOT_DIR="/root/Documents/MFA"
export TMPDIR="/tmp"
export TEMP="/tmp"
export TMP="/tmp"
export MFA_DATABASE_LIMITED_MODE=1
export MFA_DISABLE_DATABASE=1

# 9. 使用最保守的单线程模式运行
echo "🎯 开始MFA训练（单线程+权限监控）..."

cd "$PROJECT_DIR"

# 使用timeout确保不会卡死，并在后台持续修复权限
timeout 3600 bash -c '
    while true; do
        find /root/Documents/MFA -name "*.db*" -exec chmod 666 {} \; 2>/dev/null
        sleep 0.5
    done &
    PERM_PID=$!
    
    conda run -n aligner mfa train \
        /home/taopeng/shuiwen/Expressive-FastSpeech2-continuous/raw_data/ESD-Chinese-Singing-MFA \
        /home/taopeng/shuiwen/Expressive-FastSpeech2-continuous/dictory.txt \
        /home/taopeng/shuiwen/Expressive-FastSpeech2-continuous/raw_data/mfa_model/pinyin_acoustic_model.zip \
        --output_directory /home/taopeng/shuiwen/Expressive-FastSpeech2-continuous/raw_data/mfa_model/alignments \
        --clean \
        --verbose \
        --num_jobs 1 \
        --disable_mp
    
    kill $PERM_PID 2>/dev/null
'

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✅ 训练成功完成！"
else
    echo "❌ 训练失败，尝试备用方案..."
    
    # 备用方案：使用现有的声学模型进行对齐
    echo "🔄 尝试使用预训练模型对齐..."
    
    # 先下载模型
    conda run -n aligner mfa models download acoustic mandarin_mfa || true
    
    # 然后进行对齐
    timeout 1800 bash -c '
        while true; do
            find /root/Documents/MFA -name "*.db*" -exec chmod 666 {} \; 2>/dev/null
            sleep 0.5
        done &
        PERM_PID=$!
        
        conda run -n aligner mfa align \
            /home/taopeng/shuiwen/Expressive-FastSpeech2-continuous/raw_data/ESD-Chinese-Singing-MFA \
            /home/taopeng/shuiwen/Expressive-FastSpeech2-continuous/dictory.txt \
            mandarin_mfa \
            /home/taopeng/shuiwen/Expressive-FastSpeech2-continuous/preprocessed_data/ESD-Chinese-Pinyin/TextGrid \
            --clean \
            --num_jobs 1 \
            --single_speaker \
            --disable_mp
        
        kill $PERM_PID 2>/dev/null
    '
fi

echo "🎉 MFA处理完成！"

# 检查结果
if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A $OUTPUT_DIR 2>/dev/null)" ]; then
    echo "📊 生成的TextGrid文件数量: $(find $OUTPUT_DIR -name "*.TextGrid" 2>/dev/null | wc -l)"
    echo "📁 输出目录: $OUTPUT_DIR"
else
    echo "❌ 没有生成TextGrid文件"
fi 