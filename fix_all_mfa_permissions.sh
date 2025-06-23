#!/bin/bash

# 完整的MFA权限修复脚本
# 解决所有数据库和文件权限问题

echo "🔧 开始完整的MFA权限修复..."

# 1. 修复所有MFA相关目录权限
echo "📁 修复目录权限..."
chmod -R 777 /root/Documents/MFA/ 2>/dev/null || true
chmod -R 777 /root/.local/share/montreal-forced-aligner/ 2>/dev/null || true

# 2. 修复所有数据库文件权限
echo "🗄️  修复数据库权限..."
find /root/Documents/MFA -name "*.db" -exec chmod 666 {} \; 2>/dev/null || true
find /root/Documents/MFA -name "*.db-journal" -exec chmod 666 {} \; 2>/dev/null || true
find /root/Documents/MFA -name "*.db-wal" -exec chmod 666 {} \; 2>/dev/null || true

# 3. 设置环境变量
echo "🌐 设置环境变量..."
export MFA_ROOT_DIR="/tmp/mfa_work"
export TMPDIR="/tmp/mfa_work"
export TEMP="/tmp/mfa_work"
mkdir -p /tmp/mfa_work
chmod 777 /tmp/mfa_work

# 4. 创建输出目录
echo "📂 创建输出目录..."
OUTPUT_DIR="/home/taopeng/shuiwen/Expressive-FastSpeech2-continuous/preprocessed_data/ESD-Chinese-Pinyin/TextGrid"
mkdir -p "$OUTPUT_DIR"
chmod 777 "$OUTPUT_DIR"

# 5. 删除可能损坏的数据库文件并重新创建
echo "🔄 清理可能损坏的数据库..."
DB_FILE="/root/Documents/MFA/ESD-Chinese-Singing-MFA/ESD-Chinese-Singing-MFA.db"
if [ -f "$DB_FILE" ]; then
    echo "删除现有数据库文件..."
    rm -f "$DB_FILE"
    rm -f "$DB_FILE-journal"
    rm -f "$DB_FILE-wal"
fi

echo "✅ 权限修复完成！"

# 6. 测试权限是否正确
echo "🧪 测试权限..."
TEST_DIR="/root/Documents/MFA/test_write_permission"
mkdir -p "$TEST_DIR"
if echo "test" > "$TEST_DIR/test.txt" 2>/dev/null; then
    echo "✅ 写入权限正常"
    rm -f "$TEST_DIR/test.txt"
    rmdir "$TEST_DIR"
else
    echo "❌ 写入权限仍有问题"
fi

echo "�� 现在可以重新运行MFA命令了！" 