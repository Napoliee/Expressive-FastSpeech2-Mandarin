#!/bin/bash
# 中文情感语音合成测试脚本

echo "=== 中文情感语音合成测试 ==="

# 配置
STEP=1000  # 使用较早的检查点进行测试
CONFIG_DIR="config/ESD-Chinese-Singing-MFA"

# 检查模型文件是否存在
if [ ! -f "output/ckpt/ESD-Chinese-Singing-MFA/${STEP}.pth.tar" ]; then
    echo "错误：找不到模型检查点 output/ckpt/ESD-Chinese-Singing-MFA/${STEP}.pth.tar"
    echo "请检查训练是否已开始并保存了检查点"
    echo "可用的检查点："
    ls output/ckpt/ESD-Chinese-Singing-MFA/ 2>/dev/null || echo "检查点目录不存在"
    exit 1
fi

# 创建输出目录
mkdir -p output/result/ESD-Chinese-Singing-MFA/

echo "使用检查点: ${STEP}"
echo "开始测试不同情感的语音合成..."

# 测试文本
TEXT="今天天气真好"
SPEAKER="0001"

# 测试不同情感
for emotion in "Happy" "Sad" "Angry" "Surprise" "Neutral"; do
    echo "正在合成: $emotion 情感..."
    
    python synthesize_chinese_pinyin.py \
        --restore_step $STEP \
        --mode single \
        --text "$TEXT" \
        --speaker_id "$SPEAKER" \
        --emotion "$emotion" \
        -p $CONFIG_DIR/preprocess.yaml \
        -m $CONFIG_DIR/model.yaml \
        -t $CONFIG_DIR/train.yaml
    
    if [ $? -eq 0 ]; then
        echo "✓ $emotion 情感合成成功"
    else
        echo "✗ $emotion 情感合成失败"
    fi
done

echo ""
echo "=== 测试完成 ==="
echo "输出文件位置: output/result/ESD-Chinese-Singing-MFA/"
echo "查看生成的音频文件："
ls -la output/result/ESD-Chinese-Singing-MFA/*.wav 2>/dev/null || echo "没有找到生成的音频文件"

echo ""
echo "=== 测试音素序列输入 ==="
python synthesize_chinese_pinyin.py \
    --restore_step $STEP \
    --mode single \
    --text "{ni hao shi jie}" \
    --speaker_id "0002" \
    --emotion "Happy" \
    -p $CONFIG_DIR/preprocess.yaml \
    -m $CONFIG_DIR/model.yaml \
    -t $CONFIG_DIR/train.yaml

echo "测试脚本执行完毕！" 