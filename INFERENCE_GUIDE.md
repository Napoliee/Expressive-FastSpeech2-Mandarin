# 中文情感语音合成推理指南

## 模型推理方法

训练完成后，可以使用以下方法进行语音合成：

### 1. 单句合成

```bash
python synthesize_chinese_pinyin.py \
    --restore_step 50000 \
    --mode single \
    --text "你好世界" \
    --speaker_id "0001" \
    --emotion "Happy" \
    --pitch_control 1.0 \
    --energy_control 1.0 \
    --duration_control 1.0 \
    -p config/ESD-Chinese-Singing-MFA/preprocess.yaml \
    -m config/ESD-Chinese-Singing-MFA/model.yaml \
    -t config/ESD-Chinese-Singing-MFA/train.yaml
```

### 2. 批量合成

```bash
python synthesize_chinese_pinyin.py \
    --restore_step 50000 \
    --mode batch \
    --source preprocessed_data/ESD-Chinese-Singing-MFA/val.txt \
    -p config/ESD-Chinese-Singing-MFA/preprocess.yaml \
    -m config/ESD-Chinese-Singing-MFA/model.yaml \
    -t config/ESD-Chinese-Singing-MFA/train.yaml
```

### 3. 使用音素序列合成

```bash
python synthesize_chinese_pinyin.py \
    --restore_step 50000 \
    --mode single \
    --text "{ni hao shi jie}" \
    --speaker_id "0005" \
    --emotion "Sad" \
    -p config/ESD-Chinese-Singing-MFA/preprocess.yaml \
    -m config/ESD-Chinese-Singing-MFA/model.yaml \
    -t config/ESD-Chinese-Singing-MFA/train.yaml
```

## 参数说明

### 必需参数
- `--restore_step`: 要加载的模型检查点步数（例如：10000, 50000）
- `--mode`: 合成模式（single/batch）
- `-p`: 预处理配置文件路径
- `-m`: 模型配置文件路径  
- `-t`: 训练配置文件路径

### 单句模式参数
- `--text`: 要合成的中文文本或音素序列
- `--speaker_id`: 说话人ID（0001-0010）
- `--emotion`: 情感类型（Angry/Happy/Neutral/Sad/Surprise）

### 批量模式参数
- `--source`: 源文件路径（格式如train.txt或val.txt）

### 控制参数
- `--pitch_control`: 音调控制（0.5-2.0，默认1.0）
- `--energy_control`: 能量控制（0.5-2.0，默认1.0）
- `--duration_control`: 语速控制（0.5-2.0，默认1.0）

## 说话人和情感

### 可用说话人
- 0001-0010: 中文说话人

### 可用情感
- **Angry**: 愤怒（高唤醒度，低愉悦度）
- **Happy**: 开心（高唤醒度，高愉悦度）
- **Neutral**: 中立（中等唤醒度，中等愉悦度）
- **Sad**: 伤心（低唤醒度，低愉悦度）
- **Surprise**: 惊讶（高唤醒度，中等愉悦度）

## 输出文件

合成的音频文件将保存在配置文件中指定的结果目录：
```
output/result/ESD-Chinese-Singing-MFA/
```

文件命名格式：
- 单句模式：`{speaker_id}_{emotion}_{step}.wav`
- 批量模式：`{basename}_{step}.wav`

## 模型检查点

训练过程中的模型检查点保存在：
```
output/ckpt/ESD-Chinese-Singing-MFA/
```

可以使用不同步数的检查点进行推理，通常建议使用：
- 早期检查点（10000-30000步）：可能音质较差但训练快
- 中期检查点（50000-100000步）：平衡音质和训练时间
- 后期检查点（150000+步）：最佳音质但需要更长训练时间

## 故障排除

### 1. 找不到检查点文件
确保指定的`--restore_step`对应的检查点文件存在：
```bash
ls output/ckpt/ESD-Chinese-Singing-MFA/
```

### 2. 音素映射错误
如果遇到未知音素，脚本会自动使用填充符号，并显示警告信息。

### 3. 内存不足
- 减少批量大小
- 使用较小的模型配置
- 使用CPU推理（速度较慢）

### 4. 音质问题
- 尝试不同的检查点步数
- 调整控制参数（pitch_control, energy_control, duration_control）
- 检查训练是否充分收敛

## 示例脚本

创建一个简单的测试脚本：

```bash
#!/bin/bash
# test_synthesis.sh

STEP=50000
CONFIG_DIR="config/ESD-Chinese-Singing-MFA"

# 测试不同情感
for emotion in "Happy" "Sad" "Angry" "Surprise" "Neutral"; do
    python synthesize_chinese_pinyin.py \
        --restore_step $STEP \
        --mode single \
        --text "今天天气真好" \
        --speaker_id "0001" \
        --emotion $emotion \
        -p $CONFIG_DIR/preprocess.yaml \
        -m $CONFIG_DIR/model.yaml \
        -t $CONFIG_DIR/train.yaml
done
```

运行测试：
```bash
chmod +x test_synthesis.sh
./test_synthesis.sh
``` 