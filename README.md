# Expressive-FastSpeech2-Mandarin-Emotional-Speech-Synthesis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An implementation of Expressive-FastSpeech2 for Mandarin emotional speech synthesis with multi-dimensional emotion control. This project enables controllable synthesis of Mandarin speech with discrete emotion categories (Angry, Happy, Neutral, Sad, Surprise) and continuous arousal/valence dimensions.

## 🎯 Features

- **Multi-Dimensional Emotion Control**: Supports both categorical emotions and continuous arousal/valence control
- **Mandarin Chinese Support**: Native pinyin phoneme processing with MFA alignment
- **10-Speaker Model**: Trained on ESD Mandarin subset with balanced speaker representation
- **High-Quality Synthesis**: 22.05kHz audio generation with HiFi-GAN vocoder
- **Comprehensive Pipeline**: Complete data preprocessing, training, and synthesis workflow

## 🔧 Installation

### Requirements
- Python 3.8+
- CUDA 11.0+ (recommended)
- 16GB+ RAM
- 50GB+ storage space

### Setup Environment

```bash
# Clone repository
git clone https://github.com/your-repo/Expressive-FastSpeech2-Mandarin.git
cd Expressive-FastSpeech2-Mandarin

# Install dependencies
pip install -r requirements.txt

# Install additional packages
pip install pypinyin textgrid tgt

# Install Montreal Forced Aligner
conda install -c conda-forge montreal-forced-aligner

# Download MFA models
mfa download acoustic mandarin_pinyin
mfa download dictionary mandarin_pinyin
```

## 📊 Dataset Preparation

### Download ESD Dataset

```bash
# Download ESD dataset
wget https://github.com/HLTSingapore/Emotional-Speech-Data/releases/download/v1.0/ESD.zip
unzip ESD.zip
mv ESD raw_data/ESD-Original
```

### Data Preprocessing

```bash
# 1. Reorganize ESD Chinese data
python prepare_esd_data_fixed.py

# 2. Convert Chinese text to pinyin
python convert_to_pinyin_fixed.py

# 3. Run MFA alignment
python train_pinyin_pipeline.py

# 4. Extract features for training
python preprocess.py config/ESD-Chinese-Singing-MFA/preprocess.yaml
```

## 🚀 Quick Start

### Download Pre-trained Model

```bash
# Download pre-trained checkpoint (900k steps)
# Place in: output/ckpt/ESD-Chinese-Singing-MFA/900000.pth.tar
```

### Single Sentence Synthesis

```bash
python synthesize_chinese_pinyin.py \
    --restore_step 900000 \
    --mode single \
    --text "今天天气真好" \
    --speaker_id "0001" \
    --emotion "Happy" \
    --pitch_control 1.0 \
    --energy_control 1.0 \
    --duration_control 1.0 \
    -p config/ESD-Chinese-Singing-MFA/preprocess.yaml \
    -m config/ESD-Chinese-Singing-MFA/model.yaml \
    -t config/ESD-Chinese-Singing-MFA/train.yaml
```

### Batch Synthesis

```bash
python synthesize_chinese_pinyin.py \
    --restore_step 900000 \
    --mode batch \
    --source preprocessed_data/ESD-Chinese-Singing-MFA/test.txt \
    -p config/ESD-Chinese-Singing-MFA/preprocess.yaml \
    -m config/ESD-Chinese-Singing-MFA/model.yaml \
    -t config/ESD-Chinese-Singing-MFA/train.yaml
```

## 🎛️ Emotion Control

### Supported Emotions

| Emotion | Arousal | Valence | Description |
|---------|---------|---------|-------------|
| **Angry** | 0.9 | 0.1 | High activation, negative valence |
| **Happy** | 0.8 | 0.8 | High activation, positive valence |
| **Neutral** | 0.5 | 0.5 | Medium activation, neutral valence |
| **Sad** | 0.3 | 0.2 | Low activation, negative valence |
| **Surprise** | 0.8 | 0.6 | High activation, medium-positive valence |

### Supported Speakers

| Speaker Range | Gender | Count | Usage |
|---------------|--------|-------|-------|
| 0001-0005 | Male | 5 | `--speaker_id "0001"` |
| 0006-0010 | Female | 5 | `--speaker_id "0006"` |

### Control Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `--pitch_control` | 0.5-2.0 | 1.0 | Pitch adjustment |
| `--energy_control` | 0.5-2.0 | 1.0 | Volume/energy control |
| `--duration_control` | 0.5-2.0 | 1.0 | Speech rate (lower = faster) |

## 🔬 Training from Scratch

### Configuration

Key training parameters from `config/ESD-Chinese-Singing-MFA/train.yaml`:

```yaml
optimizer:
  batch_size: 4
  betas: [0.9, 0.98]
  eps: 0.000000001
  warm_up_step: 4000
  anneal_steps: [300000, 400000, 500000]
  anneal_rate: 0.3

step:
  total_step: 900000
  save_step: 100000
  val_step: 1000
  log_step: 100
```

### Start Training

```bash
python train.py \
    -p config/ESD-Chinese-Singing-MFA/preprocess.yaml \
    -m config/ESD-Chinese-Singing-MFA/model.yaml \
    -t config/ESD-Chinese-Singing-MFA/train.yaml
```

### Multi-GPU Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
    -p config/ESD-Chinese-Singing-MFA/preprocess.yaml \
    -m config/ESD-Chinese-Singing-MFA/model.yaml \
    -t config/ESD-Chinese-Singing-MFA/train.yaml
```

### Monitor Training

```bash
# View training logs
tail -f output/log/ESD-Chinese-Singing-MFA/train/log.txt

# Launch TensorBoard
tensorboard --logdir output/log/ESD-Chinese-Singing-MFA/
```

## 📁 Project Structure

```
Expressive-FastSpeech2-Mandarin/
├── config/
│   └── ESD-Chinese-Singing-MFA/
│       ├── preprocess.yaml      # Preprocessing configuration
│       ├── model.yaml           # Model architecture settings
│       └── train.yaml           # Training parameters
├── raw_data/
│   ├── ESD-Original/            # Original ESD dataset
│   └── ESD-Chinese-Singing-MFA/ # Processed dataset
├── preprocessed_data/
│   └── ESD-Chinese-Singing-MFA/ # Extracted features
├── output/
│   ├── ckpt/                    # Model checkpoints
│   ├── log/                     # Training logs
│   └── result/                  # Synthesis results
├── model/
│   ├── fastspeech2.py          # Main model architecture
│   ├── modules.py              # Model components
│   └── loss.py                 # Loss functions
├── text/
│   └── symbols_pinyin.py       # Pinyin phoneme symbols
├── dataset_chinese.py          # Chinese dataset loader
├── synthesize_chinese_pinyin.py # Synthesis script
├── train.py                    # Training script
└── preprocess.py              # Preprocessing script
```

## 🎵 Audio Samples

Generated audio samples are saved in `output/result/ESD-Chinese-Singing-MFA/` with the following naming convention:

- **Single synthesis**: `{speaker_id}_{emotion}_{custom_name}.wav`
- **Batch synthesis**: `{original_filename}_{tag}.wav`

## 📈 Model Performance

### Training Statistics

| Metric | Value |
|--------|-------|
| **Dataset Size** | ~17,500 utterances (10 speakers × 350 utterances × 5 emotions) |
| **Training Steps** | 900,000 steps (~257 epochs) |
| **Training Time** | ~72 hours on GTX 1080 Ti |
| **Model Parameters** | ~28M parameters |
| **Final Loss** | Total Loss ~1.2, Mel Loss ~0.4 |

### Audio Quality

| Specification | Value |
|---------------|-------|
| **Sampling Rate** | 22.05 kHz |
| **Mel Channels** | 80 |
| **F0 Range** | 50-500 Hz |
| **Vocoder** | HiFi-GAN Universal |

## 🔧 Technical Details

### Model Architecture

```yaml
transformer:
  encoder_layer: 4      # 4-layer encoder
  encoder_head: 2       # 2 attention heads
  encoder_hidden: 256   # 256 hidden dimensions
  decoder_layer: 6      # 6-layer decoder
  decoder_head: 2
  decoder_hidden: 256

multi_speaker: True     # Multi-speaker support
multi_emotion: True     # Multi-emotion support
```

### Emotion Embedding

| Component | Dimensions | Description |
|-----------|------------|-------------|
| **Emotion Embedding** | 128 | Categorical emotion representation |
| **Arousal Embedding** | 64 | Activation level control |
| **Valence Embedding** | 64 | Pleasantness control |
| **Total Representation** | 256 | Combined emotion vector |

### Feature Processing

- **Phoneme Vocabulary**: 117 symbols (44 pinyin + punctuation + special tokens)
- **Pitch/Energy**: Phoneme-level features with 256-bin quantization
- **Duration**: Log-transformed with MSE loss

## 🛠️ Troubleshooting

### Common Issues

#### 1. MFA Permission Error
```bash
sudo chown -R $USER:$USER ~/.local/share/montreal-forced-aligner/
chmod -R 755 ~/.local/share/montreal-forced-aligner/
```

#### 2. CUDA Out of Memory
- Reduce batch size in `train.yaml`
- Use gradient accumulation
- Try single GPU training

#### 3. Phoneme Mapping Error
```bash
python -c "from text.symbols_pinyin import symbols; print(len(symbols))"
```

## 🙏 Acknowledgments

- [Expressive-FastSpeech2](https://github.com/keonlee9420/Expressive-FastSpeech2) by Keon Lee
- [ESD Dataset](https://github.com/HLTSingapore/Emotional-Speech-Data) by Zhou et al.
- [Montreal Forced Aligner](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner)
- [HiFi-GAN](https://github.com/jik876/hifi-gan) vocoder

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
