import os
import shutil
from tqdm import tqdm

clips_dir = 'raw_data/ESD-Chinese/clips'
base_dir = 'raw_data/ESD-Chinese'

# 获取所有wav文件
wav_files = [f for f in os.listdir(clips_dir) if f.endswith('.wav')]

# 按说话人分组
speakers = set()
for wav_file in wav_files:
    speaker_id = wav_file.split('_')[0]
    speakers.add(speaker_id)

print(f"找到 {len(speakers)} 个说话人: {sorted(speakers)}")

# 为每个说话人创建目录并移动文件
for speaker in sorted(speakers):
    speaker_dir = os.path.join(base_dir, speaker)
    os.makedirs(speaker_dir, exist_ok=True)
    
    # 移动该说话人的所有文件
    speaker_files = [f for f in wav_files if f.startswith(speaker + '_')]
    
    print(f"处理说话人 {speaker}: {len(speaker_files)} 个文件")
    
    for wav_file in tqdm(speaker_files, desc=f"Moving {speaker}"):
        # 移动wav文件
        src_wav = os.path.join(clips_dir, wav_file)
        dst_wav = os.path.join(speaker_dir, wav_file)
        shutil.move(src_wav, dst_wav)
        
        # 创建对应的lab文件（如果不存在的话）
        lab_file = wav_file.replace('.wav', '.lab')
        lab_path = os.path.join(speaker_dir, lab_file)
        
        if not os.path.exists(lab_path):
            # 从filelist.txt中查找对应的文本
            with open('raw_data/ESD-Chinese/filelist.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith(f'clips/{wav_file}|'):
                        text = line.split('|')[1]
                        with open(lab_path, 'w', encoding='utf-8') as lab_f:
                            lab_f.write(text)
                        break

# 删除空的clips目录
if os.path.exists(clips_dir) and not os.listdir(clips_dir):
    os.rmdir(clips_dir)
    
print("文件重新组织完成！") 