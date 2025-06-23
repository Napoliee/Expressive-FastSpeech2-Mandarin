import os
import shutil
from tqdm import tqdm

def reorganize_data():
    # 定义源目录和目标目录
    src_dir = "/home/taopeng/shuiwen/Expressive-FastSpeech2-continuous/raw_data/ESD-Chinese-Singing-MFA"
    dst_dir = "/home/taopeng/shuiwen/Expressive-FastSpeech2-continuous/raw_data/ESD-Chinese-Pinyin"

    # 获取所有文件
    files = os.listdir(src_dir)
    wav_files = [f for f in files if f.endswith(".wav")]
    lab_files = [f for f in files if f.endswith(".lab")]

    # 按说话人ID分组
    speaker_files = {}
    for wav_file in wav_files:
        speaker_id = wav_file.split("_")[0]  # 例如从"0001_000001.wav"中提取"0001"
        if speaker_id not in speaker_files:
            speaker_files[speaker_id] = {"wav": [], "lab": []}
        speaker_files[speaker_id]["wav"].append(wav_file)

    for lab_file in lab_files:
        speaker_id = lab_file.split("_")[0]
        if speaker_id in speaker_files:  # 只处理有对应wav文件的lab文件
            speaker_files[speaker_id]["lab"].append(lab_file)

    # 在源目录和目标目录中创建说话人子文件夹
    for speaker_id in tqdm(sorted(speaker_files.keys()), desc="处理说话人"):
        # 在源目录中创建说话人目录
        src_speaker_dir = os.path.join(src_dir, speaker_id)
        os.makedirs(src_speaker_dir, exist_ok=True)

        # 在目标目录中创建说话人目录
        dst_speaker_dir = os.path.join(dst_dir, speaker_id)
        os.makedirs(dst_speaker_dir, exist_ok=True)

        # 移动wav文件到源目录的说话人子文件夹
        for wav_file in speaker_files[speaker_id]["wav"]:
            src_wav = os.path.join(src_dir, wav_file)
            new_src_wav = os.path.join(src_speaker_dir, wav_file)
            dst_wav = os.path.join(dst_speaker_dir, wav_file)
            
            # 移动到源目录的子文件夹
            if os.path.exists(src_wav):
                shutil.move(src_wav, new_src_wav)
            
            # 复制到目标目录
            if os.path.exists(new_src_wav):
                shutil.copy2(new_src_wav, dst_wav)

        # 移动lab文件到源目录的说话人子文件夹
        for lab_file in speaker_files[speaker_id]["lab"]:
            src_lab = os.path.join(src_dir, lab_file)
            new_src_lab = os.path.join(src_speaker_dir, lab_file)
            
            # 移动到源目录的子文件夹
            if os.path.exists(src_lab):
                shutil.move(src_lab, new_src_lab)

    # 创建filelist.txt
    filelist_content = []
    for speaker_id in sorted(speaker_files.keys()):
        src_speaker_dir = os.path.join(src_dir, speaker_id)
        for wav_file in sorted(speaker_files[speaker_id]["wav"]):
            lab_file = wav_file.replace(".wav", ".lab")
            lab_path = os.path.join(src_speaker_dir, lab_file)
            
            if os.path.exists(lab_path):
                with open(lab_path, "r") as f:
                    pinyin_text = f.read().strip()
                
                # 构建filelist条目
                wav_rel_path = os.path.join(speaker_id, wav_file)  # 相对路径
                filelist_content.append(f"{wav_rel_path}|{pinyin_text}|{speaker_id}|default")

    # 保存filelist.txt
    filelist_path = os.path.join(dst_dir, "filelist.txt")
    with open(filelist_path, "w") as f:
        f.write("\n".join(filelist_content))

    print(f"\n✅ 完成！")
    print(f"总文件数: {len(wav_files)}")
    print(f"说话人数: {len(speaker_files)}")
    for speaker_id in sorted(speaker_files.keys()):
        print(f"  - {speaker_id}: {len(speaker_files[speaker_id]['wav'])}个wav文件, {len(speaker_files[speaker_id]['lab'])}个lab文件")
    print(f"源目录: {src_dir}")
    print(f"目标目录: {dst_dir}")
    print(f"Filelist: {filelist_path}")

if __name__ == "__main__":
    reorganize_data() 