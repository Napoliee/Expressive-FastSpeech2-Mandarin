#!/usr/bin/env python3
"""
监控语音生成进度
"""

import os
import time
import glob

def main():
    output_dir = "output/result/ESD-Chinese-Singing-MFA/"
    target_count = 25  # 目标文件数量
    
    print("=== 监控语音生成进度 ===")
    print(f"目标文件数量: {target_count}")
    print(f"输出目录: {output_dir}")
    print()
    
    while True:
        if os.path.exists(output_dir):
            wav_files = glob.glob(os.path.join(output_dir, "*.wav"))
            current_count = len(wav_files)
            
            print(f"\r当前进度: {current_count}/{target_count} ({current_count/target_count*100:.1f}%)", end="", flush=True)
            
            if current_count >= target_count:
                print(f"\n\n🎉 生成完成！总共生成了 {current_count} 个音频文件")
                
                # 按情感分组显示
                emotions = ["Happy", "Sad", "Angry", "Surprise", "Neutral"]
                print("\n文件列表:")
                for emotion in emotions:
                    emotion_files = [f for f in wav_files if os.path.basename(f).startswith(emotion)]
                    if emotion_files:
                        print(f"  {emotion}: {len(emotion_files)} 个文件")
                        for f in sorted(emotion_files):
                            basename = os.path.basename(f)
                            file_size = os.path.getsize(f) / 1024  # KB
                            print(f"    {basename} ({file_size:.1f} KB)")
                break
        else:
            print(f"\r等待输出目录创建...", end="", flush=True)
        
        time.sleep(2)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n监控已停止") 