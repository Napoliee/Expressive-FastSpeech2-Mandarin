import tgt
import numpy as np

def debug_alignment():
    tg = tgt.io.read_textgrid('preprocessed_data/ESD-Chinese/TextGrid/0010/0010_000897.TextGrid')
    tier = tg.get_tier_by_name('phones')
    
    print("=== TextGrid 原始音素 ===")
    for i, t in enumerate(tier._objects):
        print(f"{i}: '{t.text}' [{t.start_time:.3f}-{t.end_time:.3f}]")
    
    print(f"\n总音素数: {len(tier._objects)}")
    
    # 模拟get_alignment的处理逻辑
    sil_phones = ["sil", "sp", "spn"]
    phones = []
    durations = []
    start_time = 0
    end_time = 0
    end_idx = 0
    
    print("\n=== 处理过程 ===")
    for i, t in enumerate(tier._objects):
        s, e, p = t.start_time, t.end_time, t.text
        print(f"处理第{i}个音素: '{p}' [{s:.3f}-{e:.3f}]")
        
        # Trim leading silences
        if phones == []:
            if p in sil_phones:
                print(f"  跳过开头静音: {p}")
                continue
            else:
                start_time = s
                print(f"  设置开始时间: {s:.3f}")

        if p not in sil_phones:
            # For ordinary phones
            phones.append(p)
            end_time = e
            end_idx = len(phones)
            print(f"  添加非静音音素: {p}, end_idx={end_idx}")
        else:
            # For silent phones
            phones.append(p)
            print(f"  添加静音音素: {p}")

        durations.append(
            int(
                np.round(e * 22050 / 256)  # sampling_rate / hop_length
                - np.round(s * 22050 / 256)
            )
        )
        print(f"  时长: {durations[-1]}")

    print(f"\n修剪前: phones={len(phones)}, end_idx={end_idx}")
    print(f"phones: {phones}")
    
    # Trim tailing silences
    phones = phones[:end_idx]
    durations = durations[:end_idx]
    
    print(f"\n修剪后: phones={len(phones)}")
    print(f"最终phones: {phones}")
    print(f"最终durations: {durations}")
    print(f"durations总和: {sum(durations)}")

if __name__ == "__main__":
    debug_alignment() 