import json

with open('raw_data/ESD-Chinese/speaker_info.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

with open('raw_data/ESD-Chinese/speaker_info.txt', 'w', encoding='utf-8') as f:
    for speaker_id in data:
        f.write(f'{speaker_id}|M|25\n')  # 假设性别为M，年龄为25

print('成功创建 speaker_info.txt 文件') 