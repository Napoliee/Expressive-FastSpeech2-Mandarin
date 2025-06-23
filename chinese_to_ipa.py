#!/usr/bin/env python3
"""
中文到IPA音素转换器
"""

import re
import random

def create_smart_chinese_to_ipa():
    """
    创建一个更智能的中文到IPA转换器
    基于我们MFA训练数据中实际出现的音素
    """
    
    # 从我们的训练数据中观察到的常见中文字符到IPA映射
    char_to_ipa = {
        # 基本字符
        '你': ['n', 'i˨˩˦'],
        '好': ['x', 'aw˨˩˦'], 
        '我': ['w', 'o˨˩˦'],
        '是': ['ʂ', 'ʐ̩˥˩'],
        '的': ['t', 'i˥˩'],
        '啊': ['a˥˩'],
        '小': ['ɕ', 'j', 'aw˨˩˦'],
        '样': ['j', 'a˥˩', 'ŋ'],
        '来': ['l', 'aj˧˥'],
        '自': ['ts', 'z̩˥˩'],
        '西': ['ɕ', 'i˥˩'],
        '部': ['p', 'u˥˩'],
        '草': ['tʰs', 'aw˨˩˦'],
        '原': ['ɥ', 'an˧˥'],
        
        # 常见字符扩展
        '世': ['ʂ', 'ʐ̩˥˩'],
        '界': ['tɕ', 'j', 'e˥˩'],
        '美': ['m', 'ej˨˩˦'],
        '丽': ['l', 'i˥˩'],
        '中': ['ʈʂ', 'oŋ˥˩'],
        '国': ['k', 'u̯o˥˩'],
        '今': ['tɕ', 'in˥˩'],
        '天': ['tʰ', 'j', 'an˥˩'],
        '明': ['m', 'iŋ˥˩'],
        '谢': ['ɕ', 'j', 'e˥˩'],
        '开': ['kʰ', 'aj˥˩'],
        '心': ['ɕ', 'in˥˩'],
        '生': ['ʂ', 'əŋ˥˩'],
        '日': ['ʐ̩˥˩'],
        '快': ['kʰ', 'uaj˥˩'],
        '乐': ['l', 'ə˥˩'],
        '语': ['y˨˩˦'],
        '音': ['in˥˩'],
        '合': ['x', 'ə˧˥'],
        '成': ['ʈʂʰ', 'əŋ˧˥'],
        '测': ['tʰs', 'ə˥˩'],
        '试': ['ʂ', 'ʐ̩˥˩'],
        
        # 数字
        '一': ['i˥˩'],
        '二': ['ʌ˧˥', 'ʐ̩˥˩'],
        '三': ['s', 'an˥˩'],
        '四': ['s', 'z̩˥˩'],
        '五': ['w', 'u˨˩˦'],
        '六': ['l', 'j', 'ow˥˩'],
        '七': ['tɕʰ', 'i˥˩'],
        '八': ['p', 'a˥˩'],
        '九': ['tɕ', 'j', 'ow˨˩˦'],
        '十': ['ʂ', 'ʐ̩˧˥'],
        
        # 标点符号处理
        '，': ['spn'],
        '。': ['spn'],
        '？': ['spn'],
        '！': ['spn'],
        '、': ['spn'],
        '：': ['spn'],
        '；': ['spn'],
    }
    
    return char_to_ipa

def chinese_text_to_ipa(text):
    """
    将中文文本转换为IPA音素序列
    """
    char_to_ipa = create_smart_chinese_to_ipa()
    
    # 清理文本，移除多余空格
    text = re.sub(r'\s+', '', text.strip())
    
    ipa_phones = []
    
    # 遍历每个字符
    for char in text:
        if char in char_to_ipa:
            # 找到映射的音素
            phones = char_to_ipa[char]
            ipa_phones.extend(phones)
        elif '\u4e00' <= char <= '\u9fff':  # 中文字符
            # 未知中文字符，使用一些通用音素
            fallback_phones = random.choice([
                ['ʂ', 'ʐ̩˥˩'],  # 类似"是"
                ['t', 'i˥˩'],   # 类似"的"
                ['x', 'aw˨˩˦'], # 类似"好"
                ['l', 'i˥˩'],   # 类似"丽"
            ])
            ipa_phones.extend(fallback_phones)
            print(f"⚠️  未知字符 '{char}' 使用回退音素: {fallback_phones}")
        else:
            # 非中文字符，跳过或当作静音
            if char.strip():  # 非空白字符
                ipa_phones.append('spn')
    
    # 如果没有音素，添加默认音素
    if not ipa_phones:
        ipa_phones = ['n', 'i˨˩˦', 'x', 'aw˨˩˦']  # 默认"你好"
    
    # 在音素间适当添加静音（模拟自然停顿）
    if len(ipa_phones) > 6:  # 长文本添加静音
        enhanced_phones = []
        for i, phone in enumerate(ipa_phones):
            enhanced_phones.append(phone)
            # 每隔几个音素添加短暂静音
            if i > 0 and (i + 1) % 4 == 0 and i < len(ipa_phones) - 1:
                enhanced_phones.append('spn')
        ipa_phones = enhanced_phones
    
    return ipa_phones

def test_ipa_conversion():
    """测试IPA转换"""
    test_texts = [
        "你好",
        "你好啊，我是小样，来自西部草原",
        "世界很美丽",
        "中国加油",
        "今天天气很好",
        "生日快乐",
        "语音合成测试",
    ]
    
    print("=== 中文到IPA转换测试 ===")
    for text in test_texts:
        ipa_phones = chinese_text_to_ipa(text)
        ipa_text = "{" + " ".join(ipa_phones) + "}"
        print(f"原文: {text}")
        print(f"IPA: {ipa_text}")
        print(f"音素数: {len(ipa_phones)}")
        print("-" * 40)

if __name__ == "__main__":
    test_ipa_conversion() 