#!/usr/bin/env python3
"""
测试具体的问题音素
"""

from text.symbols_ipa import _symbol_to_id

def test_specific_phonemes():
    """测试推理中出现警告的音素"""
    
    problem_phonemes = ['iŋ˥˩', 'tʂʰ', 'ən˧˥', 'tj', 'an˨˩˦']
    
    print("=== 测试问题音素 ===")
    
    for phoneme in problem_phonemes:
        ipa_symbol = '@' + phoneme
        if ipa_symbol in _symbol_to_id:
            print(f'✅ {phoneme} -> {ipa_symbol} (ID: {_symbol_to_id[ipa_symbol]})')
        else:
            print(f'❌ {phoneme} -> {ipa_symbol} (NOT FOUND)')
            
            # 寻找相似的音素
            similar_phonemes = []
            for symbol in _symbol_to_id.keys():
                if symbol.startswith('@') and phoneme[:2] in symbol:
                    similar_phonemes.append(symbol)
            
            if similar_phonemes:
                print(f'   相似音素: {similar_phonemes[:5]}')

if __name__ == "__main__":
    test_specific_phonemes() 