"""
IPA-based text processing for ESD-Chinese
"""

import re
import numpy as np
from text.symbols_ipa import symbols, _symbol_to_id, _id_to_symbol

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")

def text_to_sequence_ipa(text, cleaner_names=None):
    """
    Convert IPA phoneme text to sequence of IDs
    
    Args:
        text: IPA phoneme string like "{t w ej˥˩ ʂ ej˧˥ spn n a˥˩}"
        cleaner_names: ignored for IPA processing
    
    Returns:
        List of integers corresponding to the phonemes
    """
    sequence = []
    
    # Check for curly braces and extract phonemes
    if text.startswith('{') and text.endswith('}'):
        # Extract phonemes from curly braces
        phonemes = text[1:-1].split()
        sequence = _phonemes_to_sequence(phonemes)
    else:
        # Treat as space-separated phonemes
        phonemes = text.split()
        sequence = _phonemes_to_sequence(phonemes)
    
    return sequence

def _phonemes_to_sequence(phonemes):
    """Convert phoneme list to ID sequence"""
    sequence = []
    for phoneme in phonemes:
        # Add @ prefix for IPA phonemes
        ipa_symbol = "@" + phoneme
        if ipa_symbol in _symbol_to_id:
            sequence.append(_symbol_to_id[ipa_symbol])
        else:
            # Unknown phoneme, use a default
            print(f"Warning: Unknown phoneme '{phoneme}', using '@spn'")
            if "@spn" in _symbol_to_id:
                sequence.append(_symbol_to_id["@spn"])
            else:
                sequence.append(1)  # UNK token
    
    return sequence

def sequence_to_text_ipa(sequence):
    """Convert sequence back to text"""
    result = []
    for id in sequence:
        if id < len(symbols):
            symbol = symbols[id]
            if symbol.startswith('@'):
                result.append(symbol[1:])  # Remove @ prefix
            else:
                result.append(symbol)
    return ' '.join(result)
