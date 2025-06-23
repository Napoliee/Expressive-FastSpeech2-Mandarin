""" Pinyin-based symbols for ESD-Chinese dataset """

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# Pinyin phonemes from MFA alignment (actual phone symbols used)
_pinyin_phonemes = [
    'a', 'ai', 'ao', 'b', 'c', 'ch', 'd', 'e', 'ei', 'er', 'f', 'g', 'h', 'i', 
    'ia', 'iao', 'ie', 'iu', 'j', 'k', 'l', 'm', 'n', 'ng', 'o', 'ou', 'p', 'q', 
    'r', 's', 'sh', 'spn', 't', 'u', 'ua', 'uai', 'ue', 'ui', 'uo', 'w', 'x', 'y', 'z', 'zh'
]

# Export all symbols
symbols = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    + list(_letters)
    + _pinyin_phonemes
)

# Create symbol to ID mapping
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)} 