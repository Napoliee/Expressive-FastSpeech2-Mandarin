""" IPA-based symbols for ESD-Chinese dataset """

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# IPA phonemes from MFA alignment (with @ prefix for uniqueness)
_ipa_phonemes = ['@aj˥˩', '@aj˧˥', '@aj˨˩˦', '@aj˩', '@aw˥˩', '@aw˧˥', '@aw˨˩˦', '@a˥˩', '@a˧˥', '@a˨˩˦', '@a˩', '@ej˥˩', '@ej˧˥', '@ej˨˩˦', '@e˥˩', '@e˧˥', '@e˨˩˦', '@e˩', '@f', '@i˥˩', '@i˧˥', '@i˨˩˦', '@i˩', '@j', '@k', '@kʰ', '@l', '@m', '@n', '@ow˥˩', '@ow˧˥', '@ow˨˩˦', '@ow˩', '@o˥˩', '@o˧˥', '@o˨˩˦', '@p', '@pʰ', '@s', '@spn', '@t', '@ts', '@tsʰ', '@tɕ', '@tɕʰ', '@tʰ', '@u˥˩', '@u˧˥', '@u˨˩˦', '@w', '@x', '@y˥˩', '@y˧˥', '@y˨˩˦', '@z̩˥˩', '@z̩˨˩˦', '@z̩˩', '@ŋ', '@ɕ', '@ə˥˩', '@ə˧˥', '@ə˨˩˦', '@ə˩', '@ɥ', '@ɻ', '@ʂ', '@ʈʂ', '@ʈʂʰ', '@ʐ', '@ʐ̩˥˩', '@ʐ̩˧˥', '@ʐ̩˨˩˦', '@ʐ̩˩', '@ʔ']

# Export all symbols
symbols = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    + list(_letters)
    + _ipa_phonemes
)

# Create symbol to ID mapping
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}
