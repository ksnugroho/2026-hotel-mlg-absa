"""
Normalisasi teks. Negasi dan kata evaluatif dipertahankan.
"""

import re
import unicodedata


SLANG_DICT = {
    'gak': 'tidak', 'ga': 'tidak', 'nggak': 'tidak', 'ngga': 'tidak',
    'gk': 'tidak', 'tdk': 'tidak', 'tdk.': 'tidak', "ga'": 'tidak',
    'bgt': 'banget', 'bngt': 'banget', 'bget': 'banget',
    'udh': 'sudah', 'udah': 'sudah', 'sdh': 'sudah',
    'blm': 'belum', 'blom': 'belum',
    'krn': 'karena', 'karna': 'karena',
    'tapi': 'tetapi', 'tp': 'tetapi',
    'yg': 'yang', 'yng': 'yang',
    'utk': 'untuk', 'buat': 'untuk',
    'dgn': 'dengan', 'dg': 'dengan',
    'sy': 'saya', 'aku': 'saya',
    'hrs': 'harus',
    'kl': 'kalau', 'klo': 'kalau', 'klw': 'kalau',
    'jg': 'juga', 'juga': 'juga',
    'sm': 'sama',
    'bs': 'bisa', 'bsa': 'bisa',
    'emg': 'memang', 'emang': 'memang',
    'msh': 'masih',
    'lg': 'lagi',
    'org': 'orang',
    'sdg': 'sedang',
    'stlh': 'setelah',
    'sblm': 'sebelum',
    'krg': 'kurang',
    'lbh': 'lebih',
    'tmpt': 'tempat',
    'kamarnya': 'kamar',
    'hotelnya': 'hotel',
    'staffnya': 'staff',
}


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize('NFC', text)


def remove_excess_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def normalize_url_mention(text: str) -> str:
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    return text


def normalize_repeated_chars(text: str, max_repeat: int = 2) -> str:
    return re.sub(r'(.)\1{' + str(max_repeat) + r',}', r'\1' * max_repeat, text)


def normalize_punctuation(text: str) -> str:
    text = re.sub(r'[^\w\s\.,!?\-]', ' ', text)
    return text


def apply_slang_normalization(text: str) -> str:
    tokens = text.split()
    normalized = [SLANG_DICT.get(t.lower(), t) for t in tokens]
    return ' '.join(normalized)


def preprocess_for_lexicon(text: str) -> str:
    """Preprocessing ringan untuk pendekatan leksikon."""
    text = normalize_unicode(text)
    text = normalize_url_mention(text)
    text = normalize_repeated_chars(text)
    text = text.lower()
    text = apply_slang_normalization(text)
    text = remove_excess_whitespace(text)
    return text


def preprocess_for_svm(text: str) -> str:
    """Preprocessing untuk TF-IDF + SVM."""
    text = normalize_unicode(text)
    text = normalize_url_mention(text)
    text = normalize_repeated_chars(text)
    text = text.lower()
    text = apply_slang_normalization(text)
    text = normalize_punctuation(text)
    text = remove_excess_whitespace(text)
    return text


def preprocess_for_indobert(text: str) -> str:
    """
    Preprocessing minimal untuk IndoBERT.
    Tidak dilakukan stemming, stopword removal, atau lowercase agresif.
    Tokenizer IndoBERT menangani normalisasi subword.
    """
    text = normalize_unicode(text)
    text = normalize_url_mention(text)
    text = normalize_repeated_chars(text)
    text = remove_excess_whitespace(text)
    return text
