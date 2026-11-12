import re
import unicodedata


SLANG_DICT = {
    'yg': 'yang', 'dgn': 'dengan', 'utk': 'untuk', 'krn': 'karena',
    'tdk': 'tidak', 'sdh': 'sudah', 'blm': 'belum', 'bgt': 'banget',
    'ga': 'tidak', 'gak': 'tidak', 'ngga': 'tidak', 'nggak': 'tidak',
    'tp': 'tapi', 'ttg': 'tentang', 'jg': 'juga', 'spy': 'supaya',
    'klo': 'kalau', 'klu': 'kalau', 'kl': 'kalau', 'sm': 'sama',
    'dr': 'dari', 'pd': 'pada', 'bs': 'bisa', 'bisa': 'bisa',
    'hrs': 'harus', 'emg': 'memang', 'aja': 'saja', 'aj': 'saja',
    'sy': 'saya', 'gw': 'saya', 'gue': 'saya', 'w': 'saya',
    'lu': 'kamu', 'lo': 'kamu', 'km': 'kamu',
    'bkn': 'bukan', 'jd': 'jadi', 'krg': 'kurang', 'lbh': 'lebih',
    'msh': 'masih', 'kyk': 'kayak', 'sprt': 'seperti', 'spt': 'seperti',
    'mkn': 'makan', 'msk': 'masuk', 'dtg': 'datang', 'plg': 'pulang',
    'trs': 'terus', 'udh': 'sudah', 'udah': 'sudah', 'dah': 'sudah',
    'nih': 'ini', 'tuh': 'itu', 'sih': '', 'deh': '', 'dong': '',
    'loh': '', 'lah': '', 'kan': 'kan',
}

NEGATIONS = {'tidak', 'bukan', 'jangan', 'belum', 'tanpa', 'tiada', 'tak'}


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize('NFKC', text)


def normalize_repeated_chars(text: str) -> str:
    return re.sub(r'(.)\1{2,}', r'\1\1', text)


def normalize_slang(tokens: list[str]) -> list[str]:
    return [SLANG_DICT.get(t, t) for t in tokens if SLANG_DICT.get(t, t) != '']


def tokenize(text: str) -> list[str]:
    text = text.lower().strip()
    text = normalize_unicode(text)
    text = normalize_repeated_chars(text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = normalize_slang(tokens)
    return tokens


def preprocess_aspect_input(aspect: str, review_text: str) -> list[str]:
    """Tokenize [ASPEK] {aspect} [ULASAN] {review_text} for Exp A."""
    prefix = ['[ASPEK]'] + aspect.lower().split() + ['[ULASAN]']
    body   = tokenize(review_text)
    return prefix + body


def preprocess_overall_input(review_text: str) -> list[str]:
    """Tokenize raw review_text for Exp B (overall sentiment)."""
    return tokenize(review_text)
