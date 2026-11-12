"""
Lexicon-based ABSA menggunakan InSet + hotel domain extension.

InSet diunduh otomatis dari GitHub jika belum tersedia:
  https://github.com/fajri91/InSet
"""

import os
import re
import csv
import requests
import pandas as pd
import yaml
from pathlib import Path
from src.preprocessing.text_preprocessing import preprocess_for_lexicon


INSET_POS_URL = 'https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv'
INSET_NEG_URL = 'https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv'


def download_inset(save_path: str = 'lexicons/inset_combined.csv') -> None:
    if Path(save_path).exists():
        return
    print('Mengunduh InSet dari GitHub...')
    rows = []
    for url, polarity in [(INSET_POS_URL, 1), (INSET_NEG_URL, -1)]:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        for line in resp.text.strip().split('\n'):
            parts = line.split('\t')
            if len(parts) >= 2:
                word = parts[0].strip().lower()
                try:
                    weight = float(parts[1].strip()) * polarity
                except ValueError:
                    weight = float(polarity)
                rows.append({'token': word, 'polarity_score': weight, 'source': 'InSet'})
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f'InSet disimpan ke {save_path} ({len(df)} entri)')


def load_lexicon(
    base_path:   str = 'lexicons/inset_combined.csv',
    domain_path: str = 'lexicons/hotel_domain_lexicon.csv',
) -> dict:
    """Gabungkan InSet + domain extension. Domain override InSet jika ada konflik."""
    download_inset(base_path)
    lexicon = {}
    for path in [base_path, domain_path]:
        if not Path(path).exists():
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            token = str(row['token']).lower().strip()
            score = float(row['polarity_score'])
            if score != 0:
                lexicon[token] = score
    return lexicon


def load_aspect_terms(path: str = 'lexicons/aspect_terms.csv') -> dict:
    df = pd.read_csv(path)
    result = {}
    for _, row in df.iterrows():
        terms = [t.strip() for t in str(row['terms_id']).split(',')]
        result[row['aspect']] = terms
    return result


def load_config(path: str = 'configs/lexicon.yaml') -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class LexiconABSA:
    def __init__(
        self,
        lexicon_base:   str = 'lexicons/inset_combined.csv',
        lexicon_domain: str = 'lexicons/hotel_domain_lexicon.csv',
        aspect_terms:   str = 'lexicons/aspect_terms.csv',
        config_path:    str = 'configs/lexicon.yaml',
    ):
        self.config       = load_config(config_path)
        self.lexicon      = load_lexicon(lexicon_base, lexicon_domain)
        self.aspect_terms = load_aspect_terms(aspect_terms)
        rules = self.config['rules']
        self.negation_window    = rules['negation_window']
        self.negation_words     = set(rules['negation_words'])
        self.intensifiers       = rules['intensifier_words']
        self.diminishers        = rules['diminisher_words']
        self.contrast_words     = set(rules['contrast_words'])
        self.no_match_policy    = self.config['lexicon']['no_match_policy']
        thresholds = self.config['thresholds']
        self.pos_min = thresholds['positive_min']
        self.neg_max = thresholds['negative_max']

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def _get_aspect_context(self, tokens: list[str], aspect: str) -> list[str]:
        """
        Ambil konteks lokal di sekitar istilah aspek.
        Jika tidak ada istilah aspek yang cocok → gunakan seluruh teks.
        """
        terms = self.aspect_terms.get(aspect, [])
        positions = []
        for i, tok in enumerate(tokens):
            for term in terms:
                term_tokens = term.split()
                if len(term_tokens) == 1 and tok == term:
                    positions.append(i)
                elif tokens[i:i+len(term_tokens)] == term_tokens:
                    positions.append(i)

        if not positions:
            return tokens

        # ambil window ±5 token dari tiap posisi aspek
        window = 5
        indices = set()
        for pos in positions:
            indices.update(range(max(0, pos - window), min(len(tokens), pos + window + 1)))
        return [tokens[i] for i in sorted(indices)]

    def _score_tokens(self, tokens: list[str]) -> float:
        score = 0.0
        matched = False
        i = 0
        while i < len(tokens):
            tok = tokens[i]

            # cek frasa multi-kata (2-token)
            bigram = f"{tok} {tokens[i+1]}" if i + 1 < len(tokens) else None
            if bigram and bigram in self.lexicon:
                word_score = self.lexicon[bigram]
                matched = True
                # cek negasi dalam window sebelum frasa
                neg = any(tokens[max(0, i-self.negation_window):i][j] in self.negation_words
                          for j in range(len(tokens[max(0, i-self.negation_window):i])))
                if neg:
                    word_score *= -1
                score += word_score
                i += 2
                continue

            # token tunggal
            if tok in self.lexicon:
                word_score = self.lexicon[tok]
                matched = True
                # intensifier/diminisher sebelum token
                for j in range(max(0, i - 2), i):
                    t = tokens[j]
                    if t in self.intensifiers:
                        word_score *= self.intensifiers[t]
                    elif t in self.diminishers:
                        word_score *= self.diminishers[t]
                # negasi
                neg = any(tokens[max(0, i-self.negation_window):i][k] in self.negation_words
                          for k in range(len(tokens[max(0, i-self.negation_window):i])))
                if neg:
                    word_score *= -1
                score += word_score
            i += 1
        return score, matched

    def predict_one(self, review_text: str, aspect: str) -> str:
        text = preprocess_for_lexicon(review_text)
        tokens = self._tokenize(text)
        context = self._get_aspect_context(tokens, aspect)
        score, matched = self._score_tokens(context)

        if not matched:
            return self.no_match_policy

        if score > self.pos_min:
            return 'positive'
        elif score < self.neg_max:
            return 'negative'
        else:
            return 'neutral'

    def predict(self, df: pd.DataFrame) -> list[str]:
        return [
            self.predict_one(row['review_text'], row['aspect'])
            for _, row in df.iterrows()
        ]
