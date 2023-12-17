import json
from nltk.tokenize import regexp_tokenize

mapping = []
with open('data/mapping.json', 'r', encoding="utf-8") as f:
    mapping = json.load(f).items()

mapping = sorted(mapping, key = lambda x: x[0], reverse=True)

def match(pattern: str, token: str) -> tuple[bool, str]:
    m = len(pattern)
    n = len(token)
    for i, _ in enumerate(token):
        if i == m:
            break
        elif pattern[i] != token[i]:
            return (False, token)
    return (n >= m, token[m:])

def cyrillize(token: str) -> str:
    translated = ""
    while len(token) > 0:
        success = False
        for (p, m) in mapping:
            success, new_token = match(p, token)
            if success:
                token = new_token
                translated += m
                break
        if not success:
            return token
    return translated

pattern = r'[^\s,\.,\!,\?]+'

def preprocess(comment: str) -> list[str]:
    """Cyrillizes and tokenizes the given string"""
    return [cyrillize(token.lower()) for token in regexp_tokenize(comment, pattern)]