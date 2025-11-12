
import re, os
from typing import List
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    _stemmer = StemmerFactory().create_stemmer()
except Exception:
    _stemmer = None

DEFAULT_STOPWORDS = set([
    "dan","di","ke","dari","yang","untuk","pada","dengan","sebagai","ini","itu",
    "adalah","atau","serta","oleh","agar","kali","lebih","nya","saja","dapat","tersebut",
    "kami","anda","yg","tapi","sudah","belum","tidak","tdk"
])

def clean(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str) -> List[str]:
    return text.split()

def remove_stopwords(tokens, stopwords=None):
    if stopwords is None:
        stopwords = DEFAULT_STOPWORDS
    return [t for t in tokens if t not in stopwords and len(t)>1]

def stem_tokens(tokens):
    if _stemmer:
        return [_stemmer.stem(t) for t in tokens]
    suffixes = ("lah","kah","nya","ku","mu","kan","i","an","nya")
    out = []
    for t in tokens:
        for s in suffixes:
            if t.endswith(s) and len(t)-len(s) >= 3:
                t = t[:-len(s)]
                break
        out.append(t)
    return out

def preprocess_text(text):
    c = clean(text)
    toks = tokenize(c)
    toks = remove_stopwords(toks)
    toks = stem_tokens(toks)
    return toks

def process_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith('.txt'): continue
        with open(os.path.join(input_dir,fname),'r',encoding='utf-8') as f:
            txt = f.read()
        toks = preprocess_text(txt)
        out = ' '.join(toks)
        with open(os.path.join(output_dir,fname),'w',encoding='utf-8') as fout:
            fout.write(out)
        print('Processed', fname, '->', len(toks), 'tokens')
