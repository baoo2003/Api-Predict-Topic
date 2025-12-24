import unicodedata
from underthesea import text_normalize
from bs4 import BeautifulSoup
import re
from vncorenlp import VnCoreNLP

def remove_html_tags(text: str) -> str:
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def clean_text_vn(text: str) -> str:
    text = re.sub(r"[^\w\s_]", " ", text, flags=re.UNICODE)
    return ' '.join(text.split())


def to_lower(text: str) -> str:
    return text.lower()

def standardize_unicode(text: str) -> str:
    return unicodedata.normalize('NFC', text)

def normalize_text(text: str) -> str:
    text = text_normalize(text)
    return text

def word_tokenize_vn(text: str) -> str:
    annotator = VnCoreNLP("vncorenlp/VnCoreNLP-1.2.jar", annotators="wseg")
    return ' '.join(annotator.tokenize(text)[0])

def preprocess_text(text: str, rw=False) -> str:
    text = remove_html_tags(text)
    text = to_lower(text)
    text = standardize_unicode(text)
    text = normalize_text(text)
    text = word_tokenize_vn(text)
    text = clean_text_vn(text)
    if rw:
        text = remove_stopwords(text, load_vietnamese_stopwords())
    return text

def has_invalid_char(text):
    return bool(re.search(r"[^\w\s_]", text))

def has_extra_space(text):
    return bool(re.search(r"\s{2,}", text)) or text != text.strip()

def load_vietnamese_stopwords(path="vietnamese-stopwords-dash.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

def remove_stopwords(text: str, stopwords: set) -> str:
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords]
    return " ".join(tokens)