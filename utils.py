import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

HINDI_STOP = set("है हैं था थे थी का की के को में से पर और या कि यह वह इस उस एक भी तो ही नहीं कर हो गया गई गए लिए साथ".split())
SCRIPTS = {"Hindi": (0x0900, 0x097F), "Bengali": (0x0980, 0x09FF), "Tamil": (0x0B80, 0x0BFF)}

def clean_text(text):
    text = str(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"<.*?>", " ", text)                          # HTML
    text = re.sub(r"\d+", " ", text)                            # numbers
    text = re.sub(r"[\u0964\u0965]", " ", text)                 # Hindi danda
    text = re.sub(r"[^\w\s\u0900-\u097F\u0980-\u09FF\u0B80-\u0BFF]", " ", text)  # punctuation
    text = re.sub(r"\s+", " ", text).strip().lower()
    tokens = [t for t in text.split()
              if t not in ENGLISH_STOP_WORDS and t not in HINDI_STOP]   # stopwords
    return " ".join(tokens)

def detect_language(text):
    counts = {l: 0 for l in SCRIPTS}
    counts["English"] = 0
    for ch in text:
        for lang, (lo, hi) in SCRIPTS.items():
            if lo <= ord(ch) <= hi:
                counts[lang] += 1
                break
        else:
            if ch.isascii() and ch.isalpha():
                counts["English"] += 1
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else "Unknown"