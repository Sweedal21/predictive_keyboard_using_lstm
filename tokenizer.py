# import nltk
# nltk.download('punkt')
from nltk.tokenize import word_tokenize
from collections import Counter

def tokenize_text(file_path):
    with open(file_path, "r", encoding="utf-8",errors="ignore") as f:
        text = f.read().lower()

    tokens = word_tokenize(text)
    word_counts = Counter(tokens)
    vocab = sorted(word_counts)
    word2idx = {word: idx+1 for idx, word in enumerate(vocab)}  # reserve 0
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    return tokens, word2idx, idx2word
