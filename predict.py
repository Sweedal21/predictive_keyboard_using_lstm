
import torch
from model import PredictiveModel
from tokenizer import tokenize_text
from nltk.tokenize import word_tokenize

tokens, word2idx, idx2word = tokenize_text("data/sherlock1.txt")

model = PredictiveModel(len(word2idx) + 1)
model.load_state_dict(torch.load("models/keyboard_model.pt"))
model.eval()

def predict_next(text):
    words = word_tokenize(text.lower())[-4:]
    input_seq = [word2idx.get(w, 0) for w in words]
    while len(input_seq) < 4:
        input_seq.insert(0, 0)
    input_tensor = torch.tensor(input_seq).unsqueeze(1)  # shape [seq_len, 1]
    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = torch.argmax(output).item()
    return idx2word.get(pred_idx, "<unk>")

# Test
while True:
    inp = input("\nType a phrase (or 'exit'): ")
    if inp.lower() == 'exit':
        break
    next_word = predict_next(inp)
    print(f"Suggested next word: {next_word}")
