import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import PredictiveModel
from tokenizer import tokenize_text
import pickle
import os

# Custom dataset for predictive keyboard
class TextDataset(Dataset):
    def __init__(self, tokens, word2idx, seq_len=4):
        self.data = []
        self.seq_len = seq_len
        for i in range(len(tokens) - seq_len):
            input_seq = tokens[i:i + seq_len]
            target = tokens[i + seq_len]
            try:
                input_ids = [word2idx[word] for word in input_seq]
                target_id = word2idx[target]
                self.data.append((input_ids, target_id))
            except KeyError:
                continue  # skip words not in vocabulary

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x), torch.tensor(y)

# Load and tokenize text
tokens, word2idx, idx2word = tokenize_text("D:/sweedal-vscode/predictive_keyboard/data/sherlock1.txt")

# Prepare dataset and dataloader
dataset = TextDataset(tokens, word2idx, seq_len=4)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model, loss, and optimizer
vocab_size = len(word2idx) + 1
model = PredictiveModel(vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# Training loop
for epoch in range(15):
    total_loss = 0
    for x_batch, y_batch in loader:
        x_batch = x_batch.T  # shape: [seq_len, batch_size]
        preds = model(x_batch)
        loss = criterion(preds, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Save model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/keyboard_model5.pt")
print("✅ Model saved to models/keyboard_model5.pt")

# Save tokenizer (word2idx and idx2word)
with open("models/word2idx.pkl", "wb") as f:
    pickle.dump(word2idx, f)
with open("models/idx2word.pkl", "wb") as f:
    pickle.dump(idx2word, f)
print("✅ Tokenizer saved to models/word2idx.pkl and models/idx2word.pkl")



