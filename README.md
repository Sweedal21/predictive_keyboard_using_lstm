```markdown
# ⌨️ Predictive Keyboard with PyTorch & Streamlit

This project is a text autocompletion application that suggests the next word as you type. It uses an LSTM (Long Short-Term Memory) model built with PyTorch, trained on a text corpus. The project features a simple command-line interface for testing and a polished, interactive web application built with Streamlit.

---

## ✨ Features

* **LSTM-based Model**: A robust sequence model (`model.py`) built with PyTorch to learn patterns from text.
* **End-to-End Training**: A complete training script (`train.py`) that preprocesses text, builds a dataset, and trains the model from scratch.
* **Interactive Web App**: A user-friendly interface (`app_final.py`) created with Streamlit that provides real-time word suggestions.
* **Top-K Predictions**: The app suggests the top 3 most likely next words, not just the single best one.
* **CLI for Quick Testing**: A lightweight command-line script (`predict.py`) for simple model inference.
* **Modular Codebase**: The project is organized into logical modules for tokenization, model definition, training, and prediction.

---

## 🔧 How It Works (Model Architecture)

The model leverages a classic **Embedding + LSTM** architecture to predict the next word in a sequence.

1.  **Tokenization**: The input text corpus is first tokenized into words using NLTK. A vocabulary is built, mapping each unique word to an integer index.
2.  **Input Sequence**: To predict a word, the model looks at the last 4 words typed by the user.
3.  **Embedding Layer**: These 4 words (as indices) are passed to an **Embedding layer**, which converts each word into a dense vector representation, capturing semantic meaning.
4.  **LSTM Layer**: The sequence of embedding vectors is then processed by an **LSTM layer**. The LSTM captures the temporal dependencies and context from the sequence.
5.  **Linear Layer (Output)**: The output from the LSTM is fed into a final fully-connected **Linear layer**, which produces a score for every word in the vocabulary. The word with the highest score is the predicted next word.

---

## 📂 Project Structure

```

.
├── data/
│   └── sherlock1.txt     \# The training text corpus
├── models/
│   ├── keyboard\_model.pt \# Saved PyTorch model state
│   ├── word2idx.pkl      \# Saved word-to-index dictionary
│   └── idx2word.pkl      \# Saved index-to-word dictionary
├── app\_final.py          \# The main Streamlit web application
├── model.py              \# PyTorch model definition (PredictiveModel)
├── train.py              \# Script to train the model
├── predict.py            \# Command-line prediction script
├── tokenizer.py          \# Text tokenization utilities
├── requirements.txt      \# Project dependencies
└── README.md

````

---

## ⚙️ Setup and Installation

Follow these steps to set up and run the project locally.

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
````

### 2\. Create and Activate a Virtual Environment (Recommended)

```bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
```

### 3\. Install Dependencies

Install the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4\. Download NLTK Data

The tokenizer uses the NLTK `punkt` model. Run this command in your terminal or a Python interpreter to download it:

```python
import nltk
nltk.download('punkt')
```

-----

## 💡 How to Use

You must first train the model, which will generate the necessary files in the `models/` directory.

### 1\. Train the Model

Run the training script from your terminal. Make sure you have your training text (e.g., `sherlock1.txt`) in a `data/` folder.

```bash
python train.py
```

This will:

  * Preprocess the text data.
  * Train the `PredictiveModel` for 15 epochs.
  * Create a `models/` directory.
  * Save `keyboard_model.pt`, `word2idx.pkl`, and `idx2word.pkl` inside `models/`.

### 2\. Run the Streamlit Web App (Recommended)

Once the model is trained, launch the interactive web application:

```bash
streamlit run app_final.py
```

Open your browser and navigate to the local URL provided by Streamlit.

### 3\. Run the Command-Line Test

For a simpler, non-graphical test, you can use the `predict.py` script:

```bash
python predict.py
```

This will prompt you to type a phrase in the terminal and will suggest the single most likely next word.

-----

### `requirements.txt`

For reference, here is the content for your `requirements.txt` file.

```
torch
nltk
streamlit
```

