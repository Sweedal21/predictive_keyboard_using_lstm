import streamlit as st
import torch
import pickle
from model import PredictiveModel
import heapq
import datetime

# --- Utility functions ---
def encode_input(text, word2idx, max_len=4):
    words = text.strip().lower().split()[-max_len:]
    encoded = [word2idx.get(w, 0) for w in words]
    while len(encoded) < max_len:
        encoded.insert(0, 0)
    return encoded

def decode_word(idx, idx2word):
    return idx2word.get(idx, "<unk>")

# --- Load model and tokenizer ---
with open("models/word2idx.pkl", "rb") as f:
    word2idx = pickle.load(f)
with open("models/idx2word.pkl", "rb") as f:
    idx2word = pickle.load(f)

vocab_size = len(word2idx) + 1
model = PredictiveModel(vocab_size)
model.load_state_dict(torch.load("models/keyboard_model.pt"))
model.eval()

# --- Prediction function ---
def predict_top_k(text, k=3):
    input_seq = encode_input(text, word2idx)
    input_tensor = torch.tensor(input_seq).unsqueeze(1)  # shape: [seq_len, 1]
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).squeeze()
        topk_indices = heapq.nlargest(k, range(len(probs)), probs.__getitem__)
        return [decode_word(idx, idx2word) for idx in topk_indices]

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Predictive Keyboard", layout="centered")
st.title("⌨️ Predictive Keyboard")
st.markdown("Type a phrase. Press Enter or space to get predictions. Click on a word to autocomplete.")

# --- Session State ---
if "current_text" not in st.session_state:
    st.session_state.current_text = ""
if "history" not in st.session_state:
    st.session_state.history = []
if "show_suggestions" not in st.session_state:
    st.session_state.show_suggestions = False
if "last_text" not in st.session_state:
    st.session_state.last_text = ""

# --- Chat History ---
st.markdown("### 💬 Chat History:")
with st.container():
    for item in reversed(st.session_state.history):
        st.markdown(f"🕒 *{item['time']}*")
        st.success(f"{item['text']}")

# --- Input + Send Button ---
st.markdown("### ✍️ Type your message:")
input_col, send_col = st.columns([5, 1])

with input_col:
    typed = st.text_input(
        " ", 
        value=st.session_state.current_text, 
        label_visibility="collapsed", 
        key="typed_input",
        on_change=None  # We'll handle changes manually
    )

with send_col:
    if st.button("➤", use_container_width=True):
        if typed.strip():
            st.session_state.history.append({
                "text": typed.strip() + " 😊",
                "time": datetime.datetime.now().strftime("%H:%M:%S")
            })
            st.session_state.current_text = ""
            st.session_state.show_suggestions = False
            st.session_state.last_text = ""
            st.rerun()

# --- Check if we should show suggestions ---
# Show suggestions if:
# 1. Text ends with space, OR
# 2. Text has changed (indicating Enter was pressed or new input), OR  
# 3. Text is not empty and we have at least one word
should_show_suggestions = False

if typed != st.session_state.last_text:
    # Text has changed
    if typed.endswith(" ") or (typed.strip() and not typed.endswith(" ") and len(typed.strip().split()) > 0):
        should_show_suggestions = True
        st.session_state.show_suggestions = True
    elif typed.strip() == "":
        st.session_state.show_suggestions = False
    
    st.session_state.last_text = typed
    st.session_state.current_text = typed

# --- Dynamic Suggestions ---
if st.session_state.show_suggestions and typed.strip():
    # Get suggestions based on current text
    suggestion_text = typed.strip()
    if not suggestion_text.endswith(" "):
        # If doesn't end with space, add one for prediction
        suggestion_text += " "
    
    suggestions = predict_top_k(suggestion_text)
    
    if suggestions and any(word != '<unk>' for word in suggestions):
        st.markdown("### 🔮 Suggestions:")
        # Filter out unknown tokens
        valid_suggestions = [word for word in suggestions if word != '<unk>']
        
        if valid_suggestions:
            cols = st.columns(len(valid_suggestions))
            for i, word in enumerate(valid_suggestions):
                if cols[i].button(word, key=f"suggestion_{word}_{i}"):
                    # Add the selected word
                    if st.session_state.current_text.endswith(" "):
                        st.session_state.current_text += word + " "
                    else:
                        st.session_state.current_text += " " + word + " "
                    st.session_state.show_suggestions = True  # Keep showing suggestions
                    st.rerun()

# --- Instructions ---
st.markdown("---")
st.markdown("""
**Instructions:**
- Type your message in the text box
- Press **Enter** or add a **space** to see word suggestions
- Click on any suggested word to add it to your text
- Click the **➤** button to send your message
""")

# --- Styling ---
st.markdown("""
<style>
input {
    font-size: 18px !important;
}
div[data-testid="column"] > div {
    text-align: center;
}
button[kind="secondary"] {
    font-size: 16px !important;
    padding: 10px 20px;
    margin-top: 5px;
    border-radius: 10px;
    background-color: #f0f2f6;
}
button[kind="secondary"]:hover {
    background-color: #e1e5e9;
    border-color: #d1d5db;
}
</style>
""", unsafe_allow_html=True)