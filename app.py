
import streamlit as st
import torch
import pickle
import re
from transformers import BertTokenizer, BertForSequenceClassification
from huggingface_hub import hf_hub_download

# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="🏥 Medical Chatbot",
    page_icon="🏥",
    layout="centered"
)

# ─── Load Model from HF Model Hub ──────────────────────────────
@st.cache_resource
def load_model():
    MODEL_REPO = "Gromminite/medical-chatbot-bert-model"  # ← Model repo
    
    tokenizer = BertTokenizer.from_pretrained(MODEL_REPO)
    model     = BertForSequenceClassification.from_pretrained(MODEL_REPO)
    model.eval()
    
    # Download label encoder
    le_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="label_encoder.pkl",
        repo_type="model"
    )
    with open(le_path, "rb") as f:
        le = pickle.load(f)
    
    device = torch.device("cpu")  # Streamlit Cloud has no GPU
    model  = model.to(device)
    
    return model, tokenizer, le, device

with st.spinner("⏳ Loading BERT model... please wait"):
    model, tokenizer, le, device = load_model()

# ─── Text Cleaning ──────────────────────────────────────────────
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ─── Prediction ─────────────────────────────────────────────────
def predict_answer(question):
    cleaned = clean_text(question)
    inputs  = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    ).to(device)
    
    with torch.no_grad():
        outputs    = model(**inputs)
        logits     = outputs.logits
        pred_id    = torch.argmax(logits, dim=-1).item()
        confidence = torch.softmax(logits, dim=-1).max().item()
    
    answer = le.inverse_transform([pred_id])[0]
    return answer, confidence

# ─── UI ─────────────────────────────────────────────────────────
st.title("🏥 Medical Chatbot")
st.markdown("*Powered by BERT — Fine-tuned on Medical Q&A Dataset*")
st.divider()

st.markdown("""
> ⚠️ **Disclaimer:** This chatbot is for educational purposes only.
> Always consult a qualified medical professional for health advice.
""")

# ─── Chat History ───────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "👋 Hello! I am a medical chatbot powered by BERT. Ask me a medical question!"
    })

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─── Chat Input ─────────────────────────────────────────────────
if prompt := st.chat_input("Ask a medical question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("🔍 Analyzing your question..."):
            answer, confidence = predict_answer(prompt)
        
        response = f"""
{answer}

---
🎯 *Confidence: {confidence:.2%} | Model: bert-base-uncased*
"""
        st.markdown(response)
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })
