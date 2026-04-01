import streamlit as st
import torch
import pickle
import re
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="🏥 Medical Chatbot", page_icon="🏥", layout="centered")

@st.cache_resource
def load_model():
    MODEL_REPO = "Gromminite/medical-chatbot-bert-model"
    tokenizer = BertTokenizer.from_pretrained(MODEL_REPO)
    model = BertForSequenceClassification.from_pretrained(MODEL_REPO)
    model.eval()
    le_path = hf_hub_download(repo_id=MODEL_REPO, filename="label_encoder.pkl", repo_type="model")
    with open(le_path, "rb") as f:
        le = pickle.load(f)
    device = torch.device("cpu")
    model = model.to(device)
    return model, tokenizer, le, device

with st.spinner("⏳ Loading BERT model... please wait"):
    model, tokenizer, le, device = load_model()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = __import__("re").sub(r"<[^>]+>", " ", text)
    text = __import__("re").sub(r"[^a-z0-9\s]", " ", text)
    text = __import__("re").sub(r"\s+", " ", text)
    return text.strip()

def predict_answer(question, top_k=3):
    cleaned = clean_text(question)
    if len(cleaned.strip()) < 3:
        return None, 0.0, []
    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding="max_length", max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
    top_k_probs, top_k_ids = torch.topk(probs, k=min(top_k, len(probs)))
    top_k_probs = top_k_probs.cpu().numpy()
    top_k_ids = top_k_ids.cpu().numpy()
    best_conf = float(top_k_probs[0])
    best_answer = le.inverse_transform([top_k_ids[0]])[0]
    alt_answers = []
    for i in range(1, len(top_k_ids)):
        alt = le.inverse_transform([top_k_ids[i]])[0]
        alt_conf = float(top_k_probs[i])
        if alt[:50] != best_answer[:50]:
            alt_answers.append((alt, alt_conf))
    return best_answer, best_conf, alt_answers

st.title("🏥 Medical Chatbot")
st.markdown("*Powered by BERT — Fine-tuned on Medical Q&A Dataset*")
st.divider()
st.markdown("> ⚠️ **Disclaimer:** This chatbot is for **educational purposes only**. Always consult a qualified medical professional.")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "👋 Hello! I am a medical chatbot powered by BERT. Ask me a medical question!"})

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a medical question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("🔍 Analyzing your question..."):
            answer, confidence, alternatives = predict_answer(prompt)
        if answer is None or confidence < 0.25:
            response = "I am not confident enough to answer that accurately. Try rephrasing or ask about a specific symptom or disease."
        else:
            lines_resp = ["**Answer:**", "", answer, "", "---"]
            lines_resp.append("🎯 *Confidence: " + f"{confidence:.2%}" + " | Model: bert-base-uncased*")
            if alternatives:
                lines_resp.append("")
                lines_resp.append("**📋 Other possible answers:**")
                for i, (alt, alt_conf) in enumerate(alternatives[:2], 1):
                    short = alt[:180] + "..." if len(alt) > 180 else alt
                    lines_resp.append("")
                    lines_resp.append("*Option " + str(i) + " (" + f"{alt_conf:.1%}" + "):* " + short)
            response = "\n".join(lines_resp)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})