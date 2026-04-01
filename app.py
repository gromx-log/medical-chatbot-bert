import streamlit as st
import torch
import pickle
import re
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
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
    top_k_probs, top_k_ids = torch.topk(probs, k=min(top_k, len(probs)))
    top_k_probs = top_k_probs.cpu().numpy()
    top_k_ids = top_k_ids.cpu().numpy()
    best_conf = float(top_k_probs[0])
    best_answer = le.inverse_transform([top_k_ids[0]])[0]
    alt_answers = []
    for i in range(1, len(top_k_ids)):
        alt = le.inverse_transform([top_k_ids[i]])[0]
        if alt[:50] != best_answer[:50]:
            alt_answers.append((alt, float(top_k_probs[i])))
    return best_answer, best_conf, alt_answers

st.title("🏥 Medical Chatbot")
st.markdown("*Powered by BERT — Fine-tuned on Medical Q&A Dataset*")
st.divider()
st.info("⚠️ This chatbot is for **educational purposes only**. Always consult a qualified medical professional.")

# Sidebar with model info
with st.sidebar:
    st.header("ℹ️ Model Info")
    st.markdown("**Model:** bert-base-uncased")
    st.markdown("**Task:** Medical Q&A Classification")
    st.markdown("**Classes:** 20")
    st.markdown("**Val Accuracy:** 68.32%")
    st.markdown("**Dataset:** Comprehensive Medical Q&A")
    st.divider()
    st.caption("Note: This model is trained on NIH genetic disease data. It performs best on questions about inherited conditions.")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "👋 Hello! I am a medical chatbot powered by BERT, trained on NIH genetic disease data. I work best with questions about inherited conditions and genetic diseases!"})

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
        if answer is None:
            response = "Please type a valid medical question."
        else:
            resp_lines = ["**Answer:**", "", answer, "", "---",
                "🎯 *Confidence: " + f"{confidence:.2%}" + " | Model: bert-base-uncased*"]
            if alternatives:
                resp_lines += ["", "**📋 Related answers:**"]
                for i, (alt, alt_conf) in enumerate(alternatives[:2], 1):
                    short = alt[:180] + "..." if len(alt) > 180 else alt
                    resp_lines.append("*Option " + str(i) + " (" + f"{alt_conf:.1%}" + "):* " + short)
            response = "\n".join(resp_lines)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})