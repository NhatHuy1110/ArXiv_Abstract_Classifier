# app.py
import streamlit as st
import joblib, json, numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

ART = Path("artifacts")
LABELS = json.load(open(ART/"label_names.json"))
EMB_MODEL_NAME = (ART/"emb_model_name.txt").read_text().strip()

# cache để không load lại
@st.cache_resource(show_spinner=False)
def load_models():
    emb = SentenceTransformer(EMB_MODEL_NAME)
    clf = joblib.load(ART/"lgbm_model.pkl")
    nn = joblib.load(ART/"nn_index.pkl")
    tfidf = joblib.load(ART/"tfidf_explainer.pkl")
    train_meta = json.load(open(ART/"train_meta.json"))
    class_keywords = json.load(open(ART/"class_keywords.json"))
    return emb, clf, nn, tfidf, train_meta, class_keywords

def encode_one(emb_model, text: str) -> np.ndarray:
    text = text.strip()
    prompt = f"passage: {text}"
    v = emb_model.encode([prompt], show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(v, dtype=np.float32)

st.set_page_config(page_title="ArXiv Abstract Classifier", page_icon="🧠", layout="wide")
st.title("🧠 ArXiv Abstract Classifier")
st.caption("Embeddings (E5) + LightGBM • Probabilities • Similar Papers • Class Keywords")

with st.sidebar:
    st.markdown("### Settings")
    topk = st.slider("Top similar papers", 1, 10, 3)
    show_keywords = st.checkbox("Show class keywords", value=True)
    st.divider()
    st.markdown("Model")
    st.code(f"Encoder: {EMB_MODEL_NAME}\nClassifier: LightGBM", language="yaml")

emb_model, clf, nn, tfidf, train_meta, class_keywords = load_models()

default_text = """We propose a novel neural architecture for efficient transformer inference,
reducing memory footprint while maintaining accuracy on common NLP tasks. 
Experiments on translation and summarization demonstrate competitive results."""
text = st.text_area("Paste paper abstract here:", default_text, height=220)

col1, col2 = st.columns([1,1])
with col1:
    run = st.button("🔍 Classify")
with col2:
    clear = st.button("🧹 Clear")
    if clear:
        st.experimental_rerun()

if run:
    if not text.strip():
        st.warning("Please enter an abstract.")
        st.stop()

    # 1) Encode & predict
    v = encode_one(emb_model, text)
    probs = clf.predict_proba(v)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = LABELS[pred_idx]

    st.success(f"**Predicted field:** `{pred_label}`")
    st.write("### Class probabilities")
    prob_dict = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    st.bar_chart(prob_dict)

    # 2) Similar papers (cosine via NN index on train embeddings)
    st.write("### 🔗 Most similar training papers")
    # dùng nn kneighbors để tìm index gần nhất; lấy metadata để hiển thị
    dists, idxs = nn.kneighbors(v, n_neighbors=max(topk, 3), return_distance=True)
    idxs = idxs[0].tolist()
    dists = dists[0].tolist()

    titles = train_meta["train_titles"]
    abstracts = train_meta["train_abstracts"]
    labels = train_meta["train_labels"]

    for rank, (i, d) in enumerate(zip(idxs[:topk], dists[:topk]), start=1):
        cos = 1 - d
        with st.container(border=True):
            st.markdown(f"**#{rank}. {titles[i]}**")
            st.caption(f"_Label:_ `{labels[i]}` • _Cosine similarity:_ **{cos:.3f}**")
            st.write(abstracts[i][:600] + ("..." if len(abstracts[i]) > 600 else ""))

    # 3) Class keywords
    if show_keywords:
        st.write("### 🏷️ Class keywords (TF-IDF centroids)")
        cols = st.columns(len(LABELS))
        for j, lb in enumerate(LABELS):
            with cols[j]:
                st.markdown(f"**{lb}**")
                st.write(", ".join(class_keywords.get(lb, [])[:15]))
