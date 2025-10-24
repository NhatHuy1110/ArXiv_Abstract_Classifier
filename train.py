# train.py
import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb
import re
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# ------------------------------
# 1) Load & filter data
# ------------------------------
def clean_text(s: str) -> str:
    s = s.replace("\n", " ")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\d+", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def load_arxiv_subset(max_docs_per_class=600, seed=42):
    ds = load_dataset("UniverseTBD/arxiv-abstracts-large", split="train")
    print("Available columns:", ds.column_names[:15])  # <-- debug xem tên cột

    wanted = ["astro-ph", "cond-mat", "cs", "math", "physics"]

    # Cột abstract có thể khác tên (vd. 'abs' hoặc 'text')
    abstract_field = None
    for cand in ["abstract", "abs", "text", "summary", "content"]:
        if cand in ds.column_names:
            abstract_field = cand
            break
    if not abstract_field:
        raise ValueError("❌ Không tìm thấy cột chứa abstract trong dataset.")

    rows = []
    per_class_cnt = {k: 0 for k in wanted}
    for r in ds:
        labs = r.get("categories", []) or []
        # Kiểm tra categories có dạng list hay string
        if isinstance(labs, str):
            labs = [labs]

        labs = [c for c in labs if c in wanted]
        if len(labs) != 1:
            continue
        lab = labs[0]

        if per_class_cnt[lab] >= max_docs_per_class:
            continue

        abs_text = (r.get("abstract") or "").strip()
        if len(abs_text) < 40:
            continue

        rows.append({
            "title": r.get("title", ""),
            "abstract": abs_text,
            "label": lab,
        })
        per_class_cnt[lab] += 1

        if all(v >= max_docs_per_class for v in per_class_cnt.values()):
            break

    # ✅ Kiểm tra kết quả
    if not rows:
        raise ValueError("❌ Không lấy được mẫu nào! Kiểm tra giá trị trong cột 'categories' có trùng với wanted không.")

    df = pd.DataFrame(rows)
    print("✅ Sample rows:")
    print(df.head())

    df["abstract_clean"] = df["abstract"].apply(clean_text)
    print(f"✅ Loaded {len(df)} samples.")
    return df

# ------------------------------
# 2) Embedding model
# ------------------------------
EMB_MODEL_NAME = "intfloat/multilingual-e5-base"
def encode_texts(model, texts, batch_size=64, normalize=True):
    prompts = [f"passage: {t}" for t in texts]
    emb = model.encode(
        prompts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,
    )
    return np.array(emb, dtype=np.float32)

# ------------------------------
# 3) Train & export
# ------------------------------
def main():
    print("Loading data ...")
    df = load_arxiv_subset(max_docs_per_class=600)  # tổng ~3k mẫu
    label_names = sorted(df["label"].unique())
    label2id = {lb: i for i, lb in enumerate(label_names)}
    y_full = df["label"].map(label2id).values
    X_full = df["abstract_clean"].values

    X_train_txt, X_test_txt, y_train, y_test, meta_train, meta_test = train_test_split(
        X_full, y_full, df[["title", "abstract", "label"]].values,
        test_size=0.2, stratify=y_full, random_state=42
    )

    print("Loading embedding model ...")
    emb_model = SentenceTransformer(EMB_MODEL_NAME)

    print("Encoding train/test ...")
    X_train = encode_texts(emb_model, list(X_train_txt))
    X_test  = encode_texts(emb_model, list(X_test_txt))

    print("Training LightGBM ...")
    clf = lgb.LGBMClassifier(
        boosting_type="gbdt",  # goss/dart cũng được
        n_estimators=800,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy (embeddings + LGBM): {acc:.4f}")
    print(classification_report(y_test, preds, target_names=label_names))

    # --------------------------
    # Similarity index (cosine)
    # --------------------------
    print("Fitting NearestNeighbors index ...")
    nn = NearestNeighbors(n_neighbors=5, metric="cosine", n_jobs=-1)
    nn.fit(X_train)  # index trên embeddings train

    # --------------------------
    # Class keywords by TF-IDF
    # --------------------------
    print("Building class-wise TF-IDF keywords ...")
    tfidf = TfidfVectorizer(
        stop_words="english",
        max_df=0.9,
        min_df=3,
        max_features=3000,
    )
    tfidf.fit(X_train_txt)

    # top words mỗi class = từ có mean TF-IDF cao nhất trong class
    class_keywords = {}
    vocab = np.array(tfidf.get_feature_names_out())
    X_tfidf_train = tfidf.transform(X_train_txt)
    for lb, idx in label2id.items():
        rows = (y_train == idx)
        if rows.sum() == 0:
            class_keywords[lb] = []
            continue
        mean_scores = np.asarray(X_tfidf_train[rows].mean(axis=0)).ravel()
        top_idx = np.argsort(mean_scores)[-20:][::-1]
        class_keywords[lb] = vocab[top_idx].tolist()

    # --------------------------
    # Export artifacts
    # --------------------------
    print("Saving artifacts ...")
    joblib.dump(clf, ARTIFACTS/"lgbm_model.pkl")
    (ARTIFACTS/"emb_model_name.txt").write_text(EMB_MODEL_NAME)
    joblib.dump(nn, ARTIFACTS/"nn_index.pkl")
    joblib.dump(tfidf, ARTIFACTS/"tfidf_explainer.pkl")
    json.dump(label_names, open(ARTIFACTS/"label_names.json", "w"))
    json.dump(
        {
            "train_titles": [t for t, a, l in meta_train],
            "train_abstracts": [a for t, a, l in meta_train],
            "train_labels": [str(l) for t, a, l in meta_train],
        },
        open(ARTIFACTS/"train_meta.json", "w"),
    )
    json.dump(class_keywords, open(ARTIFACTS/"class_keywords.json", "w"))
    (ARTIFACTS/"readme.txt").write_text(
        f"Accuracy: {acc:.4f}\nModel: LightGBM + {EMB_MODEL_NAME}\n"
    )
    print("Done.")

if __name__ == "__main__":
    main()
