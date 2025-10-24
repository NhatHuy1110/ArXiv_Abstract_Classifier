# 🧠 ArXiv Abstract Classifier

**ArXiv Abstract Classifier** is an end-to-end Machine Learning web application built with **Streamlit**, designed to automatically **classify scientific paper abstracts** (from arXiv) into major research domains.  

The model leverages **Sentence Embeddings (E5)** and a **LightGBM classifier**, offering both interpretability (via SHAP) and interactivity through an intuitive web interface.

---


## 🧩 Project Overview

This project extends the research and code in `PaperAbstractModelingSHAP.ipynb` into a full web application.  
It covers the **complete ML workflow**:  
data preprocessing → feature engineering → model training → explainability → deployment.

### 🎯 Objectives
- Automatically classify arXiv paper abstracts into their respective scientific domains.
- Compare and evaluate various text representation techniques:
  - Bag of Words
  - TF-IDF
  - Sentence Embeddings (E5)
- Experiment with multiple ML algorithms:
  - RandomForest, AdaBoost, GradientBoosting, XGBoost, LightGBM
- Explain model predictions using SHAP values.
- Provide a clean, interactive UI for inference and analysis.

---

## 📚 Categories

The classifier predicts one of the following **five scientific fields**:

| Label | Description |
|--------|--------------|
| `astro-ph` | Astrophysics |
| `cond-mat` | Condensed Matter Physics |
| `cs` | Computer Science |
| `math` | Mathematics |
| `physics` | General Physics |

---

## 🧠 Machine Learning Pipeline

### **1️⃣ Data Preprocessing**
- Dataset: [`UniverseTBD/arxiv-abstracts-large`](https://huggingface.co/datasets/UniverseTBD/arxiv-abstracts-large)
- Filter only abstracts with single labels among the five main domains.
- Text cleaning:
  - Lowercasing, punctuation removal, tokenization
  - Stopword removal using `nltk`

---

### **2️⃣ Feature Representation**
| Method | Description |
|--------|-------------|
| **Bag of Words** | Count vectorization of tokens |
| **TF-IDF** | Weighted term frequency emphasizing unique words |
| **E5 Embeddings** | Sentence-level dense representations using `intfloat/multilingual-e5-base` |

---

### **3️⃣ Models Evaluated**
| Algorithm | Notes |
|------------|-------|
| KMeans / KNN | Baseline clustering & similarity methods |
| Decision Tree / Naive Bayes | Fast interpretable models |
| RandomForest / AdaBoost / GradientBoosting | Ensemble-based strong baselines |
| XGBoost | High-performing gradient boosting |
| **LightGBM** | Final production model due to superior accuracy & inference speed |

---

### **4️⃣ Evaluation Metrics**
- **Accuracy** – overall model performance  
- **Precision / Recall / F1-score** – per class metrics  
- **Confusion Matrix** – class-level error distribution  
- **Coherence Score (LDA)** – for topic interpretability  

---

### **5️⃣ Explainability (SHAP)**
Model interpretability powered by **SHAP (SHapley Additive exPlanations)**:
- Global and local feature contributions.
- Visualizations:
  - `summary_plot`
  - `violin_plot`
  - `heatmap`

These explain how features (or words) influenced model predictions.

---

## 🌐 Web Application

The Streamlit app (`app.py`) provides:
- 🧾 **Input**: Abstract text of a scientific paper.
- 🎯 **Output**: Predicted research field + confidence probabilities.
- 📊 **Extras**:
  - Top similar papers (via cosine similarity on embeddings)
  - Representative keywords per class (via TF-IDF)
  - Visualization of probability distribution

### 🖼 UI Preview
