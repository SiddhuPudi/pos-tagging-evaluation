# POS Tagging Evaluation Project

## 📌 Overview
This project evaluates different Part-of-speech (POS) tagging systems using multiple evaluation metrics.

The goal is to compare traditional, statistical, and neural POS tagging models on a common dataset.

The evaluation ensures consistent token alignment across models to provide fair and accurate comparison.

---

## 🧠 Models Used

- Baseline (Rule-based)
- NLTK (Statistical)
- spaCy (Neural)
- Stanza (Deep Learning)

---

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## 🗂️ Dataset

- source: HuggingFace
- Dataset: 'batterydata/pos_tagging'
- Contains annotated sentences with Penn Treebank POS tags.

---

## 📂 Project Structure
   
```
    pos_tagging_evaluation/
    ├── dataset/
    │   └──  load_dataset.py
    ├── taggers/
    │   ├── baseline_tagger.py
    │   ├── nltk_tagger.py
    │   ├── spacy_tagger.py
    │   └──  stanza_tagger.py
    ├── evaluation/
    │   ├── metrics.py
    │   └──  confusion_matrix.py
    ├── analysis/
    │   ├── results_analysis.py
    │   └── visualization.py
    ├── report/
    │   ├── report.txt
    │   ├── Model_Accuracy_Comparision.png
    │   └──  spaCy_Confusion-Matrix.png
    ├── main.py
    ├── requirements.txt
    └── README.md
```

---

## 📦 Requirements

- Python 3.10+
- pip

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository
```bash
git clone https://github.com/SiddhuPudi/pos-tagging-evaluation
cd pos_tagging_evaluation
```

### 2️⃣ Create Virtual Environment
```bash
python3 -m venv venv
#On macOS/linux:
source venv/bin/activate
#On Windows:
venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Download Required Models
```bash
python3 -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger averaged_perceptron_tagger_en
python3 -m spacy download en_core_web_sm
python3 -c "import stanza; stanza.download('en')"
```

### 5️⃣ Run the Project
```bash
python3 main.py
```

---

## 🎯 Results Summary

| Model | Accuracy | F1 Score |
|---------|-----|-----|
| Baseline | ~0.32 | ~0.22 |
| NLTK | ~0.94 | ~0.93 |
| spaCy | ~0.95 | ~0.95 |
| Stanza | ~0.94 | ~0.94 |

---

## 🔍 Key Insights

- Rule-based models perform poorly.
- Statistical models perform well but lack context.
- Neutral models (spaCy, Stanza) perform best.
- Token alignment is critical for fair evaluation.

---

## 📊 Visualizations

- Accuracy comparison graph
- Confusion matrix (spaCy)

---

## 🚀 Future Improvements

- Add transformer-based models (BERT)
- Evaluate on noisy text (social media)
- Add runtime comparision

---

## 🧑🏻‍💻 Author

- **Thrivikram Pudi**