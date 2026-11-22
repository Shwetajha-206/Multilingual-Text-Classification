import pandas as pd, joblib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, ConfusionMatrixDisplay)
from utils import clean_text

# Stage 3-4: load + clean
df = pd.read_csv("data/dataset.csv").dropna()
df["clean"] = df["text"].apply(clean_text)
df = df[df["clean"].str.len() > 0]
print("Rows:", len(df))

# split (stratified on label)
X_tr, X_te, y_tr, y_te, l_tr, l_te = train_test_split(
    df["clean"], df["label"], df["lang"], test_size=0.2,
    random_state=42, stratify=df["label"])

# Stage 6: features - character n-grams (work for every script, no stemmer needed)
def tfidf():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                           sublinear_tf=True, max_features=200000)

# Stage 7: models
models = {
    "BoW + Naive Bayes": Pipeline([
        ("vec", CountVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=200000)),
        ("clf", MultinomialNB())]),
    "TF-IDF + Logistic Regression": Pipeline([
        ("vec", tfidf()), ("clf", LogisticRegression(max_iter=300, C=5))]),
    "TF-IDF + Linear SVM": Pipeline([
        ("vec", tfidf()), ("clf", CalibratedClassifierCV(LinearSVC(), cv=3))]),
}

# Stage 8: evaluation
best_f1, best, best_pred, rows = -1, None, None, []
for name, m in models.items():
    m.fit(X_tr, y_tr)
    pred = m.predict(X_te)
    p, r, f1, _ = precision_recall_fscore_support(y_te, pred, average="weighted", zero_division=0)
    acc = accuracy_score(y_te, pred)
    rows.append([name, acc, p, r, f1])
    print(f"{name}: acc={acc:.3f}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}")
    if f1 > best_f1:
        best_f1, best, best_pred, best_name = f1, m, pred, name

pd.DataFrame(rows, columns=["model", "accuracy", "precision", "recall", "f1"]).round(4)\
  .to_csv("models/model_comparison.csv", index=False)

print("\nBEST MODEL:", best_name)
print(classification_report(y_te, best_pred, zero_division=0))

lang_rows = []
for lang in sorted(l_te.unique()):
    mask = (l_te == lang).values
    p, r, f1, _ = precision_recall_fscore_support(y_te[mask], best_pred[mask], average="weighted", zero_division=0)
    lang_rows.append([lang, int(mask.sum()), accuracy_score(y_te[mask], best_pred[mask]), p, r, f1])
per_lang = pd.DataFrame(lang_rows, columns=["language", "samples", "accuracy", "precision", "recall", "f1"]).round(4)
print("Per-language performance:\n", per_lang.to_string(index=False))
per_lang.to_csv("models/per_language_metrics.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 8))
ConfusionMatrixDisplay.from_predictions(y_te, best_pred, ax=ax, xticks_rotation=45, colorbar=False)
plt.tight_layout(); plt.savefig("screenshots/confusion_matrix.png", dpi=120)

joblib.dump(best, "models/model.joblib")
print("\nSaved models/model.joblib")