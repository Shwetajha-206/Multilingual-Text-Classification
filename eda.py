import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("data/dataset.csv")
print(df.info())
print("Missing:\n", df.isna().sum())
print("Duplicates:", df.duplicated(subset=["text"]).sum())
print(pd.crosstab(df["label"], df["lang"]))

df["lang"].value_counts().plot(kind="bar", title="Samples per language", rot=0)
plt.tight_layout(); plt.savefig("screenshots/eda_language.png", dpi=120); plt.clf()

df["label"].value_counts().plot(kind="barh", title="Samples per category")
plt.tight_layout(); plt.savefig("screenshots/eda_labels.png", dpi=120); plt.clf()

pd.crosstab(df["label"], df["lang"]).plot(kind="bar", figsize=(11, 5), title="Category by language")
plt.tight_layout(); plt.savefig("screenshots/eda_label_by_lang.png", dpi=120); plt.clf()

df["words"] = df["text"].astype(str).str.split().str.len()
df.boxplot(column="words", by="lang"); plt.suptitle("")
plt.title("Text length (words) by language")
plt.savefig("screenshots/eda_length.png", dpi=120)
print("\nAvg words per language:\n", df.groupby("lang")["words"].mean().round(1))
print("\nCharts saved in screenshots/")