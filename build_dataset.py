import glob, re, pandas as pd

BASE = "data/l3cube/L3Cube-IndicNews"
LANGS = ["Hindi", "Bengali", "Tamil", "English"]
MAX_PER_LANG = 15000
MIN_LANGS = 3

CANON = {
    "sports": "sports", "sport": "sports", "khel": "sports",
    "entertainment": "entertainment", "cinema": "entertainment",
    "politics": "politics", "elections": "politics",
    "health": "health", "covid": "health",
    "national": "national", "india": "national",
    "world": "world", "international": "world",
    "technology": "technology", "business": "business",
    "crime": "crime", "education": "education",
    "lifestyle": "lifestyle", "style": "lifestyle",
    "auto": "auto", "automobile": "auto", "science": "science",
}

def norm(label):
    l = str(label).strip().lower()
    l = re.sub(r"[-_ ]*(hindi|bengali|tamil|english)$", "", l)
    l = re.sub(r"[-_ ]*news$", "", l)
    return l.replace("_", "-").strip()

frames = []
for lang in LANGS:
    files = sorted(glob.glob(f"{BASE}/{lang}/SHC/*.csv"))
    if not files:
        print("No files for", lang); continue
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    tcol = "text" if "text" in d.columns else d.columns[0]
    lcol = next((c for c in ["labels", "label", "category"] if c in d.columns), d.columns[-1])
    d = d[[tcol, lcol]].dropna()
    d.columns = ["text", "label"]
    d["label"] = d["label"].apply(norm)
    dropped = sorted(set(d["label"]) - set(CANON))
    d["label"] = d["label"].map(CANON)
    d = d.dropna(subset=["label"])
    d["lang"] = lang
    print(f"{lang}: {len(d)} rows kept | dropped labels: {dropped}")
    frames.append(d)

df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["text"])

n_langs = df.groupby("label")["lang"].nunique()
keep = sorted(n_langs[n_langs >= MIN_LANGS].index)
print("\nCategories kept:", keep)
df = df[df["label"].isin(keep)]
df = pd.concat([g.sample(min(len(g), MAX_PER_LANG), random_state=42) for _, g in df.groupby("lang")])
df.to_csv("data/dataset.csv", index=False)
print("\nSaved data/dataset.csv", df.shape)
print(pd.crosstab(df["label"], df["lang"]))