import os, joblib, pandas as pd, streamlit as st
from utils import clean_text, detect_language

st.set_page_config(page_title="BhashaClassify", page_icon="🌐")
st.markdown("""
<div style="background: linear-gradient(90deg, #1E6FD9, #14B8A6);
            padding: 22px 26px; border-radius: 14px; margin-bottom: 18px;">
  <h1 style="color: white; margin: 0; font-size: 2.2rem;">🌐 BhashaClassify</h1>
  <p style="color: #E6F1F5; margin: 6px 0 0 0;">
    Multilingual news classifier · English · हिंदी · বাংলা · தமிழ்
  </p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("models/model.joblib")

model = load_model()

choice = st.radio("Language", ["Auto-detect", "English", "Hindi", "Bengali", "Tamil"], horizontal=True)
text = st.text_area("Enter a news headline", height=120,
                    placeholder="e.g. भारत ने मैच जीता / India wins the match")

if st.button("Classify Text", type="primary"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        detected = detect_language(text)
        lang = detected if choice == "Auto-detect" else choice
        if choice != "Auto-detect" and detected not in (choice, "Unknown"):
            st.warning(f"You selected {choice}, but the text looks like {detected}.")
        cleaned = clean_text(text)
        if not cleaned:
            st.warning("No usable words found in the text.")
        else:
            proba = model.predict_proba([cleaned])[0]
            top = proba.argmax()
            st.success(f"Predicted category: **{model.classes_[top]}**  (confidence {proba[top]:.1%})")
            st.write(f"Language: **{lang}**")
            chart = pd.Series(proba, index=model.classes_).sort_values(ascending=False)
            st.bar_chart(chart)

with st.expander("Model performance (test set)"):
    for f, title in [("models/model_comparison.csv", "Model comparison"),
                     ("models/per_language_metrics.csv", "Per-language performance")]:
        if os.path.exists(f):
            st.write(title)
            st.dataframe(pd.read_csv(f), hide_index=True)