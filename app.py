"""Streamlit demo: streamlit run app.py"""
import streamlit as st

from detect import DEFAULT_MODEL, detect, load_detector

st.set_page_config(page_title="Prompt Detector", page_icon="🔎")
st.title("Human vs. synthetic prompt detector")


@st.cache_resource
def get_bundle():
    return load_detector(DEFAULT_MODEL)


if not DEFAULT_MODEL.exists():
    st.error("detector.joblib not found. Run `python train.py` (or `--no-lm`) first.")
    st.stop()

bundle = get_bundle()
st.caption("Mode: " + ("heuristics + GPT-2" if bundle["uses_lm"] else "heuristics only"))

text = st.text_area("Paste a prompt", height=200)
if st.button("Analyze") and text.strip():
    res = detect(text, bundle)
    p = res["probability_synthetic"]
    c1, c2 = st.columns(2)
    c1.metric("P(synthetic)", f"{p:.2f}")
    c2.metric("Verdict", res["verdict"])
    st.progress(min(max(p, 0.0), 1.0))
    st.subheader("Signals")
    if res["signals"]:
        for s in res["signals"]:
            st.write(f"- **{s['points_to']}**: {s['reason']}")
    else:
        st.write("No notable signals.")
    if res["word_count"] < 15:
        st.warning("Very short prompts are unreliable.")
