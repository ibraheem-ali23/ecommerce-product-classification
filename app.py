"""Streamlit app: type a product description and get the predicted category.

Run with:  streamlit run app.py
The app loads the model saved by `python main.py`; it never retrains.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.predict import (EXAMPLE_DESCRIPTIONS, ModelNotFoundError, load_metadata,  # noqa: E402
                         load_model, predict_text, version_warning)

st.set_page_config(page_title="Product Category Classifier", page_icon=":shopping_bags:")
st.title("E-commerce Product Classifier")
st.write("Enter a product description. The model predicts one of four categories: "
         "**Electronics, Household, Books, Clothing & Accessories**.")


@st.cache_resource
def get_model():
    return load_model()


try:
    model = get_model()
except ModelNotFoundError as exc:
    st.error("The trained model is not available.")
    st.code(str(exc))
    st.stop()

warning = version_warning()
if warning:
    st.warning(warning)
meta = load_metadata()
if meta.get("model_name"):
    st.caption(f"Loaded model: {meta['model_name']} (selected by {meta.get('selection_metric', 'metric')}).")

st.subheader("Example descriptions")
st.caption("These examples were written for demonstration only. They are NOT records from the dataset.")


def load_example():
    chosen = st.session_state["example"]
    if chosen != "(none)":
        st.session_state["description"] = EXAMPLE_DESCRIPTIONS[chosen]


st.selectbox("Load an example (optional)", ["(none)"] + list(EXAMPLE_DESCRIPTIONS),
             key="example", on_change=load_example)

text = st.text_area("Product description", key="description", height=150,
                    placeholder="e.g. Bluetooth wireless headphones with noise cancellation ...")

if st.button("Predict", type="primary"):
    try:
        result = predict_text(model, text)
    except ValueError as exc:
        st.error(str(exc))
    else:
        st.success(f"Predicted category: **{result['label']}**")
        if result["scores"]:
            label = "Probability" if result["score_type"] == "probability" else "Decision value"
            scores = pd.Series(result["scores"], name=label).sort_values(ascending=False)
            st.bar_chart(scores)
            st.dataframe(scores.round(4).to_frame())
            if result["score_type"] != "probability":
                st.caption("Decision values are distances from the SVM decision boundaries. "
                           "Larger means more confident; they are not probabilities.")
