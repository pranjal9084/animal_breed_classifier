import streamlit as st

st.title("PyTorch Installation Test")

try:
    import torch
    st.success(f"Torch imported successfully! Version: {torch.__version__}")
except Exception as e:
    st.error("Torch not installed or failed to import.")
    st.exception(e)
