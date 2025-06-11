import streamlit as st

st.title("Logistic Regression")

col1, col2, col3, col4 = st.columns(4, border=True)
col1.metric("Accuracy", "77%")
col2.metric("F1 (macro avg.)", "74%")
col3.metric("Precision (macro avg.)", "71%")
col4.metric("Recall (macro avg.)", "79%")
