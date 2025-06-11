import streamlit as st

from constants import metrics

st.title("Multinomial Naive Bayes")

cond = metrics["naive_bayes"]["condition"]
drug = metrics["naive_bayes"]["drug_name"]

st.subheader("Condition")
cond_1, cond_2, cond_3, cond_4 = st.columns(4, border=True)
cond_1.metric("Accuracy", f"{cond['accuracy']}%")
cond_2.metric("F1 (macro avg.)", f"{cond['macro_avg_f1']}%")
cond_3.metric("Precision (macro avg.)", f"{cond['macro_avg_precision']}%")
cond_4.metric("Recall (macro avg.)", f"{cond['macro_avg_recall']}%")

st.subheader("Drug name")
drug_1, drug_2, drug_3, drug_4 = st.columns(4, border=True)
drug_1.metric("Accuracy", f"{drug['accuracy']}%")
drug_2.metric("F1 (macro avg.)", f"{drug['macro_avg_f1']}%")
drug_3.metric("Precision (macro avg.)", f"{drug['macro_avg_precision']}%")
drug_4.metric("Recall (macro avg.)", f"{drug['macro_avg_recall']}%")