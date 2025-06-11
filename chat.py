import streamlit as st
import joblib
from constants import metrics

model_pages = {
    "Multinomial Naive Bayes": "naive_bayes",
    "Logistic Regression": "logistic_regression", 
    "Support Vector Machine": "svm",
    "FastText": "fasttext"
}

with st.sidebar:
    target = st.radio(
     "What do you want to predict?",
     ["Condition", "Drug Name"],
     captions=[
        "Condition likely being described",
        "Drug likely to be used"
        ]
     )
    
    model_name = st.radio(
        "Select the model",
        list(model_pages.keys()),
        captions=[
            f"Accuracy: {metrics['naive_bayes']['_'.join(target.lower().split())]['accuracy']}%",
            f"Accuracy: {metrics['logistic_regression']['_'.join(target.lower().split())]['accuracy']}%",
            f"Accuracy: {metrics['svm']['_'.join(target.lower().split())]['accuracy']}%",
            f"Accuracy: {metrics['fasttext']['_'.join(target.lower().split())]['accuracy']}%"
        ],
    )
    selected_page = model_pages[model_name]

st.subheader(f"Predict {target.lower()}")
  
if prompt := st.chat_input("Describe your condition"):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        model = joblib.load(open("models/condition/logreg.pkl", "rb"))
        response = model.predict([prompt])[0]
        st.markdown(f"Condition likely being described is **{response}**.")
        st.markdown(f"Prediction made with {model_name}.")
        st.link_button(label=f"View {model_name} report", url=selected_page, icon="📊", type="secondary")
