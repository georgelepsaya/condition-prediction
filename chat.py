import streamlit as st
import joblib
import fasttext
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
        if model_name == "Multinomial Naive Bayes":
            if target == "Condition":
                model = joblib.load(open("models/condition/mnb_condition.pkl", "rb"))
            else:
                model = joblib.load(open("models/drug/mnb_drug.pkl", "rb"))
        elif model_name == "Logistic Regression":
            if target == "Condition":
                model = joblib.load(open("models/condition/lr_condition.pkl", "rb"))
            else:
                model = joblib.load(open("models/drug/lr_drug.pkl", "rb"))
        elif model_name == "Support Vector Machine":
            if target == "Condition":
                model = joblib.load(open("models/condition/svm_condition.pkl", "rb"))
            else:
                model = joblib.load(open("models/drug/svm_drug.pkl", "rb"))
        else:
            if target == "Condition":
                model = fasttext.load_model("models/condition/ft_condition.ftz")
            else:
                model = fasttext.load_model("models/drug/ft_drug.ftz")

        response = model.predict([prompt])[0]
        
        if target == "Condition":
            st.markdown(f"Condition likely being described is **{response}**.")
            st.markdown(f"Prediction made with {model_name}.")
            st.link_button(label=f"View {model_name} report", url=selected_page, icon="📊", type="secondary")
