import streamlit as st
import joblib
import fasttext
from constants import metrics
import numpy as np
import pandas as pd

def _softmax(x):
    x = np.asarray(x, dtype=float)
    x -= x.max()
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()

def get_top_n_predictions(model, text, n=3):
    if hasattr(model, "steps"):
        clf = model.steps[-1][1]
    else:
        clf = model

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([text])[0]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function([text])[0]
        if np.ndim(scores) == 0:
            scores = np.array([-scores, scores])

        probs = _softmax(scores)
    else:
        raise AttributeError(
            "The supplied model exposes neither predict_proba "
            "nor decision_function."
        )

    classes = getattr(clf, "classes_", None)
    if classes is None:
        raise AttributeError("Estimator lacks a classes_ attribute.")

    top_idx = np.argsort(probs)[::-1][:n]
    return [(str(classes[i]), float(probs[i])) for i in top_idx]


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
            if model_name in ("Multinomial Naive Bayes", "Logistic Regression", "Support Vector Machine"):
                st.markdown(f"**My predictions made with {model_name}:**")
                top_pred = get_top_n_predictions(model, prompt)
                df = pd.DataFrame(top_pred, columns=["Condition", "Probability"])
                df["Probability"] = df["Probability"]*100
                st.dataframe(df,
                             hide_index=True,
                             column_config={
                                    "Probability": st.column_config.NumberColumn(
                                        "Probability",
                                        format="%.2f%%"
                                    )
                                }
                             )  
                st.markdown(f"Condition most likely being described is **{response}**.")
                st.link_button(label=f"View {model_name} report", url=selected_page, icon="📊", type="secondary")
        else:
            if model_name in ("Multinomial Naive Bayes", "Logistic Regression", "Support Vector Machine"):
                st.markdown(f"**My predictions made with {model_name}:**")
                top_pred = get_top_n_predictions(model, prompt)
                df = pd.DataFrame(top_pred, columns=["Drug name", "Probability"])
                df["Probability"] = df["Probability"]*100
                st.dataframe(df,
                             hide_index=True,
                             column_config={
                                    "Probability": st.column_config.NumberColumn(
                                        "Probability",
                                        format="%.2f%%"
                                    )
                                }
                             )  
                st.markdown(f"Drug most likely to be used is **{response}**.")
                st.link_button(label=f"View {model_name} report", url=selected_page, icon="📊", type="secondary")