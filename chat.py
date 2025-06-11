from turtle import onclick
import streamlit as st
import joblib

model_pages = {
    "Multinomial Naive Bayes": "naive_bayes",
    "Logistic Regression": "logistic_regression", 
    "Support Vector Machine": "svm",
    "BERT": "bert"
}

with st.sidebar:
  model_name = st.radio(
      "Select the model",
      list(model_pages.keys()),
      captions=[
          "Accuracy: 70%",
          "Accuracy: 76%",
          "Accuracy: 80%",
          "Accuracy: 89%"
      ],
  )

  selected_page = model_pages[model_name]

  target = st.radio(
     "What do you want to predict?",
     ["Condition", "Drug Name"],
     captions=[
        "Condition likely being described",
        "Drug likely to be used"
        ]
     )

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