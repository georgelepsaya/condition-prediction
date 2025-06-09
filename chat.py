import streamlit as st
import joblib

st.subheader("Predict condition or drug name")

with st.sidebar:
  models = st.radio(
      "Select the model",
      ["Multinomial Naive Bayes", "Logistic Regression", "Support Vector Machine", "BERT"],
      captions=[
          "Accuracy: 70%",
          "Accuracy: 76%",
          "Accuracy: 80%",
          "Accuracy: 89%"
      ],
  )

  target = st.radio(
     "What do you want to predict?",
     ["Condition", "Drug Name"],
     captions=[
        "Condition likely being described",
        "Drug likely to be used"
        ]
     )
  
if prompt := st.chat_input("Describe your condition"):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        model = joblib.load(open("models/log_reg.pkl", "rb"))
        response = model.predict([prompt])[0]
        st.markdown(response)