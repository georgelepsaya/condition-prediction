import streamlit as st


pg = st.navigation({
  "": [st.Page("chat.py", title="💬 Chat")],
  "📊 Reports and Metrics": [
    st.Page("reports/naive_bayes.py", title="Multinomial Naive Bayes"),
    st.Page("reports/logistic_regression.py", title="Logistic Regression"),
    st.Page("reports/svm.py", title="Support Vector Machine"),
    st.Page("reports/fasttext.py", title="FastText")
    ]
  },
  expanded=False
)

pg.run()