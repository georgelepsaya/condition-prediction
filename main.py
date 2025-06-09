import streamlit as st


pg = st.navigation([
  st.Page("chat.py", title="💬 Chat"),
  st.Page("naive_bayes.py", title="Multinomial Naive Bayes"),
  st.Page("logistic_regression.py", title="Logistic Regression"),
  st.Page("svm.py", title="Support Vector Machine"),
  st.Page("bert.py", title="BERT"),
  ],
  expanded=False
)

pg.run()