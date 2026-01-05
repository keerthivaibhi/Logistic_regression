import streamlit as st
import numpy as np
from sklearn.linear_model import LogisticRegression

st.title("Logistic Regression Demo")

# Generate sample data (NO dataset file)
X = np.array([[10], [20], [30], [40], [50], [60], [70], [80]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # 0 = Fail, 1 = Pass

# Train model
model = LogisticRegression()
model.fit(X, y)

# User input
marks = st.slider("Enter Marks", 0, 100, 45)

# Prediction
result = model.predict([[marks]])

# Output
if result[0] == 1:
    st.success("Student will PASS")
else:
    st.error("Student will FAIL")
