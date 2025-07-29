# Employee_salary_pred.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Page configuration
st.set_page_config(
    page_title="Employee Salary Predictor",
    page_icon="💼",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Salary_Data.csv")
    df.dropna(inplace=True)
    return df

df = load_data()

# Title
st.markdown("""
    <style>
        .title-container {
            text-align: center;
            padding: 1rem 0;
        }
        .title-container h1 {
            color: var(--text-color);
            font-size: 3em;
            font-weight: bold;
        }
    </style>
    <div class="title-container">
        <h1>Employee Salary Predictor</h1>
    </div>
""", unsafe_allow_html=True)
 

# Sidebar: User inputs
st.markdown("### Enter Employee Details")

col1, col2, col3 = st.columns(3)
with col1:
    Age = st.slider("Age", int(df["Age"].min()), int(df["Age"].max()), int(df["Age"].mean()))
with col2:
    Experience = st.slider("Years of Experience", int(df["Years of Experience"].min()), int(df["Years of Experience"].max()), int(df["Years of Experience"].mean()))
with col3:
    Gender = st.selectbox("Gender", df["Gender"].unique())

col4, col5 = st.columns(2)
with col4:
    Job_Title = st.selectbox("Job Title", sorted(df['Job Title'].unique()))
with col5:
    Education = st.selectbox("Education Level", sorted(df['Education Level'].unique()))


# Input DataFrame
user_input = pd.DataFrame([{
    "Age": Age,
    "Years of Experience": Experience,
    "Job Title": Job_Title,
    "Education Level": Education,
    "Gender": Gender
}])

# Feature setup
X = df[["Age", "Years of Experience", "Job Title", "Education Level", "Gender"]]
y = df["Salary"]

numeric_features = ["Age", "Years of Experience"]
categorical_features = ["Job Title", "Education Level", "Gender"]

# Preprocessing and modeling pipeline
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_features)
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

if st.button("Predict Salary"):
    predicted_salary = pipeline.predict(user_input)[0]
    st.header(f"**Predicted Salary:** ₹{predicted_salary:,.0f}")

# Feature importance chart
st.markdown("---")
st.subheader("🔍 Feature Importance")
importances = pipeline.named_steps['model'].feature_importances_
feat_names = numeric_features + list(
    pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
)
importance_series = pd.Series(importances, index=feat_names).sort_values(ascending=True)
st.bar_chart(importance_series)

# Model Evaluation
st.markdown("---")
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

col1, col2 = st.columns(2)
with col1:
    st.metric("📉 Mean Squared Error", f"{mse:.2f}")
with col2:
    st.metric("📈 R² Score", f"{r2:.3f}")

# Show dataset
with st.expander("📄 View Sample Data(DATASET)"):
    st.dataframe(df.head())
