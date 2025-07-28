# 🧠 Employee Salary Predictor

This is a **Streamlit web app** that predicts employee salaries based on:
- Age
- Education Level
- Job Title
- Years of Experience
- Gender

It uses a machine learning model (Random Forest) trained on real-world employee data.

---

## 📸 Demo
<img width="1909" height="1034" alt="image" src="https://github.com/user-attachments/assets/21289907-4b8a-4710-a8c9-aaae80fdf43e" />
https://employesalarypredictor.streamlit.app/


---

## 🚀 Features

- 📊 Predict salaries interactively
- 🎛️ Customizable inputs with sliders and dropdowns
- 📈 Displays model performance (MSE, R²)
- 🌗 Theme toggle: Light / Dark / System
- 🔥 Visualizes feature importance

---

## 🧰 Tech Stack

- **Frontend/UI**: [Streamlit](https://streamlit.io/)
- **Model**: Random Forest Regressor
- **Libraries**: pandas, scikit-learn, NumPy

---

## 🔧 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/your-username/employee-salary-predictor.git
cd employee-salary-predictor
streamlit run "Employee Salary Predictor.py"
