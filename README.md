# 🌙 SleepSense

AI-Based Phone Usage & Sleep Health Assessment using Machine Learning.

SleepSense predicts sleep quality based on daily phone usage, screen time, lifestyle factors, and physiological indicators.

---

## 🚀 Features

- **Multi-Model Support**: Compares predictions across 3 regression models:
  - **CatBoost** ($R^2 \approx 0.8044$)
  - **Gradient Boosting** ($R^2 \approx 0.8040$)
  - **Ridge Regression** ($R^2 \approx 0.7806$)
- **Interactive UI**: Built with Streamlit featuring dark mode aesthetics.
- **Personalized Recommendations**: Generates tailored advice based on user lifestyle metrics (screen time, stress, caffeine, notification volume, etc.).

---

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd ml
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Windows PowerShell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # Linux/macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

The application will launch at `http://localhost:8501`.

---

## 📁 Project Structure

```
ml/
├── app.py                                   # Main Streamlit application
├── requirements.txt                         # Python dependencies
├── sleepsense_catboost_model.cbm            # Trained CatBoost model
├── sleepsense_gradient_boosting_model.pkl   # Trained Gradient Boosting model
├── sleepsense_ridge_model.pkl               # Trained Ridge Regression model
├── sleepsense_config.pkl                    # Model feature configuration
└── .gitignore                               # Git ignore rules
```

---

## ⚠️ Disclaimer

SleepSense provides an ML-based estimate for educational and informational purposes only. It is not a medical diagnosis and should not replace professional medical advice.
