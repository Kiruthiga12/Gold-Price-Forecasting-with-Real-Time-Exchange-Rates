# 💰 **Gold Price Forecasting with Real-Time Exchange Rates**

Predict gold prices based on the USD to INR exchange rate using a regression model trained on real-world data.

![Gradio App Screenshot](https://your-image-link.com/gradio-screenshot.png)

## 🚀 Project Summary

This project leverages historical financial data and machine learning to forecast gold prices with real-time USD-INR rates. A simple and interactive Gradio web app enables users to input exchange rates and get accurate gold rate predictions.

---

## 📊 Tech Stack & Tools

- **Python**, **Pandas**, **NumPy**
- **scikit-learn** (Linear Regression, RandomizedSearchCV)
- **Matplotlib**, **Seaborn** for visualization
- **Gradio** for web-based ML demo
- **yFinance** for live financial data
- **Pickle** for model serialization

---

## 🧠 ML Workflow

1. **Data Acquisition**: Collected weekly USD-INR exchange rates using `yfinance`, and imported gold prices from CSV.
2. **Data Preprocessing**: Cleaned, merged, and scaled data using `StandardScaler`.
3. **Modeling**: Trained a Linear Regression model, tuned with `RandomizedSearchCV`.
4. **Deployment**: Built a Gradio interface for real-time prediction.

---


