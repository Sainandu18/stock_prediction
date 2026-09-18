# 📈 Stock Price Prediction with XGBoost

An end-to-end machine learning application that analyzes historical stock market data, generates **50+ technical indicators**, and uses an **XGBoost classification model** to predict whether a stock price is likely to move **UP or DOWN**. The project includes interactive data visualization and is deployed as a **Streamlit web application** for real-time analysis.

## 🚀 Key Features

### 📊 Technical Analysis

The system calculates 50+ technical indicators to capture price trends, momentum, volatility, and trading activity, including:

* Simple Moving Average (SMA)
* Exponential Moving Average (EMA)
* Relative Strength Index (RSI)
* Moving Average Convergence Divergence (MACD)
* Bollinger Bands
* Average True Range (ATR)
* Average Directional Index (ADX)
* Stochastic Oscillator
* Volume indicators
* Volatility metrics

### 🤖 Machine Learning Prediction

* **Algorithm:** XGBoost
* **Problem Type:** Binary Classification
* **Target:** Stock price movement — `UP` / `DOWN`
* **Training Data:** 5 years of historical market data
* **Features:** Technical indicators, price and volume-based features
* **Evaluation:** Accuracy and classification metrics
* **Model Output:** Predicted direction with confidence/probability

### 📈 Interactive Dashboard

The Streamlit application provides:

* Real-time stock price charts
* Historical price analysis
* Technical indicator visualizations
* Current market trend analysis
* UP/DOWN prediction
* Prediction probability/confidence
* Interactive stock/ticker selection

## 🏗️ Project Workflow

```text
Historical Stock Data
        ↓
Data Cleaning & Preprocessing
        ↓
Technical Indicator Generation
        ↓
Feature Engineering
        ↓
Train/Test Split
        ↓
XGBoost Classification
        ↓
Model Evaluation
        ↓
Stock Movement Prediction
        ↓
Streamlit Dashboard
```

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **XGBoost**
* **Matplotlib / Plotly**
* **Technical Analysis**
* **Streamlit**
* **Yahoo Finance API**
* **Git & GitHub**

## 📂 Project Structure

```text
stock-prediction/
│
├── app.py
├── model/
│   └── xgboost_model.pkl
├── data/
│   └── stock_data.csv
├── notebooks/
│   └── stock_prediction.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/stock-prediction.git

# Navigate to the project
cd stock-prediction

# Install dependencies
pip install -r requirements.txt
```

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## ☁️ Deployment

The application can be deployed using **Streamlit Community Cloud**, allowing users to interact with the stock analysis and prediction dashboard through a web browser.

## ⚠️ Disclaimer

This project is intended for **educational and research purposes only**. Stock-market predictions are inherently uncertain, and model predictions should not be considered financial or investment advice.
