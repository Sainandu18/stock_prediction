import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("REAL-TIME STOCK PRICE PREDICTION")
print("="*70)

# ==================== LOAD TRAINED MODEL ====================
print("\n[1] Loading Trained Model...")
xgb_model = xgb.XGBClassifier()
xgb_model.load_model('xgboost_model.json')
scaler = joblib.load('scaler.pkl')
print("✓ Model and scaler loaded")

# ==================== GET LATEST DATA ====================
print("\n[2] Fetching Latest Stock Data...")
stock_symbol = "AAPL"

# Download last 300 days of data
df = yf.download(stock_symbol, start='2023-01-01', progress=False)
# Fix yfinance MultiIndex columns
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df.reset_index(inplace=True)

print(f"✓ Downloaded {len(df)} records for {stock_symbol}")

# ==================== CREATE FEATURES ====================
print("\n[3] Creating Technical Indicators...")

# Copy the existing dataframe
data = df.copy()

# Create all features (same as training)
data['SMA_5'] = data['Close'].rolling(window=5).mean()
data['SMA_10'] = data['Close'].rolling(window=10).mean()
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['SMA_100'] = data['Close'].rolling(window=100).mean()
data['SMA_200'] = data['Close'].rolling(window=200).mean()

data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()

data['MACD'] = data['EMA_12'] - data['EMA_26']
data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
data['MACD_Histogram'] = data['MACD'] - data['Signal_Line']

delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

data['BB_SMA'] = data['Close'].rolling(window=20).mean()
data['BB_Std'] = data['Close'].rolling(window=20).std()
data['BB_Upper'] = data['BB_SMA'] + (data['BB_Std'] * 2)
data['BB_Lower'] = data['BB_SMA'] - (data['BB_Std'] * 2)
data['BB_Width'] = data['BB_Upper'] - data['BB_Lower']
data['BB_Position'] = (data['Close'] - data['BB_Lower']) / data['BB_Width']

data['TR'] = np.maximum(
    data['High'] - data['Low'],
    np.maximum(
        abs(data['High'] - data['Close'].shift(1)),
        abs(data['Low'] - data['Close'].shift(1))
    )
)
data['ATR'] = data['TR'].rolling(window=14).mean()

data['Return_1'] = data['Close'].pct_change(1) * 100
data['Return_5'] = data['Close'].pct_change(5) * 100
data['Return_10'] = data['Close'].pct_change(10) * 100
data['Return_20'] = data['Close'].pct_change(20) * 100
data['Return_50'] = data['Close'].pct_change(50) * 100

data['Volatility_5'] = data['Return_1'].rolling(window=5).std()
data['Volatility_10'] = data['Return_1'].rolling(window=10).std()
data['Volatility_20'] = data['Return_1'].rolling(window=20).std()

data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
data['Volume_Change'] = data['Volume'].pct_change()

data['Price_Range'] = data['High'] - data['Low']
data['Price_Range_Pct'] = ((data['High'] - data['Low']) / data['Close']) * 100
data['Close_Position'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'])

data['SMA_Ratio_5_20'] = data['SMA_5'] / data['SMA_20']
data['SMA_Ratio_10_50'] = data['SMA_10'] / data['SMA_50']
data['SMA_Ratio_20_200'] = data['SMA_20'] / data['SMA_200']

low_14 = data['Low'].rolling(window=14).min()
high_14 = data['High'].rolling(window=14).max()
data['Stochastic_K'] = 100 * ((data['Close'] - low_14) / (high_14 - low_14))
data['Stochastic_D'] = data['Stochastic_K'].rolling(window=3).mean()

data['Plus_DM'] = np.where(
    (data['High'].diff() > data['Low'].diff().abs()) & (data['High'].diff() > 0),
    data['High'].diff(),
    0
)
data['Minus_DM'] = np.where(
    (data['Low'].diff().abs() > data['High'].diff()) & (data['Low'].diff() < 0),
    data['Low'].diff().abs(),
    0
)
data['Plus_DI'] = 100 * (data['Plus_DM'].rolling(window=14).mean() / data['ATR'])
data['Minus_DI'] = 100 * (data['Minus_DM'].rolling(window=14).mean() / data['ATR'])
data['DX'] = 100 * abs(data['Plus_DI'] - data['Minus_DI']) / (data['Plus_DI'] + data['Minus_DI'])
data['ADX'] = data['DX'].rolling(window=14).mean()

# Remove rows with NaN
data.dropna(inplace=True)

print("✓ Features created")

# ==================== PREPARE FOR PREDICTION ====================
print("\n[4] Preparing Data for Prediction...")

# Get latest data point
latest_data = data.iloc[-1:].copy()

# Select EXACTLY the same features used when the scaler was trained
feature_cols = list(scaler.feature_names_in_)

print(f"✓ Number of features expected by scaler: {len(feature_cols)}")

# Check whether any required features are missing
missing_features = [
    col for col in feature_cols
    if col not in latest_data.columns
]

if missing_features:
    print("\n❌ Missing features:")
    for col in missing_features:
        print(f"   - {col}")
    exit()

# Select features in EXACT same order as training
X_latest = latest_data[feature_cols]

print("✓ Features selected in correct order")

# Scale features
X_latest_scaled = scaler.transform(X_latest)

print("✓ Data prepared for prediction")

# ==================== MAKE PREDICTION ====================
print("\n[5] Making Prediction...")

# Predict
prediction = xgb_model.predict(X_latest_scaled)[0]
prediction_proba = xgb_model.predict_proba(X_latest_scaled)[0]

print(f"\n{'='*70}")
print(f"PREDICTION FOR {stock_symbol}")
print(f"{'='*70}")
print(f"Current Price: ${latest_data['Close'].values[0]:.2f}")
print(f"Current Date: {latest_data['Date'].values[0]}")
print(f"\nPrediction: {'📈 PRICE WILL GO UP' if prediction == 1 else '📉 PRICE WILL GO DOWN'}")
print(f"Confidence (Down): {prediction_proba[0]*100:.2f}%")
print(f"Confidence (Up): {prediction_proba[1]*100:.2f}%")

# Additional technical indicators
print(f"\nTechnical Indicators:")
print(f"  RSI: {latest_data['RSI'].values[0]:.2f}", end="")
if latest_data['RSI'].values[0] > 70:
    print(" (Overbought ⚠)")
elif latest_data['RSI'].values[0] < 30:
    print(" (Oversold ⚠)")
else:
    print(" (Neutral ✓)")

print(f"  MACD: {latest_data['MACD'].values[0]:.4f}")
print(f"  Bollinger Bands Position: {latest_data['BB_Position'].values[0]:.2%}")
print(f"  Volume Ratio: {latest_data['Volume_Ratio'].values[0]:.2f}")
print(f"{'='*70}\n")

# ==================== HISTORICAL PREDICTIONS ====================
print("[6] Historical Predictions (Last 30 Days)...\n")

# Get last 30 data points
last_30 = data.tail(30).copy()
X_last_30 = last_30[feature_cols]
X_last_30_scaled = scaler.transform(X_last_30)

predictions_30 = xgb_model.predict(X_last_30_scaled)
probabilities_30 = xgb_model.predict_proba(X_last_30_scaled)

# Create results dataframe
results = pd.DataFrame({
    'Date': last_30['Date'].values,
    'Close': last_30['Close'].values,
    'Prediction': ['UP' if p == 1 else 'DOWN' for p in predictions_30],
    'Confidence_Up': probabilities_30[:, 1],
    'Confidence_Down': probabilities_30[:, 0]
})

print(results.to_string(index=False))

# Save predictions
results.to_csv('predictions_last_30_days.csv', index=False)
print("\n✓ Predictions saved to 'predictions_last_30_days.csv'")

# ==================== MODEL STATISTICS ====================
print("\n[7] Model Information\n")
print(f"Model Type: XGBoost Classifier")
print(f"Number of Features: {len(feature_cols)}")
print(f"Number of Trees: {xgb_model.n_estimators}")
print(f"Max Depth: {xgb_model.max_depth}")
print(f"Learning Rate: {xgb_model.learning_rate}")

print("\n" + "="*70)
print("✓ PREDICTION COMPLETE!")
print("="*70)