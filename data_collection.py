import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("STOCK PRICE PREDICTION - DATA COLLECTION & FEATURE ENGINEERING")
print("="*70)

# ==================== CONFIGURATION ====================
STOCK_SYMBOL = "AAPL"
START_DATE = "2019-01-01"
END_DATE = "2024-01-01"
OUTPUT_FILE = "apple_stock_complete.csv"

print(f"\n[STEP 1] Downloading {STOCK_SYMBOL} stock data...")
print(f"Date range: {START_DATE} to {END_DATE}\n")

# ==================== STEP 1: DOWNLOAD DATA ====================
try:
    df = yf.download(STOCK_SYMBOL, start=START_DATE, end=END_DATE, progress=False)
    print(f"✓ Downloaded {len(df)} records\n")
except Exception as e:
    print(f"✗ Error downloading data: {e}")
    exit()


# Fix yfinance MultiIndex columns
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Reset index to make Date a column
df.reset_index(inplace=True)

# Save the dataset
df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved dataset to: {OUTPUT_FILE}")

# Display basic info
print("First few rows:")
print(df.head())
print(f"\nDataset shape: {df.shape}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}\n")

# ==================== STEP 2: CREATE TARGET VARIABLE ====================
print("[STEP 2] Creating Target Variable...")
print("Target: 1 if price goes UP tomorrow, 0 if goes DOWN\n")

# Create future price column (next day's closing price)
df['Price_Tomorrow'] = df['Close'].shift(-1)

# Target: 1 if tomorrow's price > today's price, else 0
df['Target'] = (df['Price_Tomorrow'] > df['Close']).astype(int)

# Calculate actual price change
df['Price_Change'] = df['Price_Tomorrow'] - df['Close']
df['Price_Change_Pct'] = ((df['Price_Tomorrow'] - df['Close']) / df['Close'] * 100)

print(f"Days with price increase: {df['Target'].sum()}")
print(f"Days with price decrease: {(1 - df['Target']).sum()}")
print(f"Class balance: {df['Target'].mean()*100:.2f}% UP, {(1-df['Target']).mean()*100:.2f}% DOWN\n")

# ==================== STEP 3: FEATURE ENGINEERING ====================
print("[STEP 3] Creating Technical Indicators...\n")

# -------- 3.1: Moving Averages --------
print("  • Creating Moving Averages...")
df['SMA_5'] = df['Close'].rolling(window=5).mean()
df['SMA_10'] = df['Close'].rolling(window=10).mean()
df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['SMA_50'] = df['Close'].rolling(window=50).mean()
df['SMA_100'] = df['Close'].rolling(window=100).mean()
df['SMA_200'] = df['Close'].rolling(window=200).mean()

# -------- 3.2: Exponential Moving Average --------
print("  • Creating Exponential Moving Averages...")
df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

# -------- 3.3: MACD (Moving Average Convergence Divergence) --------
print("  • Creating MACD...")
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']

# -------- 3.4: RSI (Relative Strength Index) --------
print("  • Creating RSI...")
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# -------- 3.5: Bollinger Bands --------
print("  • Creating Bollinger Bands...")
df['BB_SMA'] = df['Close'].rolling(window=20).mean()
df['BB_Std'] = df['Close'].rolling(window=20).std()
df['BB_Upper'] = df['BB_SMA'] + (df['BB_Std'] * 2)
df['BB_Lower'] = df['BB_SMA'] - (df['BB_Std'] * 2)
df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
df['BB_Position'] = (df['Close'] - df['BB_Lower']) / df['BB_Width']

# -------- 3.6: Average True Range (ATR) --------
print("  • Creating ATR...")
df['TR'] = np.maximum(
    df['High'] - df['Low'],
    np.maximum(
        abs(df['High'] - df['Close'].shift(1)),
        abs(df['Low'] - df['Close'].shift(1))
    )
)
df['ATR'] = df['TR'].rolling(window=14).mean()

# -------- 3.7: Daily Returns --------
print("  • Creating Returns...")
df['Return_1'] = df['Close'].pct_change(1) * 100
df['Return_5'] = df['Close'].pct_change(5) * 100
df['Return_10'] = df['Close'].pct_change(10) * 100
df['Return_20'] = df['Close'].pct_change(20) * 100
df['Return_50'] = df['Close'].pct_change(50) * 100

# -------- 3.8: Volatility --------
print("  • Creating Volatility...")
df['Volatility_5'] = df['Return_1'].rolling(window=5).std()
df['Volatility_10'] = df['Return_1'].rolling(window=10).std()
df['Volatility_20'] = df['Return_1'].rolling(window=20).std()

# -------- 3.9: Volume Features --------
print("  • Creating Volume Features...")
df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
df['Volume_Change'] = df['Volume'].pct_change()

# -------- 3.10: Price Range Features --------
print("  • Creating Price Range Features...")
df['Price_Range'] = df['High'] - df['Low']
df['Price_Range_Pct'] = ((df['High'] - df['Low']) / df['Close']) * 100
df['Close_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])

# -------- 3.11: SMA Ratios --------
print("  • Creating SMA Ratios...")
df['SMA_Ratio_5_20'] = df['SMA_5'] / df['SMA_20']
df['SMA_Ratio_10_50'] = df['SMA_10'] / df['SMA_50']
df['SMA_Ratio_20_200'] = df['SMA_20'] / df['SMA_200']

# -------- 3.12: Stochastic Oscillator --------
print("  • Creating Stochastic Oscillator...")
low_14 = df['Low'].rolling(window=14).min()
high_14 = df['High'].rolling(window=14).max()
df['Stochastic_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
df['Stochastic_D'] = df['Stochastic_K'].rolling(window=3).mean()

# -------- 3.13: ADX (Average Directional Index) --------
print("  • Creating ADX...")
df['Plus_DM'] = np.where(
    (df['High'].diff() > df['Low'].diff().abs()) & (df['High'].diff() > 0),
    df['High'].diff(),
    0
)
df['Minus_DM'] = np.where(
    (df['Low'].diff().abs() > df['High'].diff()) & (df['Low'].diff() < 0),
    df['Low'].diff().abs(),
    0
)
df['Plus_DI'] = 100 * (df['Plus_DM'].rolling(window=14).mean() / df['ATR'])
df['Minus_DI'] = 100 * (df['Minus_DM'].rolling(window=14).mean() / df['ATR'])
df['DX'] = 100 * abs(df['Plus_DI'] - df['Minus_DI']) / (df['Plus_DI'] + df['Minus_DI'])
df['ADX'] = df['DX'].rolling(window=14).mean()

print("✓ All technical indicators created!\n")

# ==================== STEP 4: DATA CLEANING ====================
print("[STEP 4] Data Cleaning...")

# Drop rows with NaN values (from moving averages)
initial_rows = len(df)
df.dropna(inplace=True)
print(f"  • Removed {initial_rows - len(df)} rows with NaN values")

# Remove the last row (no target value)
df = df[:-1]
print(f"  • Removed last row (no target value)")

# Check for remaining NaN values
nan_count = df.isnull().sum().sum()
if nan_count > 0:
    print(f"  ⚠ Warning: {nan_count} NaN values remaining")
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

print(f"✓ Final dataset shape: {df.shape}")
print(f"  • Total features: {len(df.columns) - 5}")  # excluding Date and target columns
print(f"  • Data range: {df['Date'].min()} to {df['Date'].max()}\n")

# ==================== STEP 5: DATA STATISTICS ====================
print("[STEP 5] Dataset Statistics:\n")
print(f"Close Price - Min: ${df['Close'].min():.2f}, Max: ${df['Close'].max():.2f}, Mean: ${df['Close'].mean():.2f}")
print(f"Volume - Min: {df['Volume'].min():.0f}, Max: {df['Volume'].max():.0f}, Mean: {df['Volume'].mean():.0f}")
print(f"Target Distribution:\n{df['Target'].value_counts()}\n")

# ==================== STEP 6: SAVE DATA ====================
print("[STEP 6] Saving Data...")
df.to_csv(OUTPUT_FILE, index=False)
print(f"✓ Data saved to '{OUTPUT_FILE}'\n")

# ==================== STEP 7: FEATURE LIST ====================
print("[STEP 7] Feature Summary:\n")
feature_columns = [col for col in df.columns if col not in 
                   ['Date', 'Target', 'Price_Tomorrow', 'Price_Change', 'Price_Change_Pct']]

print(f"Total Features: {len(feature_columns)}\n")
print("Features:")
for i, col in enumerate(feature_columns, 1):
    print(f"  {i:2d}. {col}")

print("\n" + "="*70)
print("✓ DATA PREPARATION COMPLETE!")
print("="*70)

# Display final dataframe
print("\nFinal Dataset Preview:")
print(df.head(10))
print("\nData Info:")
print(df.info())

# Save feature names for later use
with open('feature_names.txt', 'w') as f:
    f.write('\n'.join(feature_columns))

print(f"\n✓ Feature names saved to 'feature_names.txt'")