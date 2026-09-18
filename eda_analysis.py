import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("EXPLORATORY DATA ANALYSIS (EDA)")
print("="*70)

# Load data
df = pd.read_csv('apple_stock_complete.csv')
print(f"\n✓ Data loaded: {df.shape[0]} records, {df.shape[1]} columns\n")

# ==================== 1. BASIC STATISTICS ====================
print("[1] BASIC STATISTICS\n")
print(df[['Close', 'Volume', 'Target']].describe())

# ==================== 2. TARGET DISTRIBUTION ====================
print("\n[2] TARGET DISTRIBUTION\n")
target_counts = df['Target'].value_counts()
print(f"Up days (1): {target_counts[1]} ({target_counts[1]/len(df)*100:.2f}%)")
print(f"Down days (0): {target_counts[0]} ({target_counts[0]/len(df)*100:.2f}%)")

# ==================== 3. MISSING VALUES ====================
print("\n[3] MISSING VALUES\n")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("✓ No missing values found!")
else:
    print(missing[missing > 0])

# ==================== 4. PRICE TRENDS ====================
print("\n[4] PRICE TRENDS\n")
print(f"Starting Price: ${df['Close'].iloc[0]:.2f}")
print(f"Ending Price: ${df['Close'].iloc[-1]:.2f}")
print(f"Price Change: ${df['Close'].iloc[-1] - df['Close'].iloc[0]:.2f}")
print(f"Percentage Change: {((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100):.2f}%")

# ==================== 5. VISUALIZATIONS ====================
print("\n[5] CREATING VISUALIZATIONS...\n")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 12)

# Create subplots
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle('Stock Price Analysis - AAPL', fontsize=16, fontweight='bold')

# 1. Closing Price Over Time
axes[0, 0].plot(df['Date'], df['Close'], color='blue', linewidth=2)
axes[0, 0].set_title('Closing Price Over Time', fontweight='bold')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Price ($)')
axes[0, 0].grid(True, alpha=0.3)

# 2. Volume Over Time
axes[0, 1].bar(df['Date'], df['Volume'], color='green', alpha=0.6)
axes[0, 1].set_title('Trading Volume Over Time', fontweight='bold')
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('Volume')
axes[0, 1].grid(True, alpha=0.3)

# 3. Daily Returns Distribution
axes[1, 0].hist(df['Return_1'].dropna(), bins=50, color='purple', alpha=0.7, edgecolor='black')
axes[1, 0].set_title('Daily Returns Distribution', fontweight='bold')
axes[1, 0].set_xlabel('Return (%)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].grid(True, alpha=0.3)

# 4. Target Distribution
target_labels = ['Down (0)', 'Up (1)']
colors = ['red', 'green']
axes[1, 1].pie(df['Target'].value_counts().values, labels=target_labels, autopct='%1.1f%%',
               colors=colors, startangle=90)
axes[1, 1].set_title('Target Variable Distribution', fontweight='bold')

# 5. Close Price vs Moving Averages
axes[2, 0].plot(df['Date'], df['Close'], label='Close', linewidth=2, alpha=0.7)
axes[2, 0].plot(df['Date'], df['SMA_20'], label='SMA 20', linewidth=1.5, alpha=0.7)
axes[2, 0].plot(df['Date'], df['SMA_50'], label='SMA 50', linewidth=1.5, alpha=0.7)
axes[2, 0].plot(df['Date'], df['SMA_200'], label='SMA 200', linewidth=1.5, alpha=0.7)
axes[2, 0].set_title('Price with Moving Averages', fontweight='bold')
axes[2, 0].set_xlabel('Date')
axes[2, 0].set_ylabel('Price ($)')
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

# 6. RSI Indicator
axes[2, 1].plot(df['Date'], df['RSI'], color='orange', linewidth=2)
axes[2, 1].axhline(y=70, color='red', linestyle='--', label='Overbought (70)')
axes[2, 1].axhline(y=30, color='green', linestyle='--', label='Oversold (30)')
axes[2, 1].fill_between(range(len(df)), 30, 70, alpha=0.1, color='gray')
axes[2, 1].set_title('RSI (Relative Strength Index)', fontweight='bold')
axes[2, 1].set_xlabel('Date')
axes[2, 1].set_ylabel('RSI Value')
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved as 'eda_analysis.png'")
plt.show()

# ==================== 6. CORRELATION ANALYSIS ====================
print("\n[6] CORRELATION ANALYSIS\n")

# Select numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
numeric_df = df[numeric_cols].copy()

# Calculate correlation with target
correlation = numeric_df.corr()['Target'].sort_values(ascending=False)
print("Top 15 Features Correlated with Target:\n")
print(correlation.head(16))

# Correlation heatmap
plt.figure(figsize=(12, 8))
top_features = correlation.head(16).index.tolist()
sns.heatmap(numeric_df[top_features].corr(), annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=1)
plt.title('Correlation Matrix - Top Features', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
print("\n✓ Correlation heatmap saved as 'correlation_matrix.png'")
plt.show()

# ==================== 7. FEATURE STATISTICS ====================
print("\n[7] FEATURE STATISTICS\n")
print(df[numeric_cols].describe().T[['mean', 'std', 'min', 'max']])

print("\n" + "="*70)
print("✓ EDA COMPLETE!")
print("="*70)