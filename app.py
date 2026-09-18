import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="📈 Stock Price Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 20px;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== LOAD MODEL & SCALER ====================
@st.cache_resource
def load_model():
    """Load pre-trained XGBoost model and scaler"""
    try:
        model = xgb.XGBClassifier()
        model.load_model('xgboost_model.json')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Make sure you have trained the model first using `xgboost_model.py`")
        return None, None

# ==================== CREATE FEATURES ====================
def create_features(data):
    """Create all technical indicators"""
    df = data.copy()
    
    # Moving Averages
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_100'] = df['Close'].rolling(window=100).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # EMA
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_SMA'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_SMA'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_SMA'] - (df['BB_Std'] * 2)
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / df['BB_Width']
    
    # ATR
    df['TR'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        )
    )
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    # Returns
    df['Return_1'] = df['Close'].pct_change(1) * 100
    df['Return_5'] = df['Close'].pct_change(5) * 100
    df['Return_10'] = df['Close'].pct_change(10) * 100
    df['Return_20'] = df['Close'].pct_change(20) * 100
    df['Return_50'] = df['Close'].pct_change(50) * 100
    
    # Volatility
    df['Volatility_5'] = df['Return_1'].rolling(window=5).std()
    df['Volatility_10'] = df['Return_1'].rolling(window=10).std()
    df['Volatility_20'] = df['Return_1'].rolling(window=20).std()
    
    # Volume
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    df['Volume_Change'] = df['Volume'].pct_change()
    
    # Price Range
    df['Price_Range'] = df['High'] - df['Low']
    df['Price_Range_Pct'] = ((df['High'] - df['Low']) / df['Close']) * 100
    df['Close_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
    
    # SMA Ratios
    df['SMA_Ratio_5_20'] = df['SMA_5'] / df['SMA_20']
    df['SMA_Ratio_10_50'] = df['SMA_10'] / df['SMA_50']
    df['SMA_Ratio_20_200'] = df['SMA_20'] / df['SMA_200']
    
    # Stochastic
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['Stochastic_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    df['Stochastic_D'] = df['Stochastic_K'].rolling(window=3).mean()
    
    # ADX
    df['Plus_DM'] = np.where(
        (df['High'].diff() > df['Low'].diff().abs()) & (df['High'].diff() > 0),
        df['High'].diff(), 0
    )
    df['Minus_DM'] = np.where(
        (df['Low'].diff().abs() > df['High'].diff()) & (df['Low'].diff() < 0),
        df['Low'].diff().abs(), 0
    )
    df['Plus_DI'] = 100 * (df['Plus_DM'].rolling(window=14).mean() / df['ATR'])
    df['Minus_DI'] = 100 * (df['Minus_DM'].rolling(window=14).mean() / df['ATR'])
    df['DX'] = 100 * abs(df['Plus_DI'] - df['Minus_DI']) / (df['Plus_DI'] + df['Minus_DI'])
    df['ADX'] = df['DX'].rolling(window=14).mean()
    
    return df

# ==================== FETCH DATA ====================
@st.cache_data(ttl=3600)
def fetch_stock_data(symbol, period='2y'):
    """Fetch stock data from Yahoo Finance"""
    try:
        data = yf.download(symbol, period=period, progress=False)
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

# ==================== MAKE PREDICTION ====================
def make_prediction(data, model, scaler):
    """Make prediction for latest data point"""
    try:
        # Create features
        df = create_features(data)
        df.dropna(inplace=True)
        
        # Get latest data
        latest = df.iloc[-1:].copy()
        
        # Select feature columns
        drop_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 
                     'TR']
        feature_cols = [col for col in df.columns if col not in drop_cols]
        
        X = latest[feature_cols]
        X_scaled = scaler.transform(X)
        
        # Predict
        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]
        
        return pred, proba, latest, feature_cols
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None, None, None

# ==================== MAIN APP ====================
def main():
    # Header
    st.markdown("# 📈 Stock Price Prediction with XGBoost")
    st.markdown("**Predict whether stock price will go UP or DOWN tomorrow**")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        stock_symbol = st.text_input(
            "Enter Stock Symbol",
            value="AAPL",
            help="e.g., AAPL, MSFT, GOOGL, TSLA"
        ).upper()
        
        period = st.selectbox(
            "Select Data Period",
            ["1y", "2y", "5y"],
            help="How much historical data to fetch"
        )
        
        st.markdown("---")
        st.markdown("## ℹ️ About")
        st.markdown("""
        This app uses **XGBoost** machine learning model 
        trained on historical stock data to predict price movements.
        
        **Features:**
        - 50+ Technical Indicators
        - Real-time Stock Data
        - Interactive Charts
        - Historical Predictions
        """)
    
    # Load model
    model, scaler = load_model()
    
    if model is None or scaler is None:
        st.error("❌ Model not found! Please train the model first.")
        st.info("Run `python xgboost_model.py` to train the model")
        return
    
    # Fetch data
    with st.spinner(f"📥 Fetching {stock_symbol} data..."):
        data = fetch_stock_data(stock_symbol, period)
    
    if data is None:
        st.error(f"Could not fetch data for {stock_symbol}")
        return
    
    # Make prediction
    with st.spinner("🤖 Making prediction..."):
        pred, proba, latest, feature_cols = make_prediction(data, model, scaler)
    
    if pred is None:
        st.error("Could not make prediction")
        return
    
    # ==================== DISPLAY RESULTS ====================
    st.markdown("---")
    
    # Prediction Results
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.metric("Current Price", f"${latest['Close'].values[0]:.2f}")
    
    with col2:
        if pred == 1:
            st.success("## 📈 PRICE WILL GO UP")
            pred_text = "UP"
            color = "green"
        else:
            st.error("## 📉 PRICE WILL GO DOWN")
            pred_text = "DOWN"
            color = "red"
    
    with col3:
        st.metric("Confidence", f"{max(proba)*100:.2f}%")
    
    # ==================== TABS ====================
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Chart", "📈 Indicators", "📋 Data", "🎯 Details"])
    
    # TAB 1: PRICE CHART
    with tab1:
        st.markdown("### Price History with Moving Averages")
        
        # Create features for chart
        df_chart = create_features(data).dropna()
        
        fig = go.Figure()
        
        # Close price
        fig.add_trace(go.Scatter(
            x=df_chart['Date'], y=df_chart['Close'],
            name='Close Price', line=dict(color='blue', width=2)
        ))
        
        # SMA lines
        fig.add_trace(go.Scatter(
            x=df_chart['Date'], y=df_chart['SMA_20'],
            name='SMA 20', line=dict(color='orange', width=1, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=df_chart['Date'], y=df_chart['SMA_50'],
            name='SMA 50', line=dict(color='red', width=1, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=df_chart['Date'], y=df_chart['SMA_200'],
            name='SMA 200', line=dict(color='green', width=1, dash='dash')
        ))
        
        fig.update_layout(
            title=f"{stock_symbol} Price Chart",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode='x unified',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: TECHNICAL INDICATORS
    with tab2:
        col1, col2 = st.columns(2)
        
        # RSI
        with col1:
            st.markdown("### RSI (Relative Strength Index)")
            df_chart = create_features(data).dropna()
            
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(
                x=df_chart['Date'], y=df_chart['RSI'],
                name='RSI', line=dict(color='purple')
            ))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", 
                            annotation_text="Overbought")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green",
                            annotation_text="Oversold")
            fig_rsi.update_layout(height=400, hovermode='x unified')
            st.plotly_chart(fig_rsi, use_container_width=True)
            
            rsi_value = latest['RSI'].values[0]
            if rsi_value > 70:
                st.warning(f"📊 RSI: {rsi_value:.2f} - OVERBOUGHT")
            elif rsi_value < 30:
                st.info(f"📊 RSI: {rsi_value:.2f} - OVERSOLD")
            else:
                st.success(f"📊 RSI: {rsi_value:.2f} - NEUTRAL")
        
        # MACD
        with col2:
            st.markdown("### MACD (Moving Average Convergence Divergence)")
            
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(
                x=df_chart['Date'], y=df_chart['MACD'],
                name='MACD', line=dict(color='blue')
            ))
            fig_macd.add_trace(go.Scatter(
                x=df_chart['Date'], y=df_chart['Signal_Line'],
                name='Signal Line', line=dict(color='red')
            ))
            fig_macd.add_trace(go.Bar(
                x=df_chart['Date'], y=df_chart['MACD_Histogram'],
                name='Histogram', marker=dict(color='gray', opacity=0.5)
            ))
            fig_macd.update_layout(height=400, hovermode='x unified')
            st.plotly_chart(fig_macd, use_container_width=True)
            
            macd_val = latest['MACD'].values[0]
            signal_val = latest['Signal_Line'].values[0]
            st.info(f"📈 MACD: {macd_val:.4f} | Signal: {signal_val:.4f}")
        
        # Bollinger Bands
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Bollinger Bands")
            
            fig_bb = go.Figure()
            fig_bb.add_trace(go.Scatter(
                x=df_chart['Date'], y=df_chart['Close'],
                name='Close', line=dict(color='blue')
            ))
            fig_bb.add_trace(go.Scatter(
                x=df_chart['Date'], y=df_chart['BB_Upper'],
                name='Upper Band', line=dict(color='red', dash='dash')
            ))
            fig_bb.add_trace(go.Scatter(
                x=df_chart['Date'], y=df_chart['BB_Lower'],
                name='Lower Band', line=dict(color='green', dash='dash'),
                fill='tonexty'
            ))
            fig_bb.update_layout(height=400, hovermode='x unified')
            st.plotly_chart(fig_bb, use_container_width=True)
        
        with col2:
            st.markdown("### Volume Analysis")
            
            df_chart = create_features(data).dropna()
            
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Bar(
                x=df_chart['Date'], y=df_chart['Volume'],
                name='Volume', marker=dict(color='teal', opacity=0.7)
            ))
            fig_vol.add_trace(go.Scatter(
                x=df_chart['Date'], y=df_chart['Volume_SMA'],
                name='Volume SMA', line=dict(color='red')
            ))
            fig_vol.update_layout(height=400, hovermode='x unified')
            st.plotly_chart(fig_vol, use_container_width=True)
    
    # TAB 3: PREDICTION PROBABILITY
    with tab3:
        st.markdown("### Prediction Confidence")
        
        # Create gauge chart
        fig_gauge = go.Figure(go.Indicator(
            domain={'x': [0, 1], 'y': [0, 1]},
            value=proba[1] * 100,
            mode="gauge+number+delta",
            title={'text': "Probability of UP"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgray"},
                    {'range': [30, 70], 'color': "gray"},
                    {'range': [70, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig_gauge.update_layout(height=400)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Probability DOWN", f"{proba[0]*100:.2f}%", delta=f"{proba[0]*100-50:.2f}%")
        with col2:
            st.metric("Probability UP", f"{proba[1]*100:.2f}%", delta=f"{proba[1]*100-50:.2f}%")
    
    # TAB 4: TECHNICAL DETAILS
    with tab4:
        st.markdown("### Latest Technical Indicators")
        
        indicators_dict = {
            'RSI': f"{latest['RSI'].values[0]:.2f}",
            'MACD': f"{latest['MACD'].values[0]:.4f}",
            'Signal Line': f"{latest['Signal_Line'].values[0]:.4f}",
            'ATR': f"{latest['ATR'].values[0]:.2f}",
            'Volume Ratio': f"{latest['Volume_Ratio'].values[0]:.2f}",
            'Price Range %': f"{latest['Price_Range_Pct'].values[0]:.2f}%",
            'Close Position': f"{latest['Close_Position'].values[0]:.2%}",
            'Stochastic K': f"{latest['Stochastic_K'].values[0]:.2f}",
            'ADX': f"{latest['ADX'].values[0]:.2f}",
        }
        
        # Display as columns
        col1, col2, col3 = st.columns(3)
        
        indicators_list = list(indicators_dict.items())
        for i, (key, value) in enumerate(indicators_list):
            if i % 3 == 0:
                col = col1
            elif i % 3 == 1:
                col = col2
            else:
                col = col3
            
            col.metric(key, value)
        
        # Raw data
        st.markdown("### Raw Data (Last 10 Days)")
        display_df = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].tail(10).copy()
        display_df['Close'] = display_df['Close'].apply(lambda x: f"${x:.2f}")
        display_df['Volume'] = display_df['Volume'].apply(lambda x: f"{x:,.0f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # ==================== FOOTER ====================
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"📅 Last Update: {latest['Date'].values[0]}")
    
    with col2:
        st.info(f"📊 Data Points: {len(data)}")
    
    with col3:
        st.info("🤖 Model: XGBoost Classifier")
    
    st.markdown("""
    ---
    **Disclaimer:** This is for educational purposes only. 
    Do not use for real trading without proper risk management.
    """)

if __name__ == "__main__":
    main()