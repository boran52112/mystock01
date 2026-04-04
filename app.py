import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from FinMind.data import DataLoader
from datetime import datetime, timedelta

# --- 網頁配置 ---
st.set_page_config(page_title="AI 專家級 9 大指標分析系統", layout="wide")

st.markdown("""
    <style>
    .report-card { background-color: #ffffff; padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4; box-shadow: 0 2px 4px rgba(0,0,0,0.05); min-height: 200px; }
    .strategy-card { background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #ff4b4b; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 AI 專家級 9 大指標技術分析系統")

# --- 取得台股中文名稱輔助函式 ---
@st.cache_data(ttl=86400) # 股票清單一天抓一次即可
def get_taiwan_stock_name(code):
    """透過 FinMind 取得台股中文名稱"""
    try:
        dl = DataLoader()
        df_info = dl.taiwan_stock_info()
        # 過濾出對應代號的名稱
        stock_item = df_info[df_info['stock_id'] == code]
        if not stock_item.empty:
            return stock_item.iloc[0]['stock_name']
    except:
        pass
    return None

# --- 資料抓取 (DB1) ---
@st.cache_data(ttl=3600)
def fetch_complete_data(code):
    """抓取股價與台灣特有籌碼數據"""
    df_price, actual_sym, final_name = None, None, f"股票 {code}"
    
    # 優先嘗試取得中文名稱
    chinese_name = get_taiwan_stock_name(code)

    for suffix in [".TW", ".TWO"]:
        try:
            temp_sym = f"{code}{suffix}"
            ticker_obj = yf.Ticker(temp_sym)
            df = ticker_obj.history(period="1y")
            
            if not df.empty:
                # 若 FinMind 沒抓到中文名，才用 yfinance 的名稱
                if chinese_name:
                    final_name = chinese_name
                else:
                    final_name = ticker_obj.info.get('shortName') or temp_sym
                
                df.reset_index(inplace=True)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
                df_price, actual_sym = df, temp_sym
                break
        except:
            continue
            
    if df_price is None:
        return None, None, None, False

    # 抓取籌碼
    try:
        dl = DataLoader()
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        df_inst = dl.taiwan_stock_institutional_investors(stock_id=code, start_date=start_date)
        df_margin = dl.taiwan_stock_margin_purchase_short_sale(stock_id=code, start_date=start_date)
        
        df_pivot = pd.DataFrame()
        if not df_inst.empty:
            df_pivot = df_inst.pivot_table(index='date', columns='name', values='buy_sell', aggfunc='sum').reset_index()
            df_pivot.rename(columns={'date': 'Date'}, inplace=True)
            f_col, t_col = '外陸資買賣超股數(不含外資自營商)', '投信買賣超股數'
            df_pivot['Foreign_Buy'] = df_pivot[f_col] / 1000 if f_col in df_pivot.columns else 0
            df_pivot['Trust_Buy'] = df_pivot[t_col] / 1000 if t_col in df_pivot.columns else 0
            df_pivot = df_pivot[['Date', 'Foreign_Buy', 'Trust_Buy']]

        if not df_margin.empty:
            df_margin = df_margin[['date', 'MarginPurchaseTodayBalance', 'ShortSaleTodayBalance']]
            df_margin.rename(columns={'date': 'Date', 'MarginPurchaseTodayBalance': 'Margin_Bal', 'Short_Bal': 'ShortSaleTodayBalance'}, inplace=True)
            
        if not df_pivot.empty: df_price = pd.merge(df_price, df_pivot, on='Date', how='left')
        if not df_margin.empty: df_price = pd.merge(df_price, df_margin, on='Date', how='left')
        df_price.fillna(method='ffill', inplace=True)
        df_price.fillna(0, inplace=True)
        return df_price, actual_sym, final_name, True
    except:
        return df_price, actual_sym, final_name, False

# --- 技術分析與專家建議 (DB2 & DB3) ---
def generate_ta_db2(df):
    db2 = df.copy()
    db2['SMA_5'] = db2['Close'].rolling(5).mean()
    db2['SMA_20'] = db2['Close'].rolling(20).mean()
    delta = db2['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    db2['RSI'] = 100 - (100 / (1 + gain/loss))
    exp1 = db2['Close'].ewm(span=12).mean()
    exp2 = db2['Close'].ewm(span=26).mean()
    db2['MACD'] = exp1 - exp2
    db2['MACD_S'] = db2['MACD'].ewm(span=9).mean()
    db2['MACD_H'] = db2['MACD'] - db2['MACD_S']
    db2['BB_Mid'] = db2['Close'].rolling(20).mean()
    std = db2['Close'].rolling(20).std()
    db2['BB_Up'] = db2['BB_Mid'] + 2*std
    db2['BB_Low'] = db2['BB_Mid'] - 2*std
    l9, h9 = db2['Low'].rolling(9).min(), db2['High'].rolling(9).max()
    rsv = (db2['Close'] - l9) / (h9 - l9) * 100
    db2['K'] = rsv.ewm(com=2).mean()
    db2['D'] = db2['K'].ewm(com=2).mean()
    db2['Prev_Close'] = db2['Close'].shift(1)
    db2['Prev_High'] = db2['High'].shift(1)
    db2['Prev_Low'] = db2['Low'].shift(1)
    if 'Margin_Bal' in db2.columns:
        db2['Margin_Diff'] = db2['Margin_Bal'].diff()
    return db2.dropna()

def generate_expert_db3(db2, has_chip):
    latest = db2.iloc[-1]
    prev = db2.iloc[-2]
    analysis = []
    bull, bear = 0, 0
    
    # 簡化判斷邏輯
    checks = [
        ("均線", latest['SMA_5'] > latest['SMA_20'], "多頭排列", "死叉偏弱"),
        ("動能", latest['RSI'] < 30, "超賣支撐", "RSI 正常"), # 簡化
        ("MACD", latest['MACD_H'] > 0, "波段向上", "波段向下"),
        ("布林", latest['Close'] > latest['BB_Up'], "強勢噴發", "通道運行"),
        ("KD", latest['K'] < 20, "低檔超賣", "KD 正常"),
    ]
    # (此處為了精簡僅列範例，實際會跑完 9 個指標)
    for name, cond, up_msg, down_msg in checks:
        if cond:
            analysis.append([name, "🟢 看多", up_msg]); bull += 1
        else:
            analysis.append([name, "⚪ 中立", down_msg])
    
    # K線
    body = abs(latest['Close'] - latest['Open'])
    lower_s = min(latest['Close'], latest['Open']) - latest['Low']
    if lower_s > body * 2 and body > 0:
        analysis.append(["K線", "🟢 看多", "下影線支撐"]); bull += 1
    else:
        analysis.append(["K線", "⚪ 中立", "無特殊型態"])

    # 缺口
    if latest['Low'] > prev['High']:
        analysis.append(["缺口", "🟢 看多", "跳空強勢"]); bull += 1
    else:
        analysis.append(["缺口", "⚪ 中立", "無缺口"])

    # 籌碼 (簡化)
    if has_chip:
        inst_net = latest['Foreign_Buy'] + latest['Trust_Buy']
        if inst_net > 500:
            analysis.append(["法人", "🟢 看多", f"買超{int(inst_net)}張"]); bull += 1
        elif inst_net < -500:
            analysis.append(["法人", "🔴 看空", f"賣超{int(inst_net)}張"]); bear += 1
        else:
            analysis.append(["法人", "⚪ 中立", "觀望"])
        
        if latest['Margin_Diff'] < -500:
            analysis.append(["散戶", "🟢 看多", "融資退場，籌碼穩"]); bull += 1
        else:
            analysis.append(["散戶", "⚪ 中立", "穩定"])
    else:
        analysis.append(["法人", "⚪ 未知", "無資料"])
        analysis.append(["散戶", "⚪ 未知", "無資料"])

    report = f"#### 🔍 指標解析\n目前看多 {bull} 項，建議操作以{'偏多' if bull > 4 else '盤整'}為主。"
    strategy = f"#### 🎯 策略\n若回測 5MA ({latest['SMA_5']:.1f}) 可考慮少量介入。"
    
    return pd.DataFrame(analysis, columns=["分析維度", "訊號", "專家描述"]), bull, bear, report, strategy

# --- 主程式 UI ---
s_code = st.text_input("📈 請輸入台股代號", "2330").strip()

if s_code:
    with st.spinner("正在讀取資料..."):
        df_raw, actual_sym, s_name, has_c = fetch_complete_data(s_code)
        if df_raw is not None:
            db2 = generate_ta_db2(df_raw)
            db3_df, bulls, bears, rpt, stg = generate_expert_db3(db2, has_c)
            
            # 格式化股價顯示：若是整數則去小數點
            def fmt(val):
                return f"{val:.2f}".rstrip('0').rstrip('.') if val % 1 != 0 else f"{int(val)}"

            curr_p = db2.iloc[-1]['Close']
            prev_p = db2.iloc[-2]['Close']
            diff = curr_p - prev_p
            
            st.subheader(f"📊 分析標的：{s_code} - {s_name}")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("當前股價", f"{fmt(curr_p)} TWD", f"{diff:.2f}")
            m2.metric("看多指標", f"{bulls} / 9")
            m3.metric("看空指標", f"{bears} / 9")
            
            # K線圖
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_width=[0.3, 0.7])
            fig.add_trace(go.Candlestick(x=db2['Date'], open=db2['Open'], high=db2['High'], low=db2['Low'], close=db2['Close'], name="K線"), row=1, col=1)
            fig.add_trace(go.Scatter(x=db2['Date'], y=db2['SMA_5'], line=dict(color='blue', width=1), name="5MA"), row=1, col=1)
            fig.add_trace(go.Scatter(x=db2['Date'], y=db2['SMA_20'], line=dict(color='orange', width=1.5), name="20MA"), row=1, col=1)
            colors = ['#EF5350' if c >= o else '#26A69A' for c, o in zip(db2['Close'], db2['Open'])]
            fig.add_trace(go.Bar(x=db2['Date'], y=db2['Volume'], marker_color=colors, name="成交量"), row=2, col=1)
            fig.update_layout(xaxis_rangeslider_visible=False, height=500, margin=dict(t=30, l=10, r=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
            
            # 診斷區
            st.markdown("### 🧠 專家系統研判 (DB3)")
            cl, cr = st.columns([4, 6])
            with cl:
                st.dataframe(db3_df, hide_index=True, use_container_width=True)
            with cr:
                st.markdown(f'<div class="report-card">{rpt}</div>', unsafe_allow_html=True)
                st.write("")
                st.markdown(f'<div class="strategy-card">{stg}</div>', unsafe_allow_html=True)
