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

# --- 資料抓取 (DB1) ---
@st.cache_data(ttl=3600)
def fetch_complete_data(code):
    """抓取股價、名稱與台灣特有籌碼數據"""
    df_price, actual_sym, stock_name = None, None, "未知股票"
    
    # 嘗試上市與上櫃後綴
    for suffix in [".TW", ".TWO"]:
        try:
            temp_sym = f"{code}{suffix}"
            ticker_obj = yf.Ticker(temp_sym)
            # 抓取歷史資料
            df = ticker_obj.history(period="1y")
            
            if not df.empty:
                # 抓取股票名稱 (yfinance 有時會抓到英文，若無則顯示代號)
                stock_name = ticker_obj.info.get('shortName') or ticker_obj.info.get('longName') or temp_sym
                
                df.reset_index(inplace=True)
                # 處理新版 yfinance 可能產生的 MultiIndex
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
                df_price, actual_sym = df, temp_sym
                break
        except:
            continue
            
    if df_price is None:
        return None, None, None, False

    # 抓取籌碼資料
    try:
        dl = DataLoader()
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        df_inst = dl.taiwan_stock_institutional_investors(stock_id=code, start_date=start_date)
        df_margin = dl.taiwan_stock_margin_purchase_short_sale(stock_id=code, start_date=start_date)
        
        df_pivot = pd.DataFrame()
        if not df_inst.empty:
            df_pivot = df_inst.pivot_table(index='date', columns='name', values='buy_sell', aggfunc='sum').reset_index()
            df_pivot.rename(columns={'date': 'Date'}, inplace=True)
            f_col = '外陸資買賣超股數(不含外資自營商)'
            t_col = '投信買賣超股數'
            df_pivot['Foreign_Buy'] = df_pivot[f_col] / 1000 if f_col in df_pivot.columns else 0
            df_pivot['Trust_Buy'] = df_pivot[t_col] / 1000 if t_col in df_pivot.columns else 0
            df_pivot = df_pivot[['Date', 'Foreign_Buy', 'Trust_Buy']]

        if not df_margin.empty:
            df_margin = df_margin[['date', 'MarginPurchaseTodayBalance', 'ShortSaleTodayBalance']]
            df_margin.rename(columns={'date': 'Date', 'MarginPurchaseTodayBalance': 'Margin_Bal', 'ShortSaleTodayBalance': 'Short_Bal'}, inplace=True)
            
        if not df_pivot.empty:
            df_price = pd.merge(df_price, df_pivot, on='Date', how='left')
        if not df_margin.empty:
            df_price = pd.merge(df_price, df_margin, on='Date', how='left')
            
        df_price.fillna(method='ffill', inplace=True)
        df_price.fillna(0, inplace=True)
        return df_price, actual_sym, stock_name, True
    except:
        return df_price, actual_sym, stock_name, False

# --- 技術指標計算 (DB2) ---
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
        db2['Short_Diff'] = db2['Short_Bal'].diff()
    return db2.dropna()

# --- 專家分析引擎 (DB3) ---
def generate_expert_db3(db2, has_chip):
    latest = db2.iloc[-1]
    prev = db2.iloc[-2]
    analysis = []
    bull_score, bear_score, neutral_score = 0, 0, 0
    
    # 指標判斷邏輯 (略，與前版相同)
    # 均線
    if latest['SMA_5'] > latest['SMA_20']:
        analysis.append(["1. 均線理論", "🟢 看多", "多頭排列"])
        bull_score += 1
    else:
        analysis.append(["1. 均線理論", "🔴 看空", "趨勢偏弱"])
        bear_score += 1
    # RSI
    if latest['RSI'] > 75:
        analysis.append(["2. 動能 (RSI)", "🔴 看空", "超買"])
        bear_score += 1
    elif latest['RSI'] < 25:
        analysis.append(["2. 動能 (RSI)", "🟢 看多", "超賣"])
        bull_score += 1
    else:
        analysis.append(["2. 動能 (RSI)", "⚪ 中立", "震盪區"])
        neutral_score += 1
    # MACD
    if latest['MACD_H'] > 0:
        analysis.append(["3. 波段 (MACD)", "🟢 看多", "紅柱向上"])
        bull_score += 1
    else:
        analysis.append(["3. 波段 (MACD)", "🔴 看空", "綠柱向下"])
        bear_score += 1
    # 布林
    if latest['Close'] > latest['BB_Up']:
        analysis.append(["4. 布林通道", "🟢 看多", "強勢上攻"])
        bull_score += 1
    elif latest['Close'] < latest['BB_Low']:
        analysis.append(["4. 布林通道", "🔴 看空", "跌穿底線"])
        bear_score += 1
    else:
        analysis.append(["4. 布林通道", "⚪ 中立", "通道內盤整"])
        neutral_score += 1
    # KD
    if latest['K'] > 80:
        analysis.append(["5. KD 指標", "🔴 看空", "高檔區"])
        bear_score += 1
    elif latest['K'] < 20:
        analysis.append(["5. KD 指標", "🟢 看多", "低檔區"])
        bull_score += 1
    else:
        analysis.append(["5. KD 指標", "⚪ 中立", "無訊號"])
        neutral_score += 1
    # K線
    body = abs(latest['Close'] - latest['Open'])
    lower_s = min(latest['Close'], latest['Open']) - latest['Low']
    if lower_s > body * 2 and body > 0:
        analysis.append(["6. K線型態", "🟢 看多", "下影線支撐"])
        bull_score += 1
    else:
        analysis.append(["6. K線型態", "⚪ 中立", "常規型態"])
        neutral_score += 1
    # 缺口
    if latest['Low'] > prev['High']:
        analysis.append(["7. 缺口理論", "🟢 看多", "跳空上漲"])
        bull_score += 1
    elif latest['High'] < prev['Low']:
        analysis.append(["7. 缺口理論", "🔴 看空", "跳空下跌"])
        bear_score += 1
    else:
        analysis.append(["7. 缺口理論", "⚪ 中立", "無缺口"])
        neutral_score += 1
    # 籌碼
    if has_chip:
        inst_net = latest['Foreign_Buy'] + latest['Trust_Buy']
        if inst_net > 500:
            analysis.append(["8. 籌碼 (法人)", "🟢 看多", "大戶進場"])
            bull_score += 1
        elif inst_net < -500:
            analysis.append(["8. 籌碼 (法人)", "🔴 看空", "大戶撤退"])
            bear_score += 1
        else:
            analysis.append(["8. 籌碼 (法人)", "⚪ 中立", "觀望"])
            neutral_score += 1
        if latest['Margin_Diff'] > 500 and latest['Close'] < prev['Close']:
            analysis.append(["9. 籌碼 (散戶)", "🔴 看空", "散戶被套"])
            bear_score += 1
        elif latest['Margin_Diff'] < -500 and latest['Short_Diff'] > 200:
            analysis.append(["9. 籌碼 (散戶)", "🟢 看多", "軋空預備"])
            bull_score += 1
        else:
            analysis.append(["9. 籌碼 (散戶)", "⚪ 中立", "穩定"])
            neutral_score += 1
    else:
        analysis.append(["8. 籌碼 (法人)", "⚪ 未知", "缺少資料"])
        analysis.append(["9. 籌碼 (散戶)", "⚪ 未知", "缺少資料"])

    # 專家報告引擎 (保證不留白)
    report = "#### 🔍 1. 核心矛盾與信度解析\n"
    summary_text = f"目前市場呈現 **{'偏空' if bear_score > bull_score else '偏多' if bull_score > bear_score else '中立震盪'}** 態勢。"
    summary_text += f"在 9 大指標中，看多 {bull_score} 項，看空 {bear_score} 項。"
    report += f"> {summary_text}\n\n"

    conflict_found = False
    if latest['SMA_5'] > latest['SMA_20'] and latest['RSI'] > 75:
        report += "- **指標衝突(強勢鈍化)**：均線雖然多頭排列，但動能指標已達超買。專家建議：以趨勢為優先，不可在此時輕易放空。\n"
        conflict_found = True
    if not has_chip:
        report += "- **資料限制提醒**：目前缺少籌碼面數據，信度以技術面型態為主。\n"
        conflict_found = True
    
    if not conflict_found:
        report += "- **趨勢評論**：各項指標方向大致相符，目前盤勢較為明確，無顯著背離。\n"

    # 具體策略
    strategy = "#### 🎯 2. 具體操作策略建議\n"
    entry_p = latest['SMA_5'].round(2)
    stop_p = (latest['Close'] * 0.95).round(2)
    if bull_score > bear_score + 2:
        strategy += f"- **方向**：【偏多介入】\n- **建議點位**：拉回至 **{entry_p} 元** 附近佈局。\n- **風險**：跌破 **{stop_p} 元** 停損。"
    elif bear_score > bull_score + 2:
        strategy += f"- **方向**：【偏空觀望】\n- **建議點位**：待股價站穩 **{latest['SMA_20'].round(2)} 元** 再說。\n- **風險**：目前下跌動能強勁。"
    else:
        strategy += f"- **方向**：【區間震盪】\n- **建議位階**：在 **{latest['BB_Low'].round(2)}** 與 **{latest['BB_Up'].round(2)}** 間低買高賣。"

    return pd.DataFrame(analysis, columns=["分析維度", "訊號", "專家描述"]), bull_score, bear_score, report, strategy

# --- 主程式 UI ---
s_code = st.text_input("📈 請輸入台股代號 (如 2330, 5490)", "2330").strip()

if s_code:
    with st.spinner("正在讀取資料..."):
        df_raw, actual_sym, s_name, has_c = fetch_complete_data(s_code)
        if df_raw is not None:
            db2 = generate_ta_db2(df_raw)
            db3_df, bulls, bears, rpt, stg = generate_expert_db3(db2, has_c)
            
            # 修正小數點問題，保留兩位
            curr_p = round(float(db2.iloc[-1]['Close']), 2)
            prev_p = round(float(db2.iloc[-2]['Close']), 2)
            diff = round(curr_p - prev_p, 2)
            
            st.subheader(f"📊 分析標的：{s_code} - {s_name}")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("當前股價", f"{curr_p} TWD", f"{diff}")
            m2.metric("看多指標", f"{bulls} / 9")
            m3.metric("看空指標", f"{bears} / 9")
            
            # 圖表
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_width=[0.3, 0.7])
            fig.add_trace(go.Candlestick(x=db2['Date'], open=db2['Open'], high=db2['High'], low=db2['Low'], close=db2['Close'], name="K線"), row=1, col=1)
            fig.add_trace(go.Scatter(x=db2['Date'], y=db2['SMA_5'], line=dict(color='blue', width=1), name="5MA"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_raw['Date'], y=db2['SMA_20'], line=dict(color='orange', width=1.5), name="20MA"), row=1, col=1)
            colors = ['red' if c >= o else 'green' for c, o in zip(db2['Close'], db2['Open'])]
            fig.add_trace(go.Bar(x=db2['Date'], y=db2['Volume'], marker_color=colors, name="成交量"), row=2, col=1)
            fig.update_layout(xaxis_rangeslider_visible=False, height=500, margin=dict(t=30, l=10, r=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 🧠 專家系統研判 (DB3)")
            cl, cr = st.columns([4, 6])
            with cl:
                st.dataframe(db3_df, hide_index=True, use_container_width=True)
            with cr:
                st.markdown(f'<div class="report-card">{rpt}</div>', unsafe_allow_html=True)
                st.write("")
                st.markdown(f'<div class="strategy-card">{stg}</div>', unsafe_allow_html=True)
        else:
            st.error("查無資料，請確認代號是否正確。")
