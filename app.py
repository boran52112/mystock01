import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from FinMind.data import DataLoader
from datetime import datetime, timedelta

# --- 網頁配置 ---
st.set_page_config(page_title="AI 專家實事求是決策系統", layout="wide")

st.markdown("""
    <style>
    .report-card { background-color: #ffffff; padding: 25px; border-radius: 15px; border-left: 8px solid #1f77b4; box-shadow: 0 4px 12px rgba(0,0,0,0.08); line-height: 1.8; }
    .strategy-card { background-color: #f8f9fa; padding: 25px; border-radius: 15px; border-left: 8px solid #d62728; border: 1px solid #eee; line-height: 1.8; }
    .logic-tag { background-color: #fce4ec; color: #c2185b; padding: 2px 8px; border-radius: 4px; font-size: 0.85rem; font-weight: bold; border: 1px solid #f8bbd0; margin-right: 5px; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 AI 專家級 9 大指標實事求是分析系統 (V3.0)")

# --- 1. 資料抓取 ---
@st.cache_data(ttl=3600)
def fetch_stock_full_data(code):
    df_p, actual_sym, s_name = None, None, f"股票 {code}"
    try:
        dl = DataLoader()
        info = dl.taiwan_stock_info()
        item = info[info['stock_id'] == code]
        if not item.empty: s_name = item.iloc[0]['stock_name']
    except: pass

    for suffix in [".TW", ".TWO"]:
        try:
            temp_sym = f"{code}{suffix}"
            ticker = yf.Ticker(temp_sym)
            df = ticker.history(period="1y")
            if not df.empty:
                df.reset_index(inplace=True)
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
                df_p, actual_sym = df, temp_sym
                break
        except: continue
    
    if df_p is None: return None, None, None, False
    
    try:
        start = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        df_i = dl.taiwan_stock_institutional_investors(stock_id=code, start_date=start)
        df_m = dl.taiwan_stock_margin_purchase_short_sale(stock_id=code, start_date=start)
        df_chip = pd.DataFrame()
        if not df_i.empty:
            df_i = df_i.pivot_table(index='date', columns='name', values='buy_sell', aggfunc='sum').reset_index()
            df_i.rename(columns={'date': 'Date'}, inplace=True)
            df_i['Foreign_Buy'] = df_i.get('外陸資買賣超股數(不含外資自營商)', 0) / 1000
            df_i['Trust_Buy'] = df_i.get('投信買賣超股數', 0) / 1000
            df_chip = df_i[['Date', 'Foreign_Buy', 'Trust_Buy']]
        if not df_m.empty:
            df_m = df_m[['date', 'MarginPurchaseTodayBalance', 'ShortSaleTodayBalance']]
            df_m.rename(columns={'date': 'Date', 'MarginPurchaseTodayBalance': 'Margin_Bal', 'ShortSaleTodayBalance': 'Short_Bal'}, inplace=True)
            df_chip = pd.merge(df_chip, df_m, on='Date', how='outer') if not df_chip.empty else df_m
        if not df_chip.empty: df_p = pd.merge(df_p, df_chip, on='Date', how='left')
        df_p.fillna(method='ffill', inplace=True).fillna(0, inplace=True)
        return df_p, actual_sym, s_name, True
    except: return df_p, actual_sym, s_name, False

# --- 2. 指標計算 ---
def calculate_db2(df):
    d = df.copy()
    d['SMA_5'] = d['Close'].rolling(5).mean(); d['SMA_20'] = d['Close'].rolling(20).mean()
    delta = d['Close'].diff(); gain = delta.where(delta > 0, 0).rolling(14).mean(); loss = -delta.where(delta < 0, 0).rolling(14).mean()
    d['RSI'] = 100 - (100 / (1 + gain/loss))
    exp1 = d['Close'].ewm(span=12).mean(); exp2 = d['Close'].ewm(span=26).mean()
    d['MACD_H'] = (exp1-exp2) - (exp1-exp2).ewm(span=9).mean()
    d['BB_Mid'] = d['Close'].rolling(20).mean(); std = d['Close'].rolling(20).std()
    d['BB_Up'] = d['BB_Mid'] + 2*std; d['BB_Low'] = d['BB_Mid'] - 2*std
    l9, h9 = d['Low'].rolling(9).min(), d['High'].rolling(9).max()
    rsv = (d['Close'] - l9) / (h9 - l9) * 100
    d['K'] = rsv.ewm(com=2).mean(); d['D'] = d['K'].ewm(com=2).mean()
    d['Margin_Diff'] = d['Margin_Bal'].diff() if 'Margin_Bal' in d.columns else 0
    return d.dropna()

# --- 3. 實事求是決策系統 (核心修正) ---
def generate_expert_report(db2, has_chip):
    latest, prev = db2.iloc[-1], db2.iloc[-2]
    bull_score, bear_score = 0, 0
    analysis_table = []
    
    # 1. 均線 (W3)
    is_sma_bull = latest['SMA_5'] > latest['SMA_20']
    if is_sma_bull: analysis_table.append(["1. 均線趨勢", "🟢 看多", "多頭排列 (W3)"]); bull_score += 3
    else: analysis_table.append(["1. 均線趨勢", "🔴 看空", "趨勢向下 (W3)"]); bear_score += 3

    # 2. RSI (W1)
    if latest['RSI'] > 75: analysis_table.append(["2. 動能 RSI", "🔴 看空", "超買區"]); bear_score += 1
    elif latest['RSI'] < 25: analysis_table.append(["2. 動能 RSI", "🟢 看多", "超賣區"]); bull_score += 1
    else: analysis_table.append(["2. 動能 RSI", "⚪ 中立", "震盪區間"])

    # 3. MACD (W1)
    is_macd_bull = latest['MACD_H'] > 0
    if is_macd_bull: analysis_table.append(["3. 波段 MACD", "🟢 看多", "動能增強"]); bull_score += 1
    else: analysis_table.append(["3. 波段 MACD", "🔴 看空", "波段向下"]); bear_score += 1

    # 4. 布林 (W1)
    if latest['Close'] > latest['BB_Up']: analysis_table.append(["4. 布林通道", "🟢 看多", "強勢表態"]); bull_score += 1
    elif latest['Close'] < latest['BB_Low']: analysis_table.append(["4. 布林通道", "🔴 看空", "下殺破位"]); bear_score += 1
    else: analysis_table.append(["4. 布林通道", "⚪ 中立", "軌道內盤整"])

    # 5. KD (W1)
    if latest['K'] > 80: analysis_table.append(["5. KD 指標", "🔴 看空", "高檔區"]); bear_score += 1
    elif latest['K'] < 20: analysis_table.append(["5. KD 指標", "🟢 看多", "低檔區"]); bull_score += 1
    else: analysis_table.append(["5. KD 指標", "⚪ 中立", "無訊號"])

    # 6. K線 (W2)
    body = abs(latest['Close'] - latest['Open']); lower_s = min(latest['Close'], latest['Open']) - latest['Low']
    is_shadow_bull = lower_s > body * 1.5 and body > 0
    if is_shadow_bull: analysis_table.append(["6. K線型態", "🟢 看多", "影線支撐 (W2)"]); bull_score += 2
    else: analysis_table.append(["6. K線型態", "⚪ 中立", "常態K線"])

    # 7. 缺口 (W2)
    is_gap_bull = latest['Low'] > prev['High']
    if is_gap_bull: analysis_table.append(["7. 缺口理論", "🟢 看多", "向上跳空 (W2)"]); bull_score += 2
    else: analysis_table.append(["7. 缺口理論", "⚪ 中立", "無跳空"])

    # 8. 法人 (W3)
    if has_chip:
        inst_net = latest['Foreign_Buy'] + latest['Trust_Buy']
        if inst_net > 500: analysis_table.append(["8. 法人籌碼", "🟢 看多", "法人大買 (W3)"]); bull_score += 3
        elif inst_net < -500: analysis_table.append(["8. 法人籌碼", "🔴 看空", "法人倒貨 (W3)"]); bear_score += 3
        else: analysis_table.append(["8. 法人籌碼", "⚪ 中立", "無大動作"])
    else: analysis_table.append(["8. 法人籌碼", "⚪ 未知", "缺少資料"])

    # 9. 散戶 (W2)
    if has_chip:
        if latest['Margin_Diff'] > 500 and latest['Close'] < prev['Close']:
            analysis_table.append(["9. 散戶籌碼", "🔴 看空", "資增價跌 (W2)"]); bear_score += 2
        elif latest['Margin_Diff'] < -500 and latest['Close'] > prev['Close']:
            analysis_table.append(["9. 散戶籌碼", "🟢 看多", "資減籌碼穩 (W2)"]); bull_score += 2
        else: analysis_table.append(["9. 散戶籌碼", "⚪ 中立", "散戶穩定"])
    else: analysis_table.append(["9. 散戶籌碼", "⚪ 未知", "缺少資料"])

    # --- DB3 核心矛盾解析 (不再出錯，實事求是) ---
    score_diff = bull_score - bear_score
    mood = "多方佔優勢" if score_diff > 0 else "空方佔優勢" if score_diff < 0 else "多空勢均力敵"
    
    rpt_html = f"<div class='report-title'>🔍 1. 核心矛盾與信度解析</div><ul>"
    rpt_html += f"<li><b>盤勢診斷：</b>目前的加權總分顯示市場由 <b style='color:{'green' if score_diff > 0 else 'red'}'>{mood}</b>。</li>"
    
    # 深度解析邏輯
    contradictions = []
    if is_sma_bull and not is_macd_bull:
        contradictions.append("<li><span class='logic-tag'>動能背離</span> 均線(趨勢)看多但 MACD(動能)看空。專家認為這是『強勢中的修正』，非反轉訊號，持股應防守而非出清。</li>")
    elif not is_sma_bull and is_shadow_bull:
        contradictions.append("<li><span class='logic-tag'>底部反彈</span> 趨勢看空但K線出現長下影支撐。專家提醒此處不宜加碼放空，因為下方買盤已開始承接。</li>")
    elif not is_sma_bull and not is_macd_bull:
        contradictions.append("<li><span class='logic-tag'>空頭一致性</span> 均線趨勢與波段動能同步向下。專家解析：這是典型的空頭結構，下跌具備高度信度，建議全面避險。</li>")
    
    if not contradictions:
        rpt_html += f"<li>指標方向高度同步，信度極高。</li>"
    else:
        rpt_html += "".join(contradictions)
    
    if not has_chip:
        rpt_html += "<li>⚠️ <b>注意：</b>目前缺乏法人籌碼數據，信度完全依賴技術型態。</li>"
    rpt_html += "</ul>"

    # --- DB3 具體操作策略 ---
    stg_html = "<div class='report-title'>🎯 2. 具體操作策略與動作</div><ul>"
    if score_diff >= 4:
        stg_html += f"<li><b style='color:green; font-size:1.1rem;'>✅ 建議動作：買入 / 持股續抱</b></li><li><b>原因：</b>多項核心指標(W3)與趨勢同步看多。</li>"
    elif score_diff <= -4:
        stg_html += f"<li><b style='color:red; font-size:1.1rem;'>🚨 建議動作：積極放空 / 全面清倉</b></li><li><b>原因：</b>趨勢與動能同步破壞，且空方分數具有顯著優勢。</li>"
    elif score_diff < 0:
        stg_html += f"<li><b>建議動作：消極避險 (賣出觀望)</b></li><li><b>原因：</b>空方微幅領先。雖不一定要放空，但應先收回資金避開下行風險。</li>"
    else:
        stg_html += f"<li><b>建議動作：觀望 (Hold)</b></li><li><b>原因：</b>多空拉鋸中。</li>"
    stg_html += "</ul>"

    return pd.DataFrame(analysis_table, columns=["維度", "訊號", "專家解析"]), bull_score, bear_score, rpt_html, stg_html

# --- 繪圖功能 ---
def plot_chart(df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_width=[0.3, 0.7])
    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="K線"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_5'], line=dict(color='blue', width=1), name="5MA"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], line=dict(color='orange', width=1.5), name="20MA"), row=1, col=1)
    colors = ['#EF5350' if c >= o else '#26A69A' for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], marker_color=colors, name="成交量"), row=2, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False, height=500, margin=dict(t=30, l=10, r=10, b=10))
    return fig

# --- 主程式執行 ---
code = st.text_input("📈 請輸入台股代號", "2330").strip()
if code:
    with st.spinner("AI 專家執行全維度診斷中..."):
        df_raw, sym, s_name, has_c = fetch_stock_full_data(code)
        if df_raw is not None:
            db2 = calculate_db2(df_raw)
            db3_df, b_score, r_score, rpt_h, stg_h = generate_expert_report(db2, has_c)
            
            st.subheader(f"📊 分析標的：{code} - {s_name}")
            col1, col2, col3 = st.columns(3)
            col1.metric("當前股價", f"{db2.iloc[-1]['Close']:.2f} TWD")
            col2.metric("多方加權總分", f"{b_score} 分")
            col3.metric("空方加權總分", f"{r_score} 分")
            
            st.plotly_chart(plot_chart(db2.tail(100)), use_container_width=True)
            
            st.markdown("### 🧠 專家系統診斷與全維度對策 (DB3)")
            cl, cr = st.columns([4, 6])
            with cl:
                st.dataframe(db3_df, hide_index=True, use_container_width=True)
            with cr:
                st.markdown(f'<div class="report-card">{rpt_h}</div>', unsafe_allow_html=True)
                st.write("")
                st.markdown(f'<div class="strategy-card">{stg_h}</div>', unsafe_allow_html=True)
        else: st.error("查無資料。")
