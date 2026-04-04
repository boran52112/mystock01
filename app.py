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

# 自定義 CSS 提升質感
st.markdown("""
    <style>
    .report-card { background-color: #ffffff; padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .strategy-card { background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #ff4b4b; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 AI 專家級 9 大指標技術分析系統")

# --- 資料抓取 (DB1) ---
@st.cache_data(ttl=3600)
def fetch_complete_data(code):
    """抓取股價與台灣特有籌碼數據"""
    df_price, actual_sym = None, None
    # 支援上市與上櫃
    for suffix in [".TW", ".TWO"]:
        try:
            temp_sym = f"{code}{suffix}"
            df = yf.download(temp_sym, period="1y", progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                df.reset_index(inplace=True)
                df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
                df_price, actual_sym = df, temp_sym
                break
        except: continue
            
    if df_price is None: return None, None, False

    # 抓取籌碼
    try:
        dl = DataLoader()
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        # 法人
        df_inst = dl.taiwan_stock_institutional_investors(stock_id=code, start_date=start_date)
        # 融資券
        df_margin = dl.taiwan_stock_margin_purchase_short_sale(stock_id=code, start_date=start_date)
        
        # 處理法人資料
        df_pivot = pd.DataFrame()
        if not df_inst.empty:
            df_pivot = df_inst.pivot_table(index='date', columns='name', values='buy_sell', aggfunc='sum').reset_index()
            df_pivot.rename(columns={'date': 'Date'}, inplace=True)
            f_col = '外陸資買賣超股數(不含外資自營商)'
            t_col = '投信買賣超股數'
            df_pivot['Foreign_Buy'] = df_pivot[f_col] / 1000 if f_col in df_pivot.columns else 0
            df_pivot['Trust_Buy'] = df_pivot[t_col] / 1000 if t_col in df_pivot.columns else 0
            df_pivot = df_pivot[['Date', 'Foreign_Buy', 'Trust_Buy']]

        # 處理融資券
        if not df_margin.empty:
            df_margin = df_margin[['date', 'MarginPurchaseTodayBalance', 'ShortSaleTodayBalance']]
            df_margin.rename(columns={'date': 'Date', 'MarginPurchaseTodayBalance': 'Margin_Bal', 'ShortSaleTodayBalance': 'Short_Bal'}, inplace=True)
            
        # 合併
        if not df_pivot.empty: df_price = pd.merge(df_price, df_pivot, on='Date', how='left')
        if not df_margin.empty: df_price = pd.merge(df_price, df_margin, on='Date', how='left')
        
        df_price.fillna(method='ffill', inplace=True)
        df_price.fillna(0, inplace=True)
        return df_price, actual_sym, True
    except:
        return df_price, actual_sym, False

# --- 技術指標計算 (DB2) ---
def generate_ta_db2(df):
    db2 = df.copy()
    # 1. 均線
    db2['SMA_5'] = db2['Close'].rolling(window=5).mean()
    db2['SMA_20'] = db2['Close'].rolling(window=20).mean()
    # 2. RSI
    delta = db2['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    db2['RSI'] = 100 - (100 / (1 + gain/loss))
    # 3. MACD
    exp1 = db2['Close'].ewm(span=12).mean()
    exp2 = db2['Close'].ewm(span=26).mean()
    db2['MACD'] = exp1 - exp2
    db2['MACD_S'] = db2['MACD'].ewm(span=9).mean()
    db2['MACD_H'] = db2['MACD'] - db2['MACD_S']
    # 4. 布林
    db2['BB_Mid'] = db2['Close'].rolling(window=20).mean()
    std = db2['Close'].rolling(window=20).std()
    db2['BB_Up'] = db2['BB_Mid'] + 2*std
    db2['BB_Low'] = db2['BB_Mid'] - 2*std
    # 5. KD
    l9, h9 = db2['Low'].rolling(9).min(), db2['High'].rolling(9).max()
    rsv = (db2['Close'] - l9) / (h9 - l9) * 100
    db2['K'] = rsv.ewm(com=2).mean()
    db2['D'] = db2['K'].ewm(com=2).mean()
    # 6-9 其他輔助與籌碼差值
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
    bull_score, bear_score = 0, 0
    
    # 1. 均線理論
    if latest['SMA_5'] > latest['SMA_20']:
        analysis.append(["1. 均線理論", "🟢 看多", "短均線在長均線之上，多頭排列"])
        bull_score += 1
    else:
        analysis.append(["1. 均線理論", "🔴 看空", "均線死叉，趨勢偏弱"])
        bear_score += 1

    # 2. RSI 動能
    if latest['RSI'] > 75:
        analysis.append(["2. 動能 (RSI)", "🔴 看空", f"RSI:{latest['RSI'].round(1)} 進入超買區"])
        bear_score += 1
    elif latest['RSI'] < 25:
        analysis.append(["2. 動能 (RSI)", "🟢 看多", f"RSI:{latest['RSI'].round(1)} 進入超賣區"])
        bull_score += 1
    else:
        analysis.append(["2. 動能 (RSI)", "⚪ 中立", "動能處於正常震盪區"])

    # 3. MACD 波段
    if latest['MACD_H'] > 0:
        analysis.append(["3. 波段 (MACD)", "🟢 看多", "MACD 紅柱增長"])
        bull_score += 1
    else:
        analysis.append(["3. 波段 (MACD)", "🔴 看空", "MACD 綠柱區間"])
        bear_score += 1

    # 4. 布林通道
    if latest['Close'] > latest['BB_Up']:
        analysis.append(["4. 布林通道", "🟢 看多", "強勢突破上軌"])
        bull_score += 1
    elif latest['Close'] < latest['BB_Low']:
        analysis.append(["4. 布林通道", "🔴 看空", "跌破下軌防線"])
        bear_score += 1
    else:
        analysis.append(["4. 布林通道", "⚪ 中立", "通道內盤整"])

    # 5. KD 短線
    if latest['K'] > 80:
        analysis.append(["5. KD 指標", "🔴 看空", "KD 高檔鈍化/超買"])
        bear_score += 1
    elif latest['K'] < 20:
        analysis.append(["5. KD 指標", "🟢 看多", "KD 低檔超賣區"])
        bull_score += 1
    else:
        analysis.append(["5. KD 指標", "⚪ 中立", "KD 無明顯訊號"])

    # 6. K線型態
    body = abs(latest['Close'] - latest['Open'])
    lower_s = min(latest['Close'], latest['Open']) - latest['Low']
    if lower_s > body * 2 and body > 0:
        analysis.append(["6. K線型態", "🟢 看多", "出現長下影線支撐"])
        bull_score += 1
    else:
        analysis.append(["6. K線型態", "⚪ 中立", "無特殊K線型態"])

    # 7. 缺口理論
    if latest['Low'] > prev['High']:
        analysis.append(["7. 缺口理論", "🟢 看多", "向上跳空缺口，多方強勁"])
        bull_score += 1
    elif latest['High'] < prev['Low']:
        analysis.append(["7. 缺口理論", "🔴 看空", "向下跳空缺口，空方表態"])
        bear_score += 1
    else:
        analysis.append(["7. 缺口理論", "⚪ 中立", "無跳空缺口"])

    # 8. 法人籌碼
    if has_chip:
        inst_net = latest['Foreign_Buy'] + latest['Trust_Buy']
        if inst_net > 500:
            analysis.append(["8. 籌碼 (法人)", "🟢 看多", f"法人今日大買 {int(inst_net)} 張"])
            bull_score += 1
        elif inst_net < -500:
            analysis.append(["8. 籌碼 (法人)", "🔴 看空", f"法人今日大賣 {int(inst_net)} 張"])
            bear_score += 1
        else:
            analysis.append(["8. 籌碼 (法人)", "⚪ 中立", "法人動作不大"])
    else:
        analysis.append(["8. 籌碼 (法人)", "⚪ 未知", "缺少資料"])

    # 9. 散戶籌碼
    if has_chip:
        if latest['Margin_Diff'] > 500 and latest['Close'] < prev['Close']:
            analysis.append(["9. 籌碼 (散戶)", "🔴 看空", "融資增股價跌，籌碼凌亂"])
            bear_score += 1
        elif latest['Margin_Diff'] < -500 and latest['Short_Diff'] > 200:
            analysis.append(["9. 籌碼 (散戶)", "🟢 看多", "資減券增，醞釀軋空"])
            bull_score += 1
        else:
            analysis.append(["9. 籌碼 (散戶)", "⚪ 中立", "散戶籌碼穩定"])
    else:
        analysis.append(["9. 籌碼 (散戶)", "⚪ 未知", "缺少資料"])

    # ==========================
    # 🧠 專家深度解析引擎 (Enhanced)
    # ==========================
    report = "#### 🔍 1. 核心矛盾與信度解析\n"
    
    # 矛盾 A: 趨勢與鈍化
    if latest['SMA_5'] > latest['SMA_20'] and (latest['RSI'] > 75 or latest['K'] > 80):
        report += "- **【多頭高檔鈍化】**：目前雖然技術指標顯示「超買」，但均線呈現多頭排列。專家認為此時「趨勢優先」，超買指標反映的是強勢而非反轉。**信度：技術面 > 震盪指標。**\n"
    
    # 矛盾 B: 價量與籌碼
    if latest['Close'] > prev['Close'] and has_chip and (latest['Foreign_Buy'] + latest['Trust_Buy'] < -1000):
        report += "- **【警戒：背離訊號】**：股價今日雖漲，但法人卻大幅賣超。這可能是散戶追價、主力出貨的跡象，需防範假突破。**信度：籌碼面 > 技術面。**\n"
    
    # 矛盾 C: 底部支撐
    if latest['Close'] < latest['SMA_20'] and lower_s > body * 2:
        report += "- **【底部支撐確認】**：雖然跌破月線趨勢偏空，但在低檔出現長下影線。專家認為此價位有強力資金護盤，不宜在此殺低。**信度：K線型態優先。**\n"

    if bull_score == bear_score:
        report += "- **【多空拉鋸】**：目前指標 50/50，市場正在等待新的利多或利空表態，建議縮小部位觀望。\n"

    # ==========================
    # 🎯 2. 具體操作策略
    # ==========================
    strategy = "#### 🎯 2. 具體操作策略建議\n"
    target_entry = latest['SMA_5'].round(2)
    stop_loss = (latest['BB_Low'] if latest['Close'] > latest['BB_Mid'] else latest['Prev_Low']).round(2)
    
    if bull_score > bear_score + 2:
        strategy += f"> **💪 強勢偏多**：建議分批佈局。最佳介入點為 **{target_entry} 元** (5日線附近)。若帶量突破 **{latest['BB_Up'].round(2)} 元** 可加碼。停損設在 **{stop_loss} 元**。"
    elif bear_score > bull_score + 2:
        strategy += f"> **⚠️ 偏空觀望**：目前空方佔優。建議暫時空手。除非股價站穩 **{latest['SMA_20'].round(2)} 元** 且法人轉買，否則不宜接刀。下一波支撐看 **{latest['BB_Low'].round(2)} 元**。"
    else:
        strategy += f"> **⚖️ 區間震盪**：不建議追高殺低。可在 **{latest['BB_Low'].round(2)}** 與 **{latest['BB_Up'].round(2)}** 之間進行高拋低吸。等待 MACD 紅柱再次增長時再轉為積極。"

    return pd.DataFrame(analysis, columns=["分析維度", "訊號", "專家狀態描述"]), bull_score, bear_score, report, strategy

# --- 繪圖函數 ---
def plot_stock_chart(df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_width=[0.3, 0.7])
    # K線 + 均線 + 布林
    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="K線"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_5'], line=dict(color='blue', width=1), name="5MA"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], line=dict(color='orange', width=1.5), name="20MA"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Up'], line=dict(color='rgba(255,0,0,0.2)', dash='dot'), name="布林上軌"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Low'], line=dict(color='rgba(0,255,0,0.2)', dash='dot'), name="布林下軌"), row=1, col=1)
    # 成交量
    colors = ['red' if c >= o else 'green' for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], marker_color=colors, name="成交量"), row=2, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False, height=600, margin=dict(t=30, l=10, r=10, b=10))
    return fig

# --- 執行介面 ---
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    s_code = st.text_input("📈 請輸入台股代號 (如 2330, 8069)", "2330").strip()
with col_h2:
    st.write("")
    st.write("")
    go_btn = st.button("執行全維度分析", use_container_width=True)

if s_code:
    with st.spinner("正在啟動 AI 專家大腦計算中..."):
        raw_data, sym_name, has_c = fetch_complete_data(s_code)
        if raw_data is not None:
            db2 = generate_ta_db2(raw_data)
            db3_df, bulls, bears, rpt, stg = generate_expert_db3(db2, has_c)
            
            # 1. 核心指標摘要
            c_price = db2.iloc[-1]['Close']
            prev_price = db2.iloc[-2]['Close']
            m1, m2, m3 = st.columns(3)
            m1.metric("當前股價", f"{c_price} TWD", f"{round(c_price-prev_price, 2)}")
            m2.metric("看多指標", f"{bulls} / 9")
            m3.metric("看空指標", f"{bears} / 9")
            
            # 2. 圖表
            st.plotly_chart(plot_stock_chart(db2.tail(120)), use_container_width=True)
            
            # 3. 專家診斷區
            st.markdown("### 🧠 專家系統研判 (DB3)")
            col_res_l, col_res_r = st.columns([4, 6])
            with col_res_l:
                st.write("**9 大指標訊號一覽**")
                st.dataframe(db3_df, hide_index=True, use_container_width=True)
            with col_res_r:
                st.markdown('<div class="report-card">', unsafe_allow_html=True)
                st.markdown(rpt)
                st.markdown('</div>', unsafe_allow_html=True)
                st.write("")
                st.markdown('<div class="strategy-card">', unsafe_allow_html=True)
                st.markdown(stg)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # 4. 數據表
            with st.expander("查看技術指標完整數據庫 (DB2)"):
                st.dataframe(db2.tail(20))
        else:
            st.error("查無資料，請確認代號是否正確。")
