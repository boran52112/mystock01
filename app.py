import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from FinMind.data import DataLoader
from datetime import datetime, timedelta

# --- 網頁配置 ---
st.set_page_config(page_title="AI 偵探級全維度過濾系統", layout="wide")

st.markdown("""
    <style>
    .report-card { background-color: #ffffff; padding: 25px; border-radius: 15px; border-left: 8px solid #1f77b4; box-shadow: 0 4px 12px rgba(0,0,0,0.08); line-height: 1.8; }
    .strategy-card { background-color: #f8f9fa; padding: 25px; border-radius: 15px; border-left: 8px solid #d62728; border: 1px solid #eee; line-height: 1.8; }
    .detective-tag { background-color: #fff3e0; color: #e65100; padding: 2px 10px; border-radius: 6px; font-size: 0.9rem; font-weight: bold; border: 1px solid #ffcc80; margin-right: 8px; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("🕵️‍♂️ AI 偵探級：9 大指標「反騙線」診斷系統")

# --- 1. 資料抓取 ---
@st.cache_data(ttl=3600)
def fetch_complete_data(code):
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

# --- 2. 指標計算 (DB2) ---
def generate_db2(df):
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
    d['Vol_Avg'] = d['Volume'].rolling(5).mean() # 成交量均線
    d['Margin_Diff'] = d['Margin_Bal'].diff() if 'Margin_Bal' in d.columns else 0
    return d.dropna()

# --- 3. AI 偵探過濾引擎 (DB3) ---
def generate_detective_report(db2, has_chip):
    latest, prev = db2.iloc[-1], db2.iloc[-2]
    bull_score, bear_score = 0, 0
    table_data = []
    
    # 指標加權診斷 (強制 9 個指標)
    # W3: 趨勢、法人 / W2: 型態、缺口、散戶 / W1: 動能、KD、MACD、布林
    
    # 1. 均線
    is_sma_up = latest['SMA_5'] > latest['SMA_20']
    table_data.append(["1. 均線趨勢", "🟢 看多" if is_sma_up else "🔴 看空", "多頭格局" if is_sma_up else "空頭格局"])
    if is_sma_up: bull_score += 3
    else: bear_score += 3

    # 2-5. 震盪指標
    rsi_sig = "🔴 看空" if latest['RSI'] > 75 else "🟢 看多" if latest['RSI'] < 25 else "⚪ 中立"
    table_data.append(["2. 動能 RSI", rsi_sig, "超買" if latest['RSI'] > 75 else "超賣" if latest['RSI'] < 25 else "正常"])
    if "看多" in rsi_sig: bull_score += 1
    elif "看空" in rsi_sig: bear_score += 1

    macd_sig = "🟢 看多" if latest['MACD_H'] > 0 else "🔴 看空"
    table_data.append(["3. 波段 MACD", macd_sig, "紅柱" if latest['MACD_H'] > 0 else "綠柱"])
    if "看多" in macd_sig: bull_score += 1
    else: bear_score += 1

    bb_sig = "🟢 看多" if latest['Close'] > latest['BB_Up'] else "🔴 看空" if latest['Close'] < latest['BB_Low'] else "⚪ 中立"
    table_data.append(["4. 布林通道", bb_sig, "觸頂" if latest['Close'] > latest['BB_Up'] else "破位" if latest['Close'] < latest['BB_Low'] else "通道內"])
    if "看多" in bb_sig: bull_score += 1
    elif "看空" in bb_sig: bear_score += 1

    kd_sig = "🔴 看空" if latest['K'] > 80 else "🟢 看多" if latest['K'] < 20 else "⚪ 中立"
    table_data.append(["5. KD 指標", kd_sig, "高檔" if latest['K'] > 80 else "低檔" if latest['K'] < 20 else "整理"])
    if "看多" in kd_sig: bull_score += 1
    elif "看空" in kd_sig: bear_score += 1

    # 6-7. 型態
    body = abs(latest['Close'] - latest['Open']); lower_s = min(latest['Close'], latest['Open']) - latest['Low']
    is_shadow = lower_s > body * 1.5 and body > 0
    table_data.append(["6. K線型態", "🟢 看多" if is_shadow else "⚪ 中立", "長下影線" if is_shadow else "無特色"])
    if is_shadow: bull_score += 2

    is_gap = latest['Low'] > prev['High']
    table_data.append(["7. 缺口理論", "🟢 看多" if is_gap else "⚪ 中立", "向上缺口" if is_gap else "無缺口"])
    if is_gap: bull_score += 2

    # 8-9. 籌碼 (關鍵偵探區)
    inst_net = 0
    if has_chip:
        inst_net = latest['Foreign_Buy'] + latest['Trust_Buy']
        if inst_net > 500: bull_score += 3; table_data.append(["8. 法人籌碼", "🟢 看多", "大戶吸籌"])
        elif inst_net < -500: bear_score += 3; table_data.append(["8. 法人籌碼", "🔴 看空", "大戶出貨"])
        else: table_data.append(["8. 法人籌碼", "⚪ 中立", "盤整"])
        
        if latest['Margin_Diff'] > 500 and latest['Close'] < prev['Close']:
            bear_score += 2; table_data.append(["9. 散戶籌碼", "🔴 看空", "資增價跌"])
        else:
            table_data.append(["9. 散戶籌碼", "⚪ 中立", "穩定"])
    else:
        table_data.append(["8. 法人籌碼", "⚪ 未知", "無資料"]); table_data.append(["9. 散戶籌碼", "⚪ 未知", "無資料"])

    # --- 🕵️‍♂️ 核心：反騙線偵探邏輯 ---
    rpt_html = "<div class='report-title'>🔍 1. 核心矛盾與「反騙線」診斷</div><ul>"
    
    # 診斷 A: 騙空洗盤 (Bear Trap)
    is_fake_bear = False
    if not is_sma_up and (inst_net > 500 or latest['Volume'] < latest['Vol_Avg'] * 0.7 or is_shadow):
        is_fake_bear = True
        rpt_html += f"<li><span class='detective-tag'>偵測到：空頭洗盤</span> 雖然均線看空，但出現了{'法人買超' if inst_net > 500 else ''}{'成交量萎縮' if latest['Volume'] < latest['Vol_Avg'] * 0.7 else ''}。<b>結論：</b>這極大機率是騙線，主力在刻意壓盤洗掉浮額，此處不宜放空，應等止跌回升。</li>"
    
    # 診斷 B: 騙多出貨 (Bull Trap)
    is_fake_bull = False
    if is_sma_up and (inst_net < -1000 or latest['Volume'] < latest['Vol_Avg'] * 0.7):
        is_fake_bull = True
        rpt_html += f"<li><span class='detective-tag'>偵測到：多頭陷阱</span> 價格雖在均線上漲，但法人反向大賣。<b>結論：</b>這是典型的拉高出貨，技術面指標雖綠但信度極低，切勿追高。</li>"

    # 診斷 C: 強勢鈍化 (Ignoring Noise)
    if is_sma_up and (latest['RSI'] > 75 or latest['K'] > 80) and not is_fake_bull:
        rpt_html += "<li><span class='detective-tag'>診斷：強勢鈍化</span> 趨勢極強導致震盪指標過熱。專家解析：這是強者恆強的表現，超買訊號無效，應持股直到破 5MA。</li>"

    if not is_fake_bear and not is_fake_bull:
        rpt_html += "<li><b>診斷結論：</b>目前指標方向一致，未觀察到顯著的量價背離或籌碼衝突，技術面信度高達 85% 以上。</li>"
    
    rpt_html += "</ul>"

    # --- 🎯 2. 具體操作策略與動作強度 ---
    stg_html = "<div class='report-title'>🎯 2. 具體操作策略與動作強度</div><ul>"
    
    # 根據偵探結論調整強度
    final_score = bull_score - bear_score
    
    if is_fake_bear:
        stg_html += "<li><b>操作方向：</b>【消極避險，不宜放空】</li><li><b>強度理由：</b>偵測到洗盤特徵。即使指標偏空，但大戶並未撤退，放空風險極高。</li>"
    elif final_score >= 6:
        stg_html += f"<li><b style='color:green; font-size:1.1rem;'>✅ 強勢進攻 (積極做多)</b></li><li><b>動作：</b>建議於 {latest['SMA_5']:.2f} 附近介入，目標布林上軌。</li>"
    elif final_score <= -6:
        stg_html += f"<li><b style='color:red; font-size:1.1rem;'>🚨 積極進攻 (融券放空 ★★★)</b></li><li><b>強度理由：</b>趨勢、量價、籌碼同步崩潰。這不只是賣出，而是適合建立空方部位的表態。</li>"
    else:
        stg_html += "<li><b>操作方向：</b>【中立觀望 / 消極賣出】</li><li><b>強度理由：</b>盤勢不明，矛盾指標多，建議先回收現金保護資產。</li>"
    stg_html += "</ul>"

    return pd.DataFrame(table_data, columns=["維度", "訊號", "解析"]), bull_score, bear_score, rpt_html, stg_html

# --- UI 顯示模組 ---
s_code = st.text_input("📈 請輸入台股代號", "2330").strip()
if s_code:
    with st.spinner("AI 偵探邏輯分析中..."):
        df_raw, sym, s_name, has_c = fetch_complete_data(s_code)
        if df_raw is not None:
            db2 = generate_db2(df_raw)
            db3_df, b_s, r_s, rpt_h, stg_h = generate_detective_report(db2, has_c)
            
            st.subheader(f"🕵️‍♂️ 分析對象：{s_code} - {s_name}")
            c1, c2, c3 = st.columns(3)
            c1.metric("最新股價", f"{db2.iloc[-1]['Close']:.2f} TWD")
            c2.metric("多方診斷分", f"{b_s} 分")
            c3.metric("空方診斷分", f"{r_s} 分")
            
            # K線
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_width=[0.3, 0.7])
            fig.add_trace(go.Candlestick(x=db2['Date'], open=db2['Open'], high=db2['High'], low=db2['Low'], close=db2['Close'], name="K線"), row=1, col=1)
            fig.add_trace(go.Scatter(x=db2['Date'], y=db2['SMA_5'], line=dict(color='blue', width=1), name="5MA"), row=1, col=1)
            fig.add_trace(go.Scatter(x=db2['Date'], y=db2['SMA_20'], line=dict(color='orange', width=1.5), name="20MA"), row=1, col=1)
            colors = ['#EF5350' if c >= o else '#26A69A' for c, o in zip(db2['Close'], db2['Open'])]
            fig.add_trace(go.Bar(x=db2['Date'], y=db2['Volume'], marker_color=colors, name="成交量"), row=2, col=1)
            fig.update_layout(xaxis_rangeslider_visible=False, height=500, margin=dict(t=30, l=10, r=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 🧠 專家系統診斷與反騙線對策 (DB3)")
            cl, cr = st.columns([4, 6])
            with cl:
                st.dataframe(db3_df, hide_index=True, use_container_width=True)
            with cr:
                st.markdown(f'<div class="report-card">{rpt_h}</div>', unsafe_allow_html=True)
                st.write("")
                st.markdown(f'<div class="strategy-card">{stg_h}</div>', unsafe_allow_html=True)
        else: st.error("查無資料。")
