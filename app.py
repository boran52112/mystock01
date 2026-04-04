import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from FinMind.data import DataLoader
from datetime import datetime, timedelta

# --- 網頁配置 ---
st.set_page_config(page_title="AI 專家全方位診斷系統", layout="wide")

st.markdown("""
    <style>
    .report-title { font-size: 1.25rem; font-weight: bold; color: #1f77b4; margin-bottom: 12px; border-bottom: 2px solid #e1e4e8; padding-bottom: 8px; }
    .report-card { background-color: #ffffff; padding: 25px; border-radius: 15px; border-left: 8px solid #1f77b4; box-shadow: 0 4px 12px rgba(0,0,0,0.08); min-height: 280px; line-height: 1.7; }
    .strategy-card { background-color: #fcfcfc; padding: 25px; border-radius: 15px; border-left: 8px solid #d62728; box-shadow: 0 4px 12px rgba(0,0,0,0.08); min-height: 200px; line-height: 1.7; border: 1px solid #eee; }
    .logic-tag { background-color: #fff3e0; color: #e65100; padding: 2px 8px; border-radius: 4px; font-size: 0.85rem; font-weight: bold; margin-right: 5px; border: 1px solid #ffe0b2; }
    .action-bold { color: #d62728; font-weight: bold; font-size: 1.1rem; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    ul { padding-left: 1.2rem; }
    li { margin-bottom: 12px; }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 AI 專家級 9 大指標診斷與操作系統")

# --- 資料抓取輔助 ---
@st.cache_data(ttl=86400)
def get_stock_chinese_name(code):
    try:
        dl = DataLoader()
        df_info = dl.taiwan_stock_info()
        item = df_info[df_info['stock_id'] == code]
        if not item.empty: return item.iloc[0]['stock_name']
    except: pass
    return None

@st.cache_data(ttl=3600)
def fetch_complete_data(code):
    df_price, actual_sym, final_name = None, None, f"股票 {code}"
    chinese_name = get_stock_chinese_name(code)
    for suffix in [".TW", ".TWO"]:
        try:
            temp_sym = f"{code}{suffix}"
            ticker = yf.Ticker(temp_sym)
            df = ticker.history(period="1y")
            if not df.empty:
                final_name = chinese_name if chinese_name else ticker.info.get('shortName', temp_sym)
                df.reset_index(inplace=True)
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
                df_price, actual_sym = df, temp_sym
                break
        except: continue
    if df_price is None: return None, None, None, False
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
            df_margin.rename(columns={'date': 'Date', 'MarginPurchaseTodayBalance': 'Margin_Bal', 'ShortSaleTodayBalance': 'Short_Bal'}, inplace=True)
        if not df_pivot.empty: df_price = pd.merge(df_price, df_pivot, on='Date', how='left')
        if not df_margin.empty: df_price = pd.merge(df_price, df_margin, on='Date', how='left')
        df_price.fillna(method='ffill', inplace=True).fillna(0, inplace=True)
        return df_price, actual_sym, final_name, True
    except: return df_price, actual_sym, final_name, False

# --- 技術指標計算 ---
def generate_db2(df):
    db2 = df.copy()
    db2['SMA_5'] = db2['Close'].rolling(5).mean()
    db2['SMA_20'] = db2['Close'].rolling(20).mean()
    delta = db2['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    db2['RSI'] = 100 - (100 / (1 + gain/loss))
    exp1 = db2['Close'].ewm(span=12).mean(); exp2 = db2['Close'].ewm(span=26).mean()
    db2['MACD_H'] = (exp1 - exp2) - (exp1 - exp2).ewm(span=9).mean()
    db2['BB_Mid'] = db2['Close'].rolling(20).mean(); std = db2['Close'].rolling(20).std()
    db2['BB_Up'] = db2['BB_Mid'] + 2*std; db2['BB_Low'] = db2['BB_Mid'] - 2*std
    l9, h9 = db2['Low'].rolling(9).min(), db2['High'].rolling(9).max()
    rsv = (db2['Close'] - l9) / (h9 - l9) * 100
    db2['K'] = rsv.ewm(com=2).mean(); db2['D'] = db2['K'].ewm(com=2).mean()
    db2['Prev_Close'] = db2['Close'].shift(1); db2['Prev_High'] = db2['High'].shift(1); db2['Prev_Low'] = db2['Low'].shift(1)
    if 'Margin_Bal' in db2.columns: 
        db2['Margin_Diff'] = db2['Margin_Bal'].diff()
        db2['Short_Diff'] = db2['Short_Bal'].diff()
    return db2.dropna()

# --- 深度專家診斷引擎 ---
def generate_expert_diagnostic_db3(db2, has_chip):
    latest, prev = db2.iloc[-1], db2.iloc[-2]
    bull_score, bear_score = 0, 0
    analysis_table = []
    
    # 權重分配邏輯
    # 1. 趨勢 (W3)
    is_trend_up = latest['SMA_5'] > latest['SMA_20']
    if is_trend_up:
        analysis_table.append(["均線趨勢", "🟢 看多", "多頭排列 (W3)"]); bull_score += 3
    else:
        analysis_table.append(["均線趨勢", "🔴 看空", "空頭死叉 (W3)"]); bear_score += 3

    # 2. 籌碼 (W3)
    inst_net = 0
    if has_chip:
        inst_net = latest['Foreign_Buy'] + latest['Trust_Buy']
        if inst_net > 500:
            analysis_table.append(["法人籌碼", "🟢 看多", "法人積極買超 (W3)"]); bull_score += 3
        elif inst_net < -500:
            analysis_table.append(["法人籌碼", "🔴 看空", "法人積極賣超 (W3)"]); bear_score += 3
        else:
            analysis_table.append(["法人籌碼", "⚪ 中立", "法人觀望"])

    # 3. K線 (W2)
    body = abs(latest['Close'] - latest['Open']); lower_s = min(latest['Close'], latest['Open']) - latest['Low']
    is_shadow_support = lower_s > body * 1.5 and body > 0
    if is_shadow_support:
        analysis_table.append(["K線型態", "🟢 看多", "下影線護盤 (W2)"]); bull_score += 2
    else:
        analysis_table.append(["K線型態", "⚪ 中立", "常規型態"])

    # 4. MACD 動能 (W1)
    is_macd_up = latest['MACD_H'] > 0
    if is_macd_up:
        analysis_table.append(["波段MACD", "🟢 看多", "動能轉強"]); bull_score += 1
    else:
        analysis_table.append(["波段MACD", "🔴 看空", "動能消退"]); bear_score += 1

    # 5-9 其他指標 (W1) - RSI, KD, 布林, 缺口, 散戶
    if latest['RSI'] < 30: bull_score += 1; analysis_table.append(["動能RSI", "🟢 看多", "超賣"])
    elif latest['RSI'] > 70: bear_score += 1; analysis_table.append(["動能RSI", "🔴 看空", "超買"])
    else: analysis_table.append(["動能RSI", "⚪ 中立", "震盪區"])

    # --- 專家深度診斷報告 (矛盾分析) ---
    rpt_html = "<div class='report-title'>🔍 核心矛盾與信度解析</div>"
    insights = []

    # 情況 1: 趨勢與型態的衝突 (您圖片中的狀況：均線空、但影線多)
    if not is_trend_up and is_shadow_support:
        insights.append(f"<li><span class='logic-tag'>跌勢止穩</span> 雖然 <b style='color:red'>均線死叉</b> 顯示趨勢仍偏空，但今日出現 <b style='color:green'>長下影線</b>。這代表低檔買盤轉強，專家認為空方力道已受阻，此時不宜在低位加碼放空，應觀察是否能站回 5MA。</li>")
    
    # 情況 2: 趨勢與籌碼背離 (多頭陷阱)
    if is_trend_up and has_chip and inst_net < -1000:
        insights.append(f"<li><span class='logic-tag'>籌碼警告</span> 股價雖在均線上漲，但法人大幅賣超。這屬於「拉高出貨」背離，技術面的看多訊號信度降至 40%，切勿盲目追價。</li>")
    
    # 情況 3: 動能消退但趨勢仍存
    if is_trend_up and not is_macd_up:
        insights.append(f"<li><span class='logic-tag'>動能背離</span> 趨勢雖多，但 MACD 紅柱縮減。專家判定這是「漲勢衰竭」的徵兆，雖不至於放空，但應考慮分批賣出獲利。</li>")

    # 如果沒有特定大矛盾，也要給出具體的「多空勢力分析」
    if not insights:
        if bull_score > bear_score:
            insights.append(f"<li><b>強勢同步：</b>目前趨勢與動能方向一致，多方佔據主導地位。技術面信度高，適合順勢而為。</li>")
        else:
            insights.append(f"<li><b>弱勢同步：</b>目前指標集體偏空，無明顯反轉跡象。技術面信度高，操作應避開或減碼。</li>")

    rpt_html += "<ul>" + "".join(insights) + "</ul>"

    # --- 操作積極度矩陣 (賣出 vs 放空) ---
    stg_html = "<div class='report-title'>🎯 具體操作策略與積極度</div>"
    diff = bull_score - bear_score
    
    # 判定放空強度
    is_shortable = (bear_score >= 6) and (latest['Close'] < latest['SMA_5'])
    
    stg_html += "<ul>"
    if bull_score >= 7:
        stg_html += f"<li><b class='action-bold'>積極買進 (Buy Aggressive)</b>：趨勢、籌碼與型態皆站在多方。建議以 5MA ({latest['SMA_5']:.1f}) 為基準，分批建倉。</li>"
    elif bear_score >= 7 and is_shortable:
        stg_html += f"<li><b class='action-bold'>積極放空 (Short Sell)</b>：這不僅是賣出現券，更是反向進攻時機。<b>原因：</b>趨勢結構性走壞且無護盤力道。建議以融券或借券賣出參與下殺波段。</li>"
    elif diff < 0:
        stg_html += f"<li><b style='color:#e65100; font-weight:bold;'>消極賣出 (Passive Sell)</b>：目前屬於轉弱但尚未崩盤。建議「獲利了結」或「減碼」，提高現金比例。不建議在此放空，因為下殺動能尚未噴發。</li>"
    else:
        stg_html += f"<li><b>中立觀望</b>：目前多空僵持。建議等待指標出現連續性的法人買/賣超再行動作。</li>"
    stg_html += "</ul>"

    return pd.DataFrame(analysis_table, columns=["維度", "訊號", "解析"]), bull_score, bear_score, rpt_html, stg_html

# --- UI 顯示 ---
s_code = st.text_input("📈 請輸入台股代號", "2330").strip()
if s_code:
    with st.spinner("專家大腦思考中..."):
        df_raw, sym, s_name, has_c = fetch_complete_data(s_code)
        if df_raw is not None:
            db2 = generate_db2(df_raw)
            db3_df, bulls_s, bears_s, rpt_h, stg_h = generate_expert_diagnostic_db3(db2, has_c)
            
            def fmt(v): return f"{v:.2f}".rstrip('0').rstrip('.') if v % 1 != 0 else f"{int(v)}"
            curr_p = db2.iloc[-1]['Close']; diff_p = curr_p - db2.iloc[-2]['Close']
            
            st.subheader(f"📊 分析標的：{s_code} - {s_name}")
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("當前股價", f"{fmt(curr_p)} TWD", f"{diff_p:.2f}")
            col_m2.metric("多方加權分", f"{bulls_s}")
            col_m3.metric("空方加權分", f"{bears_s}")
            
            # K線圖
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_width=[0.3, 0.7])
            fig.add_trace(go.Candlestick(x=db2['Date'], open=db2['Open'], high=db2['High'], low=db2['Low'], close=db2['Close'], name="K線"), row=1, col=1)
            fig.add_trace(go.Scatter(x=db2['Date'], y=db2['SMA_5'], line=dict(color='blue', width=1), name="5MA"), row=1, col=1)
            fig.add_trace(go.Scatter(x=db2['Date'], y=db2['SMA_20'], line=dict(color='orange', width=1.5), name="20MA"), row=1, col=1)
            colors = ['#EF5350' if c >= o else '#26A69A' for c, o in zip(db2['Close'], db2['Open'])]
            fig.add_trace(go.Bar(x=db2['Date'], y=db2['Volume'], marker_color=colors, name="成交量"), row=2, col=1)
            fig.update_layout(xaxis_rangeslider_visible=False, height=500, margin=dict(t=30, l=10, r=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 🧠 專家系統診斷與對策 (DB3)")
            cl, cr = st.columns([4, 6])
            with cl:
                st.dataframe(db3_df, hide_index=True, use_container_width=True)
            with cr:
                st.markdown(f'<div class="report-card">{rpt_h}</div>', unsafe_allow_html=True)
                st.write("")
                st.markdown(f'<div class="strategy-card">{stg_h}</div>', unsafe_allow_html=True)
        else: st.error("查無資料。")
