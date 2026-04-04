import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from FinMind.data import DataLoader
from datetime import datetime, timedelta

# --- 網頁配置 ---
st.set_page_config(page_title="AI 專家級 9 大指標全維度分析", layout="wide")

st.markdown("""
    <style>
    .report-title { font-size: 1.25rem; font-weight: bold; color: #1f77b4; margin-bottom: 12px; border-bottom: 2px solid #e1e4e8; padding-bottom: 8px; }
    .report-card { background-color: #ffffff; padding: 25px; border-radius: 15px; border-left: 8px solid #1f77b4; box-shadow: 0 4px 12px rgba(0,0,0,0.08); min-height: 280px; line-height: 1.7; }
    .strategy-card { background-color: #fcfcfc; padding: 25px; border-radius: 15px; border-left: 8px solid #d62728; box-shadow: 0 4px 12px rgba(0,0,0,0.08); min-height: 200px; line-height: 1.7; border: 1px solid #eee; }
    .logic-tag { background-color: #e3f2fd; color: #0d47a1; padding: 2px 8px; border-radius: 4px; font-size: 0.85rem; font-weight: bold; margin-right: 5px; border: 1px solid #bbdefb; }
    .action-bold { color: #d32f2f; font-weight: bold; font-size: 1.1rem; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    ul { padding-left: 1.2rem; }
    li { margin-bottom: 12px; }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 AI 專家級 9 大指標全維度診斷系統")

# --- 1. 資料抓取模組 (DB1) ---
@st.cache_data(ttl=86400)
def get_stock_name(code):
    try:
        dl = DataLoader()
        df_info = dl.taiwan_stock_info()
        item = df_info[df_info['stock_id'] == code]
        if not item.empty: return item.iloc[0]['stock_name']
    except: pass
    return None

@st.cache_data(ttl=3600)
def fetch_data(code):
    df_p, actual_sym, s_name = None, None, f"股票 {code}"
    c_name = get_stock_name(code)
    for suffix in [".TW", ".TWO"]:
        try:
            temp_sym = f"{code}{suffix}"
            ticker = yf.Ticker(temp_sym)
            df = ticker.history(period="1y")
            if not df.empty:
                s_name = c_name if c_name else ticker.info.get('shortName', temp_sym)
                df.reset_index(inplace=True)
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
                df_p, actual_sym = df, temp_sym
                break
        except: continue
    if df_p is None: return None, None, None, False
    
    try:
        dl = DataLoader()
        start = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        df_i = dl.taiwan_stock_institutional_investors(stock_id=code, start_date=start)
        df_m = dl.taiwan_stock_margin_purchase_short_sale(stock_id=code, start_date=start)
        
        # 籌碼整合邏輯
        df_chip = pd.DataFrame()
        if not df_i.empty:
            df_i = df_i.pivot_table(index='date', columns='name', values='buy_sell', aggfunc='sum').reset_index()
            df_i.rename(columns={'date': 'Date'}, inplace=True)
            f_col, t_col = '外陸資買賣超股數(不含外資自營商)', '投信買賣超股數'
            df_i['Foreign_Buy'] = df_i[f_col] / 1000 if f_col in df_i.columns else 0
            df_i['Trust_Buy'] = df_i[t_col] / 1000 if t_col in df_i.columns else 0
            df_chip = df_i[['Date', 'Foreign_Buy', 'Trust_Buy']]
        
        if not df_m.empty:
            df_m = df_m[['date', 'MarginPurchaseTodayBalance', 'ShortSaleTodayBalance']]
            df_m.rename(columns={'date': 'Date', 'MarginPurchaseTodayBalance': 'Margin_Bal', 'ShortSaleTodayBalance': 'Short_Bal'}, inplace=True)
            if df_chip.empty: df_chip = df_m
            else: df_chip = pd.merge(df_chip, df_m, on='Date', how='outer')
            
        if not df_chip.empty:
            df_p = pd.merge(df_p, df_chip, on='Date', how='left')
        
        df_p.fillna(method='ffill', inplace=True).fillna(0, inplace=True)
        return df_p, actual_sym, s_name, True
    except:
        return df_p, actual_sym, s_name, False

# --- 2. 技術指標計算 (DB2) ---
def generate_db2(df):
    d = df.copy()
    d['SMA_5'] = d['Close'].rolling(5).mean()
    d['SMA_20'] = d['Close'].rolling(20).mean()
    delta = d['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    d['RSI'] = 100 - (100 / (1 + gain/loss))
    exp1 = d['Close'].ewm(span=12).mean(); exp2 = d['Close'].ewm(span=26).mean()
    d['MACD_H'] = (exp1 - exp2) - (exp1 - exp2).ewm(span=9).mean()
    d['BB_Mid'] = d['Close'].rolling(20).mean(); std = d['Close'].rolling(20).std()
    d['BB_Up'] = d['BB_Mid'] + 2*std; d['BB_Low'] = d['BB_Mid'] - 2*std
    l9, h9 = d['Low'].rolling(9).min(), d['High'].rolling(9).max()
    rsv = (d['Close'] - l9) / (h9 - l9) * 100
    d['K'] = rsv.ewm(com=2).mean(); d['D'] = d['K'].ewm(com=2).mean()
    d['Prev_Close'] = d['Close'].shift(1); d['Prev_High'] = d['High'].shift(1); d['Prev_Low'] = d['Low'].shift(1)
    if 'Margin_Bal' in d.columns:
        d['Margin_Diff'] = d['Margin_Bal'].diff()
        d['Short_Diff'] = d['Short_Bal'].diff()
    return d.dropna()

# --- 3. 核心專家診斷引擎 (DB3) ---
def generate_expert_diagnostic(db2, has_chip):
    latest, prev = db2.iloc[-1], db2.iloc[-2]
    bull_score, bear_score = 0, 0
    table_data = []

    # 維度 1: 均線趨勢 (Weight 3)
    is_trend_up = latest['SMA_5'] > latest['SMA_20']
    if is_trend_up:
        table_data.append(["1. 均線趨勢", "🟢 看多", "多頭排列 (W3)"]); bull_score += 3
    else:
        table_data.append(["1. 均線趨勢", "🔴 看空", "空頭死叉 (W3)"]); bear_score += 3

    # 維度 2: 動能 RSI (Weight 1)
    if latest['RSI'] > 75: 
        table_data.append(["2. 動能 RSI", "🔴 看空", "超買警戒"]); bear_score += 1
    elif latest['RSI'] < 25: 
        table_data.append(["2. 動能 RSI", "🟢 看多", "超賣尋支撐"]); bull_score += 1
    else: 
        table_data.append(["2. 動能 RSI", "⚪ 中立", "正常震盪"])

    # 維度 3: 波段 MACD (Weight 1)
    if latest['MACD_H'] > 0:
        table_data.append(["3. 波段 MACD", "🟢 看多", "動能轉強"]); bull_score += 1
    else:
        table_data.append(["3. 波段 MACD", "🔴 看空", "動能偏弱"]); bear_score += 1

    # 維度 4: 布林通道 (Weight 1)
    if latest['Close'] > latest['BB_Up']:
        table_data.append(["4. 布林通道", "🟢 看多", "強勢上攻軌道"]); bull_score += 1
    elif latest['Close'] < latest['BB_Low']:
        table_data.append(["4. 布林通道", "🔴 看空", "破位下殺"]); bear_score += 1
    else:
        table_data.append(["4. 布林通道", "⚪ 中立", "通道內運行"])

    # 維度 5: 短線 KD (Weight 1)
    if latest['K'] > 80:
        table_data.append(["5. KD 指標", "🔴 看空", "高檔超買"]); bear_score += 1
    elif latest['K'] < 20:
        table_data.append(["5. KD 指標", "🟢 看多", "低檔起漲"]); bull_score += 1
    else:
        table_data.append(["5. KD 指標", "⚪ 中立", "盤整中"])

    # 維度 6: K線型態 (Weight 2)
    body = abs(latest['Close'] - latest['Open'])
    lower_s = min(latest['Close'], latest['Open']) - latest['Low']
    is_shadow_support = lower_s > body * 1.5 and body > 0
    if is_shadow_support:
        table_data.append(["6. K線型態", "🟢 看多", "長下影線護盤 (W2)"]); bull_score += 2
    else:
        table_data.append(["6. K線型態", "⚪ 中立", "無特殊型態"])

    # 維度 7: 缺口理論 (Weight 2)
    if latest['Low'] > prev['High']:
        table_data.append(["7. 缺口理論", "🟢 看多", "跳空表態 (W2)"]); bull_score += 2
    elif latest['High'] < prev['Low']:
        table_data.append(["7. 缺口理論", "🔴 看空", "跳空衰竭 (W2)"]); bear_score += 2
    else:
        table_data.append(["7. 缺口理論", "⚪ 中立", "無缺口"])

    # 維度 8: 法人籌碼 (Weight 3)
    if has_chip:
        inst_net = latest['Foreign_Buy'] + latest['Trust_Buy']
        if inst_net > 500:
            table_data.append(["8. 法人籌碼", "🟢 看多", "法人掃貨 (W3)"]); bull_score += 3
        elif inst_net < -500:
            table_data.append(["8. 法人籌碼", "🔴 看空", "法人倒貨 (W3)"]); bear_score += 3
        else:
            table_data.append(["8. 法人籌碼", "⚪ 中立", "法人觀望"])
    else:
        table_data.append(["8. 法人籌碼", "⚪ 未知", "暫無資料"])

    # 維度 9: 散戶籌碼 (Weight 2)
    if has_chip:
        if latest['Margin_Diff'] > 500 and latest['Close'] < prev['Close']:
            table_data.append(["9. 散戶籌碼", "🔴 看空", "資增價跌背離 (W2)"]); bear_score += 2
        elif latest['Margin_Diff'] < -500 and latest['Close'] > prev['Close']:
            table_data.append(["9. 散戶籌碼", "🟢 看多", "資減價揚穩健 (W2)"]); bull_score += 2
        else:
            table_data.append(["9. 散戶籌碼", "⚪ 中立", "散戶穩定"])
    else:
        table_data.append(["9. 散戶籌碼", "⚪ 未知", "暫無資料"])

    # --- 深度專家報告 (矛盾辯論) ---
    rpt_html = "<div class='report-title'>🔍 1. 核心矛盾與信度解析</div>"
    insights = []
    
    # 矛盾 A: 趨勢看空 vs 下影線/籌碼護盤 (您圖片中的狀況)
    if not is_trend_up and (is_shadow_support or (has_chip and inst_net > 500)):
        insights.append(f"<li><span class='logic-tag'>底部背離</span> 目前 <b style='color:red'>均線空頭</b> 指向跌勢，但盤面出現了 <b style='color:green'>長下影線/法人護盤</b>。專家解析：這意味著空方動能遭遇實質買盤抵抗，下跌信度大幅降低，此處絕不可加碼放空，反而應留意止跌訊號。</li>")
    
    # 矛盾 B: 價漲但法人/散戶指標背離
    if latest['Close'] > prev['Close'] and has_chip and inst_net < -1000:
        insights.append(f"<li><span class='logic-tag'>多頭陷阱</span> 價格雖反彈，但 <b style='color:red'>法人大幅賣出</b>。這屬於籌碼面否定技術面的典型狀況，此上漲多為散戶追價，信度極低，應趁反彈賣出。</li>")

    # 矛盾 C: 噴發鈍化
    if is_trend_up and latest['RSI'] > 75:
        insights.append(f"<li><span class='logic-tag'>強勢鈍化</span> 均線趨勢極強但 RSI 指示超買。專家解析：強勢股噴發期會導致超買指標失效，此時應忽略 RSI，操作改以 5MA 不破為持有基準。</li>")

    if not insights:
        insights.append("<li>目前 9 大指標方向大致相符。專家解析：盤勢邏輯單一，無論多空皆具備較高的執行信度，無顯著背離風險。</li>")
    
    rpt_html += "<ul>" + "".join(insights) + "</ul>"

    # --- 操作策略 (消極避險 vs 積極進攻) ---
    stg_html = "<div class='report-title'>🎯 2. 具體操作策略與積極度</div>"
    diff = bull_score - bear_score
    is_major_break = latest['Close'] < latest['SMA_20'] and latest['SMA_5'] < latest['SMA_20']
    
    if diff >= 7:
        stg_html += f"<ul><li><b class='action-bold' style='color:#2e7d32'>積極進場做多 (Strong Buy)</b></li><li>理由：權重指標（趨勢+籌碼）與型態全數同步，勝率極高，適合擴大部位。</li></ul>"
    elif diff <= -7:
        stg_html += f"<ul><li><b class='action-bold'>積極融券放空 (Short Sell) ★★★</b></li><li>理由：趨勢崩壞且法人棄守。這不是普通賣出，而是具備下殺慣性的表態。建議積極利用融券工具獲利。</li></ul>"
    elif diff < 0:
        stg_html += f"<ul><li><b style='color:#e65100; font-weight:bold;'>消極避險 (Passive Sell)</b></li><li>理由：雖然指標偏空，但尚未出現崩潰型訊號（如法人大賣或破底）。目前僅建議「賣出持股」降低風險，不建議積極反向放空。</li></ul>"
    else:
        stg_html += "<ul><li><b>中立觀望 (Neutral)</b></li><li>理由：多空分數接近，矛盾指標互相抵銷，目前市場缺乏主導力量。</li></ul>"

    return pd.DataFrame(table_data, columns=["分析維度", "訊號", "專家狀態解析"]), bull_score, bear_score, rpt_html, stg_html

# --- UI 顯示模組 ---
s_code = st.text_input("📈 請輸入台股代號", "2330").strip()
if s_code:
    with st.spinner("AI 專家大腦深度分析中..."):
        df_raw, sym, s_name, has_c = fetch_data(s_code)
        if df_raw is not None:
            db2 = generate_db2(df_raw)
            db3_df, b_score, r_score, rpt_h, stg_h = generate_expert_diagnostic(db2, has_c)
            
            def fmt(v): return f"{v:.2f}".rstrip('0').rstrip('.') if v % 1 != 0 else f"{int(v)}"
            curr_p = db2.iloc[-1]['Close']; diff_p = curr_p - db2.iloc[-2]['Close']
            
            st.subheader(f"📊 分析對象：{s_code} - {s_name}")
            col1, col2, col3 = st.columns(3)
            col1.metric("最新股價", f"{fmt(curr_p)} TWD", f"{diff_p:.2f}")
            col2.metric("多方加權分", f"{b_score} 分")
            col3.metric("空方加權分", f"{r_score} 分")
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_width=[0.3, 0.7])
            fig.add_trace(go.Candlestick(x=db2['Date'], open=db2['Open'], high=db2['High'], low=db2['Low'], close=db2['Close'], name="K線"), row=1, col=1)
            fig.add_trace(go.Scatter(x=db2['Date'], y=db2['SMA_5'], line=dict(color='blue', width=1), name="5MA"), row=1, col=1)
            fig.add_trace(go.Scatter(x=db2['Date'], y=db2['SMA_20'], line=dict(color='orange', width=1.5), name="20MA"), row=1, col=1)
            colors = ['#EF5350' if c >= o else '#26A69A' for c, o in zip(db2['Close'], db2['Open'])]
            fig.add_trace(go.Bar(x=db2['Date'], y=db2['Volume'], marker_color=colors, name="成交量"), row=2, col=1)
            fig.update_layout(xaxis_rangeslider_visible=False, height=500, margin=dict(t=30, l=10, r=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 🧠 專家系統診斷與全維度對策 (DB3)")
            cl, cr = st.columns([4, 6])
            with cl:
                st.dataframe(db3_df, hide_index=True, use_container_width=True)
            with cr:
                st.markdown(f'<div class="report-card">{rpt_h}</div>', unsafe_allow_html=True)
                st.write("")
                st.markdown(f'<div class="strategy-card">{stg_h}</div>', unsafe_allow_html=True)
        else: st.error("查無資料。")
