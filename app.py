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

# 強化 CSS：去除 Markdown 符號感，改用專業卡片與條列樣式
st.markdown("""
    <style>
    .report-title { font-size: 1.2rem; font-weight: bold; color: #1f77b4; margin-bottom: 10px; border-bottom: 1px solid #eee; padding-bottom: 5px; }
    .report-card { background-color: #ffffff; padding: 25px; border-radius: 12px; border-left: 6px solid #1f77b4; box-shadow: 0 4px 6px rgba(0,0,0,0.05); min-height: 250px; line-height: 1.6; }
    .strategy-card { background-color: #f8f9fa; padding: 25px; border-radius: 12px; border-left: 6px solid #ff4b4b; min-height: 150px; line-height: 1.6; }
    .status-item { margin-bottom: 8px; list-style-type: none; padding-left: 0; }
    .highlight-blue { color: #1f77b4; font-weight: bold; }
    .highlight-red { color: #d62728; font-weight: bold; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 AI 專家級 9 大指標技術分析系統")

# --- 輔助：取得中文名稱 ---
@st.cache_data(ttl=86400)
def get_stock_chinese_name(code):
    try:
        dl = DataLoader()
        df_info = dl.taiwan_stock_info()
        item = df_info[df_info['stock_id'] == code]
        if not item.empty: return item.iloc[0]['stock_name']
    except: pass
    return None

# --- 資料抓取 (DB1) ---
@st.cache_data(ttl=3600)
def fetch_all_data(code):
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

# --- 指標與專家引擎 (DB2 & DB3) ---
def generate_db2(df):
    db2 = df.copy()
    db2['SMA_5'] = db2['Close'].rolling(5).mean()
    db2['SMA_20'] = db2['Close'].rolling(20).mean()
    delta = db2['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    db2['RSI'] = 100 - (100 / (1 + gain/loss))
    exp1 = db2['Close'].ewm(span=12).mean(); exp2 = db2['Close'].ewm(span=26).mean()
    db2['MACD_H'] = (exp1 - exp2) - (exp1 - exp2).ewm(span=9).mean()
    db2['BB_Mid'] = db2['Close'].rolling(20).mean(); std = db2['Close'].rolling(20).std()
    db2['BB_Up'] = db2['BB_Mid'] + 2*std; db2['BB_Low'] = db2['BB_Mid'] - 2*std
    l9, h9 = db2['Low'].rolling(9).min(), db2['High'].rolling(9).max()
    rsv = (db2['Close'] - l9) / (h9 - l9) * 100
    db2['K'] = rsv.ewm(com=2).mean(); db2['D'] = db2['K'].ewm(com=2).mean()
    db2['Prev_Close'] = db2['Close'].shift(1); db2['Prev_High'] = db2['High'].shift(1); db2['Prev_Low'] = db2['Low'].shift(1)
    if 'Margin_Bal' in db2.columns: db2['Margin_Diff'] = db2['Margin_Bal'].diff()
    return db2.dropna()

def generate_expert_db3(db2, has_chip):
    latest, prev = db2.iloc[-1], db2.iloc[-2]
    analysis = []
    bull_score, bear_score = 0, 0

    # 1.均線 2.RSI 3.MACD 4.布林 5.KD 6.K線 7.缺口 8.法人 9.散戶
    checks = [
        ("均線理論", latest['SMA_5'] > latest['SMA_20'], "🟢 看多", "多頭排列", "🔴 看空", "趨勢偏弱"),
        ("動能 (RSI)", latest['RSI'] < 25, "🟢 看多", "超賣支撐", "🔴 看空" if latest['RSI'] > 75 else "⚪ 中立", "正常震盪"),
        ("波段 (MACD)", latest['MACD_H'] > 0, "🟢 看多", "紅柱增長", "🔴 看空", "動能轉弱"),
        ("布林通道", latest['Close'] > latest['BB_Up'], "🟢 看多", "強勢噴發", "🔴 看空" if latest['Close'] < latest['BB_Low'] else "⚪ 中立", "通道內"),
        ("KD指標", latest['K'] < 20, "🟢 看多", "低階起漲", "🔴 看空" if latest['K'] > 80 else "⚪ 中立", "正常區"),
        ("缺口理論", latest['Low'] > prev['High'], "🟢 看多", "跳空強勢", "🔴 看空" if latest['High'] < prev['Low'] else "⚪ 中立", "無缺口")
    ]
    for n, cond_up, s_up, d_up, s_dw, d_dw in checks:
        if cond_up: analysis.append([n, s_up, d_up]); bull_score += 1
        elif "看空" in s_dw: analysis.append([n, s_dw, d_dw]); bear_score += 1
        else: analysis.append([n, s_dw, d_dw])

    body = abs(latest['Close'] - latest['Open']); lower_s = min(latest['Close'], latest['Open']) - latest['Low']
    if lower_s > body * 1.5: analysis.append(["K線型態", "🟢 看多", "下影線支撐"]); bull_score += 1
    else: analysis.append(["K線型態", "⚪ 中立", "常規震盪"])

    if has_chip:
        inst_net = latest['Foreign_Buy'] + latest['Trust_Buy']
        if inst_net > 500: analysis.append(["法人籌碼", "🟢 看多", f"買超 {int(inst_net)}張"]); bull_score += 1
        elif inst_net < -500: analysis.append(["法人籌碼", "🔴 看空", f"賣超 {int(inst_net)}張"]); bear_score += 1
        else: analysis.append(["法人籌碼", "⚪ 中立", "觀望"])
        if latest['Margin_Diff'] < -500: analysis.append(["散戶籌碼", "🟢 看多", "資減籌碼穩"]); bull_score += 1
        else: analysis.append(["散戶籌碼", "⚪ 中立", "穩定"])
    else:
        analysis.append(["法人籌碼", "⚪ 未知", "無資料"]); analysis.append(["散戶籌碼", "⚪ 未知", "無資料"])

    # --- 輸出 HTML 格式內容 (去除 Markdown 符號) ---
    rpt_html = "<div class='report-title'>🔍 1. 核心矛盾與信度解析</div>"
    mood = "偏多" if bull_score > bear_score else "偏空" if bear_score > bull_score else "中立盤整"
    rpt_html += f"<div><b>盤勢總結：</b>目前市場呈現 <span class='highlight-blue'>{mood}</span> 態勢。看多指標 {bull_score} 項，看空 {bear_score} 項。</div><br>"
    
    rpt_html += "<ul>"
    conflict = False
    if latest['SMA_5'] > latest['SMA_20'] and latest['RSI'] > 75:
        rpt_html += "<li><b>強勢鈍化警告：</b>均線雖然多頭，但 RSI 已進入極度超買區。專家建議此時不應盲目追高，應以 5MA 為防守線。</li>"
        conflict = True
    if has_chip and (latest['Close'] > prev['Close'] and inst_net < -1000):
        rpt_html += "<li><b>籌碼背離：</b>今日股價雖漲但法人大賣。專家提醒這可能是散戶盤或主力拉高出貨，信度存疑。</li>"
        conflict = True
    if not conflict:
        rpt_html += "<li><b>趨勢評論：</b>各項指標方向大致同調，目前無明顯背離現象。建議順著趨勢操作即可。</li>"
    if not has_chip:
        rpt_html += "<li><b>資料提醒：</b>目前缺少本土籌碼數據，僅以技術型態判定，信度約為 65%。</li>"
    rpt_html += "</ul>"

    stg_html = "<div class='report-title'>🎯 2. 具體操作策略與動作</div>"
    entry_p = f"{latest['SMA_5']:.2f}"; stop_p = f"{(latest['Close']*0.95):.2f}"
    if bull_score >= 6:
        stg_html += f"<b>【積極做多】</b><br><li><b>動作：</b>建議在 <b>{entry_p} 元</b> 附近分批介入。</li><li><b>目標：</b>上看布林上軌 <b>{latest['BB_Up']:.2f} 元</b>。</li><li><b>防守：</b>跌破 <b>{stop_p} 元</b> 應止損。</li>"
    elif bear_score >= 6:
        stg_html += f"<b>【嚴格觀望】</b><br><li><b>動作：</b>空方勢力強，不宜接刀。</li><li><b>轉強點：</b>需等站回 20MA (<b>{latest['SMA_20']:.2f} 元</b>) 再考慮。</li>"
    else:
        stg_html += f"<b>【區間震盪】</b><br><li><b>動作：</b>在 <b>{latest['BB_Low']:.2f}</b> 至 <b>{latest['BB_Up']:.2f}</b> 之間低買高賣。</li><li><b>策略：</b>等待 MACD 紅柱再次伸長為加碼訊號。</li>"

    return pd.DataFrame(analysis, columns=["分析維度", "訊號", "描述"]), bull_score, bear_score, rpt_html, stg_html

# --- UI 呈現 ---
s_code = st.text_input("📈 請輸入台股代號", "2330").strip()
if s_code:
    with st.spinner("專家系統分析中..."):
        df_raw, actual_sym, s_name, has_c = fetch_all_data(s_code)
        if df_raw is not None:
            db2 = generate_db2(df_raw)
            db3_df, bulls, bears, rpt_h, stg_h = generate_expert_db3(db2, has_c)
            def fmt(v): return f"{v:.2f}".rstrip('0').rstrip('.') if v % 1 != 0 else f"{int(v)}"
            curr_p = db2.iloc[-1]['Close']; diff = curr_p - db2.iloc[-2]['Close']
            
            st.subheader(f"📊 分析標的：{s_code} - {s_name}")
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("當前股價", f"{fmt(curr_p)} TWD", f"{diff:.2f}")
            col_m2.metric("看多指標", f"{bulls} / 9")
            col_m3.metric("看空指標", f"{bears} / 9")
            
            # K線
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_width=[0.3, 0.7])
            fig.add_trace(go.Candlestick(x=db2['Date'], open=db2['Open'], high=db2['High'], low=db2['Low'], close=db2['Close'], name="K線"), row=1, col=1)
            fig.add_trace(go.Scatter(x=db2['Date'], y=db2['SMA_5'], line=dict(color='blue', width=1), name="5MA"), row=1, col=1)
            fig.add_trace(go.Scatter(x=db2['Date'], y=db2['SMA_20'], line=dict(color='orange', width=1.5), name="20MA"), row=1, col=1)
            colors = ['#EF5350' if c >= o else '#26A69A' for c, o in zip(db2['Close'], db2['Open'])]
            fig.add_trace(go.Bar(x=db2['Date'], y=db2['Volume'], marker_color=colors, name="成交量"), row=2, col=1)
            fig.update_layout(xaxis_rangeslider_visible=False, height=500, margin=dict(t=30, l=10, r=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 🧠 專家系統研判 (DB3)")
            cl, cr = st.columns([4, 6])
            with cl:
                st.dataframe(db3_df, hide_index=True, use_container_width=True)
            with cr:
                # 使用 HTML 注入，避開 Markdown 符號
                st.markdown(f'<div class="report-card">{rpt_h}</div>', unsafe_allow_html=True)
                st.write("")
                st.markdown(f'<div class="strategy-card">{stg_h}</div>', unsafe_allow_html=True)
        else:
            st.error("查無資料，請確認代號。")
