import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from FinMind.data import DataLoader
from datetime import datetime, timedelta

# --- 網頁配置 ---
st.set_page_config(page_title="AI 專家大腦 - 9 大指標深度分析", layout="wide")

# 專業 UI 樣式表
st.markdown("""
    <style>
    .report-title { font-size: 1.25rem; font-weight: bold; color: #1f77b4; margin-bottom: 12px; border-bottom: 2px solid #e1e4e8; padding-bottom: 8px; display: flex; align-items: center; }
    .report-card { background-color: #ffffff; padding: 25px; border-radius: 15px; border-left: 8px solid #1f77b4; box-shadow: 0 4px 12px rgba(0,0,0,0.08); min-height: 280px; line-height: 1.7; }
    .strategy-card { background-color: #fcfcfc; padding: 25px; border-radius: 15px; border-left: 8px solid #d62728; box-shadow: 0 4px 12px rgba(0,0,0,0.08); min-height: 200px; line-height: 1.7; border: 1px solid #eee; }
    .logic-tag { background-color: #e1f5fe; color: #01579b; padding: 2px 8px; border-radius: 4px; font-size: 0.85rem; font-weight: bold; margin-right: 5px; }
    .action-high { color: #d62728; font-weight: bold; text-decoration: underline; }
    .action-low { color: #2ca02c; font-weight: bold; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    ul { padding-left: 1.2rem; }
    li { margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧠 AI 專家大腦：全維度指標決策系統")

# --- 資料抓取模組 (DB1) ---
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

# --- 技術指標計算 (DB2) ---
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

# --- 核心專家分析引擎 (DB3) ---
def generate_expert_logic_db3(db2, has_chip):
    latest, prev = db2.iloc[-1], db2.iloc[-2]
    bull_score, bear_score = 0, 0
    analysis_table = []
    
    # 指標定義與權重分配 (Weight System)
    # 權重 3: 均線, 法人 / 權重 2: K線, 缺口, 散戶背離 / 權重 1: RSI, MACD, KD, 布林
    
    # 1. 均線 (W3)
    if latest['SMA_5'] > latest['SMA_20']:
        analysis_table.append(["均線趨勢", "🟢 看多", "多頭排列 (W3)"]); bull_score += 3
    else:
        analysis_table.append(["均線趨勢", "🔴 看空", "空頭死叉 (W3)"]); bear_score += 3

    # 2. 法人 (W3)
    if has_chip:
        inst_net = latest['Foreign_Buy'] + latest['Trust_Buy']
        if inst_net > 800:
            analysis_table.append(["法人籌碼", "🟢 看多", "法人掃貨 (W3)"]); bull_score += 3
        elif inst_net < -800:
            analysis_table.append(["法人籌碼", "🔴 看空", "法人倒貨 (W3)"]); bear_score += 3
        else:
            analysis_table.append(["法人籌碼", "⚪ 中立", "法人觀望"])

    # 3. 散戶背離 (W2)
    if has_chip:
        if latest['Margin_Diff'] > 500 and latest['Close'] < prev['Close']:
            analysis_table.append(["散戶籌碼", "🔴 看空", "資增價跌背離 (W2)"]); bear_score += 2
        elif latest['Margin_Diff'] < -500 and latest['Close'] > prev['Close']:
            analysis_table.append(["散戶籌碼", "🟢 看多", "資減價揚穩健 (W2)"]); bull_score += 2
        else:
            analysis_table.append(["散戶籌碼", "⚪ 中立", "籌碼穩定"])

    # 4. K線型態 (W2)
    body = abs(latest['Close'] - latest['Open']); lower_s = min(latest['Close'], latest['Open']) - latest['Low']
    if lower_s > body * 2:
        analysis_table.append(["K線型態", "🟢 看多", "長下影支撐 (W2)"]); bull_score += 2
    else:
        analysis_table.append(["K線型態", "⚪ 中立", "無特殊型態"])

    # 5. 缺口 (W2)
    if latest['Low'] > prev['High']:
        analysis_table.append(["跳空缺口", "🟢 看多", "向上跳空表態 (W2)"]); bull_score += 2
    elif latest['High'] < prev['Low']:
        analysis_table.append(["跳空缺口", "🔴 看空", "向下跳空衰竭 (W2)"]); bear_score += 2
    else:
        analysis_table.append(["跳空缺口", "⚪ 中立", "無缺口"])

    # 6. RSI (W1)
    if latest['RSI'] > 75: bear_score += 1; analysis_table.append(["動能RSI", "🔴 看空", "高檔超買"])
    elif latest['RSI'] < 25: bull_score += 1; analysis_table.append(["動能RSI", "🟢 看多", "低階超賣"])
    else: analysis_table.append(["動能RSI", "⚪ 中立", "正常震盪"])

    # 7. MACD (W1)
    if latest['MACD_H'] > 0: bull_score += 1; analysis_table.append(["波段MACD", "🟢 看多", "動能增強"])
    else: bear_score += 1; analysis_table.append(["波段MACD", "🔴 看空", "動能消退"])

    # 8. KD (W1)
    if latest['K'] > 80: bear_score += 1; analysis_table.append(["短線KD", "🔴 看空", "高檔區"])
    elif latest['K'] < 20: bull_score += 1; analysis_table.append(["短線KD", "🟢 看多", "低檔區"])
    else: analysis_table.append(["短線KD", "⚪ 中立", "常規區"])

    # 9. 布林 (W1)
    if latest['Close'] > latest['BB_Up']: bull_score += 1; analysis_table.append(["布林通道", "🟢 看多", "強勢過熱"])
    elif latest['Close'] < latest['BB_Low']: bear_score += 1; analysis_table.append(["布林通道", "🔴 看空", "弱勢破底"])
    else: analysis_table.append(["布林通道", "⚪ 中立", "軌道內"])

    # --- 矛盾解析與診斷 (The Logic Brain) ---
    rpt_html = "<div class='report-title'>🔍 核心矛盾與信度解析</div>"
    conflicts = []
    
    # 矛盾A: 趨勢強但指標弱 (鈍化)
    if latest['SMA_5'] > latest['SMA_20'] and (latest['RSI'] > 75 or latest['K'] > 80):
        conflicts.append("<li><span class='logic-tag'>趨勢鈍化</span> 均線強多但震盪指標超買。<b>原因：</b>強勢股常見的高檔鈍化現象。<b>解析：</b>此時 RSI/KD 的看空訊號權重失效，應忽略該雜訊，以趨勢為準。</li>")
    
    # 矛盾B: 價漲籌碼空 (多頭陷阱)
    if latest['Close'] > prev['Close'] and has_chip and inst_net < -1000:
        conflicts.append("<li><span class='logic-tag'>多頭陷阱</span> 股價雖漲但法人大幅撤退。<b>原因：</b>這屬於「拉高出貨」的經典背離。<b>解析：</b>此上漲信度極低，極大機率是假突破，切勿追高。</li>")
    
    # 矛盾C: 價跌籌碼多 (底部護盤)
    if latest['Close'] < prev['Close'] and has_chip and inst_net > 1000:
        conflicts.append("<li><span class='logic-tag'>底部吸籌</span> 股價修正但法人逆勢大買。<b>原因：</b>主力趁亂進場護盤或分批建倉。<b>解析：</b>短線雖然看空，但長線支撐極強，不宜在此殺低。</li>")

    if not conflicts:
        rpt_html += "<div><b>指標一致性：</b>目前技術面與籌碼面方向大致相符，無顯著矛盾，信度極高。</div>"
    else:
        rpt_html += "<ul>" + "".join(conflicts) + "</ul>"

    # --- 操作策略與強度分級 (Intensity Matrix) ---
    stg_html = "<div class='report-title'>🎯 操作策略與積極度分級</div>"
    diff = bull_score - bear_score
    
    # 持股者建議
    stg_html += "<b>【針對現有持股者】</b><ul>"
    if diff >= 5:
        stg_html += "<li><b>建議動作：</b><span class='action-low'>持股續抱 / 甚至加碼</span></li><li><b>強度理由：</b>趨勢與籌碼具備雙重優勢，目前無須預設高點，直到破 5MA 再離場。</li>"
    elif diff <= -5:
        stg_html += "<li><b>建議動作：</b><span class='action-high'>立即清倉撤退</span></li><li><b>強度理由：</b>趨勢結構已破壞且法人棄守。這不是修正，而是反轉。</li>"
    else:
        stg_html += "<li><b>建議動作：</b>分批減碼，提高現金水位</li><li><b>強度理由：</b>指標出現分歧，市場動能不足，應先回收本金觀望。</li>"
    stg_html += "</ul>"

    # 空手者建議 (積極度區分)
    stg_html += "<b>【針對空手進場者】</b><ul>"
    if diff >= 7:
        stg_html += "<li><b>建議動作：</b>積極進場 (分批建立多單)</li><li><b>理由：</b>高分信度，勝率較大。</li>"
    elif diff <= -7:
        stg_html += "<li><b>建議動作：</b><span class='action-high'>積極融券放空 ★★★</span></li><li><b>強度理由：</b>不僅是看空，這屬於「崩潰型」訊號。融資若開始停損將引發斷頭潮，下殺動能充足，是高獲利放空契機。</li>"
    elif diff < 0:
        stg_html += "<li><b>建議動作：</b>消極放空或觀望</li><li><b>強度理由：</b>雖看空但動能不足，放空風險報酬比不佳，僅建議套現觀望。</li>"
    else:
        stg_html += "<li><b>建議動作：</b>耐心觀望</li><li><b>理由：</b>等待下一個權重指標（如法人轉買）表態。</li>"
    stg_html += "</ul>"

    return pd.DataFrame(analysis_table, columns=["維度", "訊號", "解析"]), bull_score, bear_score, rpt_html, stg_html

# --- UI 呈現 ---
s_code = st.text_input("📈 請輸入台股代號 (如 2330, 5490, 8069)", "2330").strip()
if s_code:
    with st.spinner("AI 專家大腦運算中..."):
        df_raw, sym, s_name, has_c = fetch_complete_data(s_code)
        if df_raw is not None:
            db2 = generate_db2(df_raw)
            db3_df, bulls_s, bears_s, rpt_h, stg_h = generate_expert_logic_db3(db2, has_c)
            
            # 格式化
            def fmt(v): return f"{v:.2f}".rstrip('0').rstrip('.') if v % 1 != 0 else f"{int(v)}"
            curr_p = db2.iloc[-1]['Close']; diff_p = curr_p - db2.iloc[-2]['Close']
            
            st.subheader(f"📊 分析標的：{s_code} - {s_name}")
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("最新收盤價", f"{fmt(curr_p)} TWD", f"{diff_p:.2f}")
            col_m2.metric("多方加權總分", f"{bulls_s} 分")
            col_m3.metric("空方加權總分", f"{bears_s} 分")
            
            # 圖表
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
        else:
            st.error("查無資料，請確認代號是否正確。")
