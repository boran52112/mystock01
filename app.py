import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from FinMind.data import DataLoader
from datetime import datetime, timedelta

# --- 網頁配置 ---
st.set_page_config(page_title="AI 專家全維度決策系統", layout="wide")

# 專業 HTML/CSS 樣式表 (確保絕不留白且排版精美)
st.markdown("""
    <style>
    .main-title { font-size: 2rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 20px; }
    .report-card { background-color: #ffffff; padding: 25px; border-radius: 15px; border-left: 10px solid #1f77b4; box-shadow: 0 4px 15px rgba(0,0,0,0.1); min-height: 400px; line-height: 1.8; }
    .strategy-card { background-color: #fdfdfd; padding: 25px; border-radius: 15px; border-left: 10px solid #d62728; box-shadow: 0 4px 15px rgba(0,0,0,0.1); border: 1px solid #eee; line-height: 1.8; }
    .section-header { font-size: 1.3rem; font-weight: bold; color: #333; margin-top: 15px; border-bottom: 2px solid #eee; padding-bottom: 5px; margin-bottom: 15px; }
    .logic-tag { background-color: #e3f2fd; color: #0d47a1; padding: 3px 10px; border-radius: 6px; font-size: 0.9rem; font-weight: bold; margin-right: 10px; border: 1px solid #bbdefb; }
    .action-active { color: #d32f2f; font-weight: bold; font-size: 1.15rem; text-decoration: underline; }
    .action-passive { color: #2e7d32; font-weight: bold; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    ul { padding-left: 1.2rem; }
    li { margin-bottom: 12px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ AI 專家大腦：全維度指標偵探決策系統")

# --- 支柱 5：資料完整性與中文名稱 ---
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
def fetch_complete_data(code):
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
        start = (datetime.now() - timedelta(days=65)).strftime('%Y-%m-%d')
        df_i = dl.taiwan_stock_institutional_investors(stock_id=code, start_date=start)
        df_m = dl.taiwan_stock_margin_purchase_short_sale(stock_id=code, start_date=start)
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
            df_chip = pd.merge(df_chip, df_m, on='Date', how='outer') if not df_chip.empty else df_m
            
        if not df_chip.empty: df_p = pd.merge(df_p, df_chip, on='Date', how='left')
        df_p.fillna(method='ffill', inplace=True).fillna(0, inplace=True)
        return df_p, actual_sym, s_name, True
    except:
        return df_p, actual_sym, s_name, False

# --- 支柱 1 & 建議功能 1：技術指標與連貫性趨勢 ---
def generate_db2(df):
    d = df.copy()
    d['SMA_5'] = d['Close'].rolling(5).mean(); d['SMA_20'] = d['Close'].rolling(20).mean()
    delta = d['Close'].diff(); gain = delta.where(delta > 0, 0).rolling(14).mean(); loss = -delta.where(delta < 0, 0).rolling(14).mean()
    d['RSI'] = 100 - (100 / (1 + gain/loss))
    exp1 = d['Close'].ewm(span=12).mean(); exp2 = d['Close'].ewm(span=26).mean()
    d['MACD_H'] = (exp1 - exp2) - (exp1 - exp2).ewm(span=9).mean()
    d['BB_Mid'] = d['Close'].rolling(20).mean(); std = d['Close'].rolling(20).std()
    d['BB_Up'] = d['BB_Mid'] + 2*std; d['BB_Low'] = d['BB_Mid'] - 2*std
    l9, h9 = d['Low'].rolling(9).min(), d['High'].rolling(9).max()
    rsv = (d['Close'] - l9) / (h9 - l9) * 100
    d['K'] = rsv.ewm(com=2).mean(); d['D'] = d['K'].ewm(com=2).mean()
    d['Margin_Diff'] = d['Margin_Bal'].diff() if 'Margin_Bal' in d.columns else 0
    d['Vol_Avg'] = d['Volume'].rolling(5).mean()
    
    # 建議功能 1：近期趨勢連貫性 (3日籌碼/量價趨勢)
    if 'Foreign_Buy' in d.columns:
        d['Inst_3D_Sum'] = (d['Foreign_Buy'] + d['Trust_Buy']).rolling(3).sum()
    d['Price_3D_Change'] = d['Close'].diff(3)
    
    return d.dropna()

# --- 支柱 2, 3, 4 & 建議功能 2：專家診斷引擎 ---
def generate_expert_brain(db2, has_chip):
    latest, prev = db2.iloc[-1], db2.iloc[-2]
    bull_score, bear_score = 0, 0
    table_data = []

    # 1. 均線趨勢 (W3)
    is_trend_up = latest['SMA_5'] > latest['SMA_20']
    table_data.append(["1. 均線趨勢", "🟢 看多" if is_trend_up else "🔴 看空", "多頭排列 (W3)" if is_trend_up else "空頭排列 (W3)"])
    if is_trend_up: bull_score += 3
    else: bear_score += 3

    # 2. 動能 RSI (W1)
    rsi_sig = "🔴 超買" if latest['RSI'] > 75 else "🟢 超賣" if latest['RSI'] < 25 else "⚪ 正常"
    table_data.append(["2. 動能 RSI", rsi_sig, f"目前數值 {latest['RSI']:.1f}"])
    if latest['RSI'] > 75: bear_score += 1
    elif latest['RSI'] < 25: bull_score += 1

    # 3. 波段 MACD (W1)
    macd_up = latest['MACD_H'] > 0
    table_data.append(["3. 波段 MACD", "🟢 看多" if macd_up else "🔴 看空", "紅柱向上" if macd_up else "綠柱向下"])
    if macd_up: bull_score += 1
    else: bear_score += 1

    # 4. 布林通道 (W1)
    if latest['Close'] > latest['BB_Up']:
        table_data.append(["4. 布林通道", "🟢 看多", "強勢上軌突破"]); bull_score += 1
    elif latest['Close'] < latest['BB_Low']:
        table_data.append(["4. 布林通道", "🔴 看空", "弱勢下軌跌破"]); bear_score += 1
    else:
        table_data.append(["4. 布林通道", "⚪ 中立", "通道內盤整"])

    # 5. 短線 KD (W1)
    kd_up = latest['K'] > latest['D']
    table_data.append(["5. KD 指標", "🟢 看多" if kd_up else "🔴 看空", "黃金交叉" if kd_up else "死亡交叉"])
    if kd_up: bull_score += 1
    else: bear_score += 1

    # 6. K線型態 (W2)
    body = abs(latest['Close'] - latest['Open']); lower_s = min(latest['Close'], latest['Open']) - latest['Low']
    is_shadow = lower_s > body * 1.5 and body > 0
    table_data.append(["6. K線型態", "🟢 看多" if is_shadow else "⚪ 中立", "長下影線護盤" if is_shadow else "一般型態"])
    if is_shadow: bull_score += 2

    # 7. 缺口理論 (W2)
    is_gap = latest['Low'] > prev['High']
    table_data.append(["7. 缺口理論", "🟢 看多" if is_gap else "⚪ 中立", "向上跳空表態" if is_gap else "無跳空"])
    if is_gap: bull_score += 2

    # 8. 法人籌碼 (W3)
    inst_net = 0
    if has_chip:
        inst_net = latest['Foreign_Buy'] + latest['Trust_Buy']
        if inst_net > 500: bull_score += 3; table_data.append(["8. 法人籌碼", "🟢 看多", f"大買 {int(inst_net)}張"])
        elif inst_net < -500: bear_score += 3; table_data.append(["8. 法人籌碼", "🔴 看空", f"大賣 {int(inst_net)}張"])
        else: table_data.append(["8. 法人籌碼", "⚪ 中立", "法人觀望"])
    else:
        table_data.append(["8. 法人籌碼", "⚪ 未知", "無籌碼資料"])

    # 9. 散戶籌碼 (W2)
    if has_chip:
        if latest['Margin_Diff'] > 500 and latest['Close'] < prev['Close']:
            bear_score += 2; table_data.append(["9. 散戶籌碼", "🔴 看空", "資增價跌背離"])
        else:
            table_data.append(["9. 散戶籌碼", "⚪ 中立", "籌碼相對穩定"])
    else:
        table_data.append(["9. 散戶籌碼", "⚪ 未知", "無籌碼資料"])

    # --- 核心辯證與反騙線診斷 (Pillar 2 & 3 & Enhancement 2) ---
    rpt_html = "<div class='section-header'>🔍 1. 專家矛盾診斷與信度解析</div><ul>"
    
    # 建議功能 2：市場環境判定
    is_range = abs(latest['SMA_5'] - latest['SMA_20']) / latest['SMA_20'] < 0.02
    env_text = "目前市場屬於 <b>「橫盤震盪區間」</b>，此時 RSI/KD 的信度優於均線。" if is_range else "目前市場屬於 <b>「趨勢表態行情」</b>，此時應以均線與法人籌碼為核心信度。"
    rpt_html += f"<li><span class='logic-tag'>環境判定</span> {env_text}</li>"

    # 矛盾 A: 噴發 vs 鈍化
    if is_trend_up and latest['RSI'] > 75:
        rpt_html += "<li><span class='logic-tag'>矛盾解析：高檔鈍化</span> 均線趨勢與 RSI 出現衝突。專家認為：在趨勢行情中超買指標常會失真，此處應判定為強者恆強，不應視為反轉訊號。</li>"
    
    # 矛盾 B: 騙多陷阱 (Bull Trap)
    is_bull_trap = is_trend_up and has_chip and inst_net < -1000
    if is_bull_trap:
        rpt_html += "<li><span class='logic-tag'>反騙線診斷：多頭陷阱</span> 股價雖漲但法人大賣超，成交量若未同步跟上則屬於『散戶追價』之虛漲，信度極低，具備高度騙線風險。</li>"

    # 矛盾 C: 洗盤騙空 (Bear Trap)
    is_bear_trap = not is_trend_up and (is_shadow or (has_chip and inst_net > 500)) and latest['Volume'] < latest['Vol_Avg']
    if is_bear_trap:
        rpt_html += "<li><span class='logic-tag'>反騙線診斷：空頭洗盤</span> 均線雖空但出現法人偷接與下影線，且下跌量縮。專家判定這極大機率是主力在洗除散戶浮額，不宜在此放空，隨時醞釀反彈。</li>"

    # 建議功能 1：趨勢連貫性 (3日分析)
    if has_chip:
        inst_3d = latest['Inst_3D_Sum']
        if inst_3d > 1500 and latest['Price_3D_Change'] > 0:
            rpt_html += f"<li><span class='logic-tag'>趨勢連貫性</span> 近3日法人持續買超累計 {int(inst_3d)}張，股價同步創高，這代表資金具有高度連續性，趨勢信度極強。</li>"

    if bull_score > bear_score + 5:
        rpt_html += "<li><b>信度結論：</b>多方維度呈現壓倒性優勢，且無顯著矛盾，操作信度極高。</li>"
    elif bear_score > bull_score + 5:
        rpt_html += "<li><b>信度結論：</b>空方壓力層層堆疊，籌碼與技術面同步走壞，應嚴格執行止損避險。</li>"
    else:
        rpt_html += "<li><b>信度結論：</b>目前多空指標交織，信度處於中等水準，操作應縮小部位，等待環境明朗。</li>"

    rpt_html += "</ul>"

    # --- 操作策略與強度分級 (Pillar 4) ---
    stg_html = "<div class='section-header'>🎯 2. 具體操作策略與動作分級</div>"
    
    # 計算決策
    diff = bull_score - bear_score
    
    # 持股者建議 (消極與積極)
    stg_html += "<b>【針對持有部位者】</b><ul>"
    if diff >= 6:
        stg_html += "<li><b>建議動作：</b><span class='action-passive'>積極續抱 / 分批加碼</span></li><li><b>理由：</b>主指標 (W3) 全部翻多，資金與趨勢一致，尚無賣出理由。</li>"
    elif diff <= -6:
        stg_html += "<li><b>建議動作：</b><span class='action-active'>積極清倉出清</span></li><li><b>理由：</b>趨勢結構性破壞。這不是修正而是轉向，持有風險極大。</li>"
    else:
        stg_html += "<li><b>建議動作：</b>消極減碼，套現觀望</li><li><b>理由：</b>多空爭奪中，不確定的風險溢價過高。</li>"
    stg_html += "</ul>"

    # 空手者建議 (積極度區分：賣出 vs 融券)
    stg_html += "<b>【針對新進場/放空者】</b><ul>"
    if diff >= 8:
        stg_html += f"<li><b>建議動作：</b>積極進場建立多單</li><li><b>介入位階：</b>建議於 5MA ({latest['SMA_5']:.2f} 元) 附近佈局。</li>"
    elif diff <= -8:
        stg_html += f"<li><b>建議動作：</b><span class='action-active'>積極融券放空 ★★★</span></li><li><b>理由：</b>這屬於崩潰型訊號，具備獲利空間。原因為支撐全破且法人連續倒貨，預期將發生多殺多，適合主動攻擊。</li>"
    elif diff < 0:
        stg_html += "<li><b>建議動作：</b>消極避險（不宜放空）</li><li><b>理由：</b>雖看空但尚未見到恐慌性拋售量，放空肉不多且易遭軋空。</li>"
    else:
        stg_html += "<li><b>建議動作：</b>空手耐心等待</li>"
    stg_html += "</ul>"

    return pd.DataFrame(table_data, columns=["分析維度", "訊號", "專家狀態解析"]), bull_score, bear_score, rpt_html, stg_html

# --- 繪圖功能 ---
def plot_advanced_chart(df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_width=[0.3, 0.7])
    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="K線"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_5'], line=dict(color='blue', width=1), name="5MA"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], line=dict(color='orange', width=1.5), name="20MA"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Up'], line=dict(color='rgba(255,0,0,0.2)', dash='dot'), name="布林上軌"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Low'], line=dict(color='rgba(0,255,0,0.2)', dash='dot'), name="布林下軌"), row=1, col=1)
    colors = ['#EF5350' if c >= o else '#26A69A' for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], marker_color=colors, name="成交量"), row=2, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False, height=550, margin=dict(t=30, l=10, r=10, b=10))
    return fig

# --- 主程式介面 ---
s_code = st.text_input("🔍 請輸入台股代號", "2330").strip()

if s_code:
    with st.spinner("AI 專家正在執行全維度深度診斷..."):
        df_raw, sym, s_name, has_c = fetch_complete_data(s_code)
        if df_raw is not None:
            db2 = generate_db2(df_raw)
            db3_df, b_score, r_score, rpt_h, stg_h = generate_expert_brain(db2, has_c)
            
            # 格式化顯示
            curr_p = db2.iloc[-1]['Close']; diff_p = curr_p - db2.iloc[-2]['Close']
            def fmt(v): return f"{v:.2f}".rstrip('0').rstrip('.') if v % 1 != 0 else f"{int(v)}"
            
            st.subheader(f"📊 分析對象：{s_code} - {s_name}")
            c1, c2, c3 = st.columns(3)
            c1.metric("當前股價", f"{fmt(curr_p)} TWD", f"{diff_p:.2f}")
            c2.metric("多方加權分", f"{b_score} 分")
            c3.metric("空方加權分", f"{r_score} 分")
            
            st.plotly_chart(plot_advanced_chart(db2.tail(100)), use_container_width=True)
            
            st.markdown("### 🧠 專家系統診斷與反騙線對策 (DB3)")
            cl, cr = st.columns([4, 6])
            with cl:
                st.dataframe(db3_df, hide_index=True, use_container_width=True)
            with cr:
                st.markdown(f'<div class="report-card">{rpt_h}</div>', unsafe_allow_html=True)
                st.write("")
                st.markdown(f'<div class="strategy-card">{stg_h}</div>', unsafe_allow_html=True)
        else:
            st.error("查無資料，請確認代號是否正確。")
