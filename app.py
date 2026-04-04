import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from FinMind.data import DataLoader
from datetime import datetime, timedelta

# --- 網頁配置 ---
st.set_page_config(page_title="AI 專家偵探決策系統", layout="wide")

st.markdown("""
    <style>
    .report-card { background-color: #ffffff; padding: 25px; border-radius: 15px; border-left: 10px solid #1f77b4; box-shadow: 0 4px 15px rgba(0,0,0,0.1); line-height: 1.8; }
    .strategy-card { background-color: #fdfdfd; padding: 25px; border-radius: 15px; border-left: 10px solid #d62728; border: 1px solid #eee; line-height: 1.8; }
    .logic-tag { background-color: #e3f2fd; color: #0d47a1; padding: 3px 10px; border-radius: 6px; font-size: 0.9rem; font-weight: bold; margin-right: 10px; border: 1px solid #bbdefb; }
    .status-box { background-color: #f1f8e9; padding: 10px; border-radius: 8px; margin-bottom: 15px; border: 1px solid #c5e1a5; }
    .action-active { color: #d32f2f; font-weight: bold; font-size: 1.15rem; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("🕵️‍♂️ AI 專家偵探：9 大指標深度辯論系統")

# --- 資料抓取與預處理 ---
@st.cache_data(ttl=86400)
def get_stock_name(code):
    try:
        dl = DataLoader()
        df_info = dl.taiwan_stock_info()
        item = df_info[df_info['stock_id'] == code]
        if not item.empty: return item.iloc[0]['stock_name']
    except Exception:
        pass
    return None

@st.cache_data(ttl=3600)
def fetch_complete_data(code):
    df_p, actual_sym, s_name = None, None, f"股票 {code}"
    c_name = get_stock_name(code)
    
    # 嘗試抓取上市或上櫃股價
    for suffix in [".TW", ".TWO"]:
        try:
            temp_sym = f"{code}{suffix}"
            ticker = yf.Ticker(temp_sym)
            df = ticker.history(period="1y")
            if not df.empty:
                s_name = c_name if c_name else ticker.info.get('shortName', temp_sym)
                df.reset_index(inplace=True)
                if isinstance(df.columns, pd.MultiIndex): 
                    df.columns = df.columns.get_level_values(0)
                # 統一日期格式
                df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
                df_p, actual_sym = df, temp_sym
                break
        except Exception: 
            continue
            
    if df_p is None: 
        return None, None, None, False
    
    # 嘗試抓取籌碼資料 (FinMind)
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
            
        if not df_chip.empty: 
            df_p = pd.merge(df_p, df_chip, on='Date', how='left')
            
        # 修正 Pandas 棄用語法：使用 ffill() 替代 fillna(method='ffill')
        df_p.ffill(inplace=True)
        df_p.fillna(0, inplace=True)
        
        return df_p, actual_sym, s_name, True
    except Exception: 
        return df_p, actual_sym, s_name, False

def generate_db2(df):
    d = df.copy()
    d['SMA_5'] = d['Close'].rolling(5).mean()
    d['SMA_20'] = d['Close'].rolling(20).mean()
    
    # 修正 RSI 邏輯：使用標準 Wilder's 平滑法 (EMA, alpha=1/14)
    delta = d['Close'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    d['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = d['Close'].ewm(span=12, adjust=False).mean()
    exp2 = d['Close'].ewm(span=26, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    d['MACD_H'] = macd_line - signal_line
    
    # 布林通道
    d['BB_Mid'] = d['Close'].rolling(20).mean()
    std = d['Close'].rolling(20).std()
    d['BB_Up'] = d['BB_Mid'] + 2*std
    d['BB_Low'] = d['BB_Mid'] - 2*std
    
    # KD 指標
    l9, h9 = d['Low'].rolling(9).min(), d['High'].rolling(9).max()
    rsv = (d['Close'] - l9) / (h9 - l9) * 100
    d['K'] = rsv.ewm(com=2, adjust=False).mean()
    d['D'] = d['K'].ewm(com=2, adjust=False).mean()
    
    d['Vol_Avg'] = d['Volume'].rolling(5).mean()
    d['Margin_Diff'] = d['Margin_Bal'].diff() if 'Margin_Bal' in d.columns else 0
    return d.dropna()

# --- 核心：偵探診斷與深度辯論引擎 (DB3) ---
def generate_expert_debate_db3(db2, has_chip):
    latest, prev = db2.iloc[-1], db2.iloc[-2]
    bull_score, bear_score = 0, 0
    table_data = []

    # 1. 均線 (W3)
    is_sma_up = latest['SMA_5'] > latest['SMA_20']
    table_data.append(["1. 均線趨勢", "🟢 看多" if is_sma_up else "🔴 看空", "多頭排列 (W3)" if is_sma_up else "空頭排列 (W3)"])
    if is_sma_up: bull_score += 3
    else: bear_score += 3

    # 2. RSI (W1)
    rsi_sig = "🔴 看空" if latest['RSI'] > 75 else "🟢 看多" if latest['RSI'] < 25 else "⚪ 中立"
    table_data.append(["2. 動能 RSI", rsi_sig, f"目前 {latest['RSI']:.1f}"])
    if "看多" in rsi_sig: bull_score += 1
    elif "看空" in rsi_sig: bear_score += 1

    # 3. MACD (W1)
    macd_up = latest['MACD_H'] > 0
    table_data.append(["3. 波段 MACD", "🟢 看多" if macd_up else "🔴 看空", "紅柱向上" if macd_up else "綠柱向下"])
    if macd_up: bull_score += 1
    else: bear_score += 1

    # 4. 布林 (W1)
    bb_sig = "🟢 看多" if latest['Close'] > latest['BB_Up'] else "🔴 看空" if latest['Close'] < latest['BB_Low'] else "⚪ 中立"
    table_data.append(["4. 布林通道", bb_sig, "通道運行"])
    if "看多" in bb_sig: bull_score += 1
    elif "看空" in bb_sig: bear_score += 1

    # 5. KD (W1)
    kd_cross = latest['K'] > latest['D']
    table_data.append(["5. KD 指標", "🟢 看多" if kd_cross else "🔴 看空", "黃金交叉" if kd_cross else "死亡交叉"])
    if kd_cross: bull_score += 1
    else: bear_score += 1

    # 6. K線 (W2)
    body = abs(latest['Close'] - latest['Open'])
    lower_s = min(latest['Close'], latest['Open']) - latest['Low']
    is_shadow = lower_s > body * 1.5 and body > 0
    table_data.append(["6. K線型態", "🟢 看多" if is_shadow else "⚪ 中立", "長下影線" if is_shadow else "一般型態"])
    if is_shadow: bull_score += 2

    # 7. 缺口 (W2)
    is_gap = latest['Low'] > prev['High']
    table_data.append(["7. 缺口理論", "🟢 看多" if is_gap else "⚪ 中立", "向上跳空" if is_gap else "無缺口"])
    if is_gap: bull_score += 2

    # 8. 法人 (W3)
    inst_net = 0
    if has_chip:
        inst_net = latest.get('Foreign_Buy', 0) + latest.get('Trust_Buy', 0)
        if inst_net > 500: 
            bull_score += 3
            table_data.append(["8. 法人籌碼", "🟢 看多", f"買超 {int(inst_net)}張"])
        elif inst_net < -500: 
            bear_score += 3
            table_data.append(["8. 法人籌碼", "🔴 看空", f"賣超 {int(inst_net)}張"])
        else: 
            table_data.append(["8. 法人籌碼", "⚪ 中立", "觀望"])
    else: 
        table_data.append(["8. 法人籌碼", "⚪ 未知", "無資料"])

    # 9. 散戶 (W2)
    if has_chip:
        if latest.get('Margin_Diff', 0) > 500 and latest['Close'] < prev['Close']:
            bear_score += 2
            table_data.append(["9. 散戶籌碼", "🔴 看空", "資增價跌"])
        else: 
            table_data.append(["9. 散戶籌碼", "⚪ 中立", "穩定"])
    else: 
        table_data.append(["9. 散戶籌碼", "⚪ 未知", "無資料"])

    # --- 🕵️‍♂️ 深度診斷報告 (全自動辯論) ---
    rpt_html = "<h4>🔍 1. 專家偵探辯論報告</h4>"
    
    # 總結狀態
    mood = "多方佔優" if bull_score > bear_score else "空方佔優" if bear_score > bull_score else "多空拉鋸"
    rpt_html += f"<div class='status-box'><b>當前局勢解析：</b>目前總加權分為 <span style='color:blue'>多方 {bull_score}</span> vs <span style='color:red'>空方 {bear_score}</span>。整體盤勢屬於 <b>{mood}</b>。</div>"
    
    rpt_html += "<ul>"
    # 辯論 1: 權重指標的統治力
    rpt_html += f"<li><b>核心信度：</b>目前的趨勢 (均線) 與 資金 (法人) 是判斷的核心。今日這兩者表現為 {'一致' if (is_sma_up and inst_net > -500) or (not is_sma_up and inst_net < 500) else '衝突'}。這決定了目前操作的容錯率。</li>"
    
    # 辯論 2: 處理「唯一」或「少數」的反面指標 (實事求是)
    found_contrary = False
    if bull_score > bear_score and bear_score > 0:
        contrary_names = [r[0] for r in table_data if "🔴 看空" in r[1]]
        rpt_html += f"<li><span class='logic-tag'>異常掃描</span> 雖然多方佔優，但 <b>{', '.join(contrary_names)}</b> 卻亮起紅燈。專家認為這可能是『噴發中的修正』或是『籌碼開始鬆動的早期預警』，不宜忽略。</li>"
        found_contrary = True
    elif bear_score > bull_score and bull_score > 0:
        contrary_names = [r[0] for r in table_data if "🟢 看多" in r[1]]
        rpt_html += f"<li><span class='logic-tag'>異常掃描</span> 雖然趨勢偏空，但 <b>{', '.join(contrary_names)}</b> 出現買進訊號。專家警告：這極大機率是『洗盤騙線』或『弱勢反彈』，在均線未轉正前，不宜輕易搶多。</li>"
        found_contrary = True

    # 辯論 3: 針對「騙線」或「洗盤」邏輯
    if not is_sma_up and is_shadow:
        rpt_html += "<li><span class='logic-tag'>洗盤辨識</span> 均線雖空，但K線出現長下影。專家偵探判定：此處有支撐力道介入，空方動能已被部分抵銷，此時不宜積極追空。</li>"
    if is_sma_up and has_chip and inst_net < -1000:
        rpt_html += "<li><span class='logic-tag'>出貨辨識</span> 股價漲但法人大賣。專家偵探判定：此為多頭陷阱，信度極低，建議趁漲減碼。</li>"
    
    if not found_contrary:
        rpt_html += "<li><b>信度評價：</b>目前 9 個指標高度同步，無任何背離發生，判斷信度高達 90% 以上。</li>"
    rpt_html += "</ul>"

    # --- 🎯 2. 具體策略與強度分級 ---
    stg_html = "<h4>🎯 2. 操作對策與積極度分級</h4>"
    diff = bull_score - bear_score
    stg_html += "<ul>"
    
    if diff >= 6:
        stg_html += f"<li><b class='action-active' style='color:green;'>【積極做多】</b> 分數絕對領先。建議於 5MA ({latest['SMA_5']:.2f}) 附近進場，止損設在 20MA。</li>"
    elif diff <= -6:
        stg_html += f"<li><b class='action-active'>【積極放空 ★★★】</b> 趨勢與籌碼同步崩潰。建議以融券建立部位，捕捉主跌段動能。</li>"
    elif diff > 0:
        stg_html += "<li><b>【消極持股】</b> 指標雖多但力道不足。建議現券持有，不加碼槓桿，隨時準備在跌破 5MA 時獲利了結。</li>"
    elif diff < 0:
        stg_html += "<li><b>【消極避險】</b> 指標偏空但尚有支撐。建議「賣出持股」觀望，但不建議反手放空。</li>"
    else:
        stg_html += "<li><b>【完全觀望】</b> 多空平手，市場無方向，應保持 100% 現金部位。</li>"
    stg_html += "</ul>"

    return pd.DataFrame(table_data, columns=["分析維度", "訊號", "解析"]), bull_score, bear_score, rpt_html, stg_html

# --- UI 呈現 ---
st.markdown("輸入台股代號 (如：2330, 0050)，系統將自動抓取歷史股價與近期籌碼數據，進行全方位分析。")
s_code = st.text_input("📈 請輸入台股代號", "2330").strip()

if s_code:
    with st.spinner("🕵️‍♂️ AI 偵探正在搜集資料與邏輯辯論中，請稍候..."):
        df_raw, sym, s_name, has_c = fetch_complete_data(s_code)
        
        if df_raw is not None and not df_raw.empty:
            db2 = generate_db2(df_raw)
            if not db2.empty:
                db3_df, b_s, r_s, rpt_h, stg_h = generate_expert_debate_db3(db2, has_c)
                
                curr_p = db2.iloc[-1]['Close']
                diff_p = curr_p - db2.iloc[-2]['Close']
                
                st.subheader(f"📊 分析標的：{s_code} - {s_name}")
                c1, c2, c3 = st.columns(3)
                c1.metric("最新股價", f"{curr_p:.2f} TWD", f"{diff_p:.2f}")
                c2.metric("多方加權分", f"{b_s} 分")
                c3.metric("空方加權分", f"{r_s} 分")
                
                # --- Plotly K線圖設定 (台股專用紅綠色) ---
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_width=[0.3, 0.7])
                
                # 台股習慣：紅漲綠跌
                fig.add_trace(go.Candlestick(
                    x=db2['Date'], open=db2['Open'], high=db2['High'], low=db2['Low'], close=db2['Close'], 
                    name="K線",
                    increasing_line_color='#EF5350',  # 漲(紅)
                    decreasing_line_color='#26A69A'   # 跌(綠)
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(x=db2['Date'], y=db2['SMA_5'], line=dict(color='blue', width=1), name="5MA"), row=1, col=1)
                fig.add_trace(go.Scatter(x=db2['Date'], y=db2['SMA_20'], line=dict(color='orange', width=1.5), name="20MA"), row=1, col=1)
                
                # 配合 K線的成交量顏色
                colors = ['#EF5350' if c >= o else '#26A69A' for c, o in zip(db2['Close'], db2['Open'])]
                fig.add_trace(go.Bar(x=db2['Date'], y=db2['Volume'], marker_color=colors, name="成交量"), row=2, col=1)
                
                fig.update_layout(xaxis_rangeslider_visible=False, height=500, margin=dict(t=30, l=10, r=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
                
                # --- 報告區塊 ---
                st.markdown("### 🧠 專家系統診斷與全維度辯論 (DB3)")
                cl, cr = st.columns([4, 6])
                with cl:
                    st.dataframe(db3_df, hide_index=True, use_container_width=True)
                with cr:
                    st.markdown(f'<div class="report-card">{rpt_h}</div>', unsafe_allow_html=True)
                    st.write("")
                    st.markdown(f'<div class="strategy-card">{stg_h}</div>', unsafe_allow_html=True)
            else:
                st.error("計算技術指標時資料不足，請確認該股票上市時間是否夠長。")
        else:
            st.error(f"❌ 找不到股票代號 {s_code} 的資料，請確認代號是否正確。")
