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
    .report-card { background-color: #ffffff; padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4; box-shadow: 0 2px 4px rgba(0,0,0,0.05); min-height: 250px; }
    .strategy-card { background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #ff4b4b; min-height: 150px;}
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 AI 專家級 9 大指標技術分析系統")

# --- 輔助函式：取得中文名稱 ---
@st.cache_data(ttl=86400)
def get_stock_chinese_name(code):
    try:
        dl = DataLoader()
        df_info = dl.taiwan_stock_info()
        item = df_info[df_info['stock_id'] == code]
        if not item.empty:
            return item.iloc[0]['stock_name']
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

    # 抓取籌碼
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
    except:
        return df_price, actual_sym, final_name, False

# --- 技術指標計算 (DB2) ---
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
    db2['BB_Mid'] = db2['Close'].rolling(20).mean()
    std = db2['Close'].rolling(20).std()
    db2['BB_Up'] = db2['BB_Mid'] + 2*std; db2['BB_Low'] = db2['BB_Mid'] - 2*std
    l9, h9 = db2['Low'].rolling(9).min(), db2['High'].rolling(9).max()
    rsv = (db2['Close'] - l9) / (h9 - l9) * 100
    db2['K'] = rsv.ewm(com=2).mean(); db2['D'] = db2['K'].ewm(com=2).mean()
    db2['Prev_Close'] = db2['Close'].shift(1); db2['Prev_High'] = db2['High'].shift(1); db2['Prev_Low'] = db2['Low'].shift(1)
    if 'Margin_Bal' in db2.columns: db2['Margin_Diff'] = db2['Margin_Bal'].diff()
    return db2.dropna()

# --- 強化版專家分析引擎 (DB3) ---
def generate_expert_db3(db2, has_chip):
    latest = db2.iloc[-1]
    prev = db2.iloc[-2]
    analysis = []
    bull_score, bear_score, neutral_score = 0, 0, 0
    
    # 1. 均線
    if latest['SMA_5'] > latest['SMA_20']:
        analysis.append(["均線理論", "🟢 看多", "多頭排列，趨勢向上"]); bull_score += 1
    else:
        analysis.append(["均線理論", "🔴 看空", "死叉向下，趨勢偏弱"]); bear_score += 1

    # 2. RSI
    if latest['RSI'] > 75:
        analysis.append(["動能 (RSI)", "🔴 看空", f"RSI {latest['RSI']:.1f} 超買區"]); bear_score += 1
    elif latest['RSI'] < 25:
        analysis.append(["動能 (RSI)", "🟢 看多", f"RSI {latest['RSI']:.1f} 超賣區"]); bull_score += 1
    else:
        analysis.append(["動能 (RSI)", "⚪ 中立", "震盪區間"]); neutral_score += 1

    # 3. MACD
    if latest['MACD_H'] > 0:
        analysis.append(["波段 (MACD)", "🟢 看多", "紅柱增長"]); bull_score += 1
    else:
        analysis.append(["波段 (MACD)", "🔴 看空", "綠柱區間"]); bear_score += 1

    # 4. 布林
    if latest['Close'] > latest['BB_Up']:
        analysis.append(["布林通道", "🟢 看多", "強勢上攻軌道"]); bull_score += 1
    elif latest['Close'] < latest['BB_Low']:
        analysis.append(["布林通道", "🔴 看空", "跌破下軌支撐"]); bear_score += 1
    else:
        analysis.append(["布林通道", "⚪ 中立", "通道內盤整"]); neutral_score += 1

    # 5. KD
    if latest['K'] > 80:
        analysis.append(["KD 指標", "🔴 看空", "高檔鈍化風險"]); bear_score += 1
    elif latest['K'] < 20:
        analysis.append(["KD 指標", "🟢 看多", "低檔超賣支撐"]); bull_score += 1
    else:
        analysis.append(["KD 指標", "⚪ 中立", "無明顯轉折"]); neutral_score += 1

    # 6. K線
    body = abs(latest['Close'] - latest['Open'])
    lower_s = min(latest['Close'], latest['Open']) - latest['Low']
    if lower_s > body * 1.5 and body > 0:
        analysis.append(["K線型態", "🟢 看多", "長下影線護盤"]); bull_score += 1
    else:
        analysis.append(["K線型態", "⚪ 中立", "常規震盪"]); neutral_score += 1

    # 7. 缺口
    if latest['Low'] > prev['High']:
        analysis.append(["缺口理論", "🟢 看多", "向上跳空強烈"]); bull_score += 1
    elif latest['High'] < prev['Low']:
        analysis.append(["缺口理論", "🔴 看空", "向下跳空衰竭"]); bear_score += 1
    else:
        analysis.append(["缺口理論", "⚪ 中立", "無缺口"]); neutral_score += 1

    # 8 & 9. 籌碼
    if has_chip:
        inst_net = latest['Foreign_Buy'] + latest['Trust_Buy']
        if inst_net > 500:
            analysis.append(["法人籌碼", "🟢 看多", f"大買 {int(inst_net)}張"]); bull_score += 1
        elif inst_net < -500:
            analysis.append(["法人籌碼", "🔴 看空", f"大賣 {int(inst_net)}張"]); bear_score += 1
        else:
            analysis.append(["法人籌碼", "⚪ 中立", "動作不明確"]); neutral_score += 1
            
        if latest['Margin_Diff'] > 500 and latest['Close'] < prev['Close']:
            analysis.append(["散戶籌碼", "🔴 看空", "資增價跌凌亂"]); bear_score += 1
        else:
            analysis.append(["散戶籌碼", "⚪ 中立", "無明顯背離"]); neutral_score += 1
    else:
        analysis.append(["法人籌碼", "⚪ 未知", "暫無資料"]); analysis.append(["散戶籌碼", "⚪ 未知", "暫無資料"])

    # --- 核心專家分析報告 (保證不留白) ---
    report = "#### 🔍 1. 核心矛盾與信度解析\n"
    summary = f"**盤勢總結**：目前市場呈現 **{'偏多' if bull_score > bear_score else '偏空' if bear_score > bull_score else '區間震盪'}** 態勢。"
    summary += f"在 9 大指標中，🟢 看多 {bull_score} 項，🔴 看空 {bear_score} 項。\n"
    report += f"> {summary}\n\n"

    # 指標衝突檢查
    conflicts = []
    if latest['SMA_5'] > latest['SMA_20'] and latest['RSI'] > 75:
        conflicts.append("- **強勢鈍化警告**：均線多頭但 RSI 已極度噴發。**專家建議**：此時不宜新進追價，應以 5MA 為防守持有現券。")
    if latest['Close'] > latest['SMA_5'] and latest['MACD_H'] < 0:
        conflicts.append("- **短線背離警告**：股價雖站上均線，但 MACD 動能尚未轉正。**專家建議**：這屬於「弱勢反彈」，信度以波段指標為準，小心假突破。")
    if has_chip and (latest['Close'] > prev['Close'] and inst_net < -1000):
        conflicts.append("- **籌碼背離**：股價上漲但法人大幅倒貨。**專家建議**：高度警戒主力出貨，此上漲信度極低。")
    
    if conflicts:
        report += "\n".join(conflicts)
    else:
        report += "- **趨勢評論**：目前各項指標方向大致同調，無顯著背離現象。建議順著目前的市場趨勢操作即可。"

    # --- 具體操作策略 (精準點位) ---
    strategy = "#### 🎯 2. 具體操作策略與動作\n"
    entry_p = latest['SMA_5'].round(2)
    target_p = latest['BB_Up'].round(2)
    stop_p = (latest['Close'] * 0.95).round(2) if bull_score > bear_score else (latest['SMA_20']).round(2)
    
    if bull_score >= 6:
        strategy += f"**【積極做多】**\n- **動作**：現價或回測 **{entry_p} 元** 可介入。\n- **止盈目標**：布林上軌 **{target_p} 元**。\n- **止損防守**：設定於 **{stop_p} 元**。"
    elif bear_score >= 6:
        strategy += f"**【嚴格觀望/空單】**\n- **動作**：壓力沉重，不宜摸底。建議等待股價重回 **{latest['SMA_20']:.2f}** 以上再表態。\n- **止損防守**：若有持股，破 **{latest['BB_Low']:.2f}** 應果斷撤退。"
    else:
        strategy += f"**【中立區間操作】**\n- **動作**：建議在 **{latest['BB_Low']:.2f}** 與 **{latest['BB_Up']:.2f}** 之間低買高賣。\n- **關鍵訊號**：等待 MACD 紅柱再次伸長為加碼訊號。"

    return pd.DataFrame(analysis, columns=["分析維度", "訊號", "專家描述"]), bull_score, bear_score, report, strategy

# --- 主程式 UI ---
s_code = st.text_input("📈 請輸入台股代號 (如 2330, 5490)", "2330").strip()

if s_code:
    with st.spinner("資料庫連線中..."):
        df_raw, actual_sym, s_name, has_c = fetch_all_data(s_code)
        if df_raw is not None:
            db2 = generate_db2(df_raw)
            db3_df, bulls, bears, rpt, stg = generate_expert_db3(db2, has_c)
            
            # 股價格式化 (1810.0 -> 1810)
            def fmt(v): return f"{v:.2f}".rstrip('0').rstrip('.') if v % 1 != 0 else f"{int(v)}"
            
            curr_p = db2.iloc[-1]['Close']
            diff = curr_p - db2.iloc[-2]['Close']
            
            st.subheader(f"📊 分析標的：{s_code} - {s_name}")
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("當前股價", f"{fmt(curr_p)} TWD", f"{diff:.2f}")
            col_m2.metric("看多指標數", f"{bulls} / 9")
            col_m3.metric("看空指標數", f"{bears} / 9")
            
            # K線與成交量
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
                st.markdown(f'<div class="report-card">{rpt}</div>', unsafe_allow_html=True)
                st.write("")
                st.markdown(f'<div class="strategy-card">{stg}</div>', unsafe_allow_html=True)
        else:
            st.error("查無資料，請確認代號。")
