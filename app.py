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
    .report-card { background-color: #ffffff; padding: 25px; border-radius: 15px; border-left: 10px solid #1f77b4; box-shadow: 0 4px 15px rgba(0,0,0,0.1); line-height: 1.8; margin-bottom: 20px;}
    .strategy-card { background-color: #fdfdfd; padding: 25px; border-radius: 15px; border-left: 10px solid #d62728; border: 1px solid #eee; line-height: 1.8; }
    .logic-tag { background-color: #e3f2fd; color: #0d47a1; padding: 3px 10px; border-radius: 6px; font-size: 0.9rem; font-weight: bold; margin-right: 10px; border: 1px solid #bbdefb; }
    .status-box { background-color: #f1f8e9; padding: 15px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #c5e1a5; font-size: 1.05rem;}
    .action-active { color: #d32f2f; font-weight: bold; font-size: 1.1rem; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("🕵️‍♂️ AI 專家偵探：9 大指標與三方角力辯論系統")

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
                df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
                df_p, actual_sym = df, temp_sym
                break
        except Exception: 
            continue
            
    if df_p is None: 
        return None, None, None, False
    
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
            
        df_p.ffill(inplace=True)
        df_p.fillna(0, inplace=True)
        return df_p, actual_sym, s_name, True
    except Exception: 
        return df_p, actual_sym, s_name, False

def generate_db2(df):
    d = df.copy()
    d['SMA_5'] = d['Close'].rolling(5).mean()
    d['SMA_20'] = d['Close'].rolling(20).mean()
    
    # RSI (Wilder's Smoothing)
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
    
    # KD
    l9, h9 = d['Low'].rolling(9).min(), d['High'].rolling(9).max()
    rsv = (d['Close'] - l9) / (h9 - l9) * 100
    d['K'] = rsv.ewm(com=2, adjust=False).mean()
    d['D'] = d['K'].ewm(com=2, adjust=False).mean()
    
    d['Vol_Avg'] = d['Volume'].rolling(5).mean()
    d['Margin_Diff'] = d['Margin_Bal'].diff() if 'Margin_Bal' in d.columns else 0
    return d.dropna()

# --- 核心引擎：三方角力辯論與具體策略 (硬派邏輯寫死) ---
def generate_expert_debate_db3(db2, has_chip):
    latest, prev = db2.iloc[-1], db2.iloc[-2]
    
    # --- 變數準備 ---
    curr_price = latest['Close']
    sma5 = latest['SMA_5']
    sma20 = latest['SMA_20']
    today_high = latest['High']
    today_low = latest['Low']
    
    # --- 維度計分與狀態判斷 ---
    trend_bull, trend_bear = 0, 0
    mom_bull, mom_bear = 0, 0
    chip_bull, chip_bear = 0, 0
    table_data = []

    # 【維度 1：趨勢面】
    # 1. 均線
    if latest['SMA_5'] > latest['SMA_20']: trend_bull += 1; table_data.append(["1. 均線 (趨勢)", "🟢 看多", "多頭排列"])
    else: trend_bear += 1; table_data.append(["1. 均線 (趨勢)", "🔴 看空", "空頭排列"])
    # 2. MACD
    if latest['MACD_H'] > 0: trend_bull += 1; table_data.append(["2. MACD (趨勢)", "🟢 看多", "紅柱向上"])
    else: trend_bear += 1; table_data.append(["2. MACD (趨勢)", "🔴 看空", "綠柱向下"])
    # 3. 布林通道
    if latest['Close'] > latest['BB_Up']: trend_bull += 1; table_data.append(["3. 布林 (趨勢)", "🟢 看多", "突破上軌"])
    elif latest['Close'] < latest['BB_Low']: trend_bear += 1; table_data.append(["3. 布林 (趨勢)", "🔴 看空", "跌破下軌"])
    else: table_data.append(["3. 布林 (趨勢)", "⚪ 中立", "通道內運行"])

    # 【維度 2：動能面】
    # 4. RSI
    if latest['RSI'] < 30: mom_bull += 1; table_data.append(["4. RSI (動能)", "🟢 看多", f"超賣區 ({latest['RSI']:.1f})"])
    elif latest['RSI'] > 70: mom_bear += 1; table_data.append(["4. RSI (動能)", "🔴 看空", f"超買區 ({latest['RSI']:.1f})"])
    else: table_data.append(["4. RSI (動能)", "⚪ 中立", f"動能平穩 ({latest['RSI']:.1f})"])
    # 5. KD
    if latest['K'] > latest['D']: mom_bull += 1; table_data.append(["5. KD (動能)", "🟢 看多", "黃金交叉"])
    else: mom_bear += 1; table_data.append(["5. KD (動能)", "🔴 看空", "死亡交叉"])
    # 6. K線型態
    body = abs(latest['Close'] - latest['Open'])
    lower_s = min(latest['Close'], latest['Open']) - latest['Low']
    if lower_s > body * 1.5 and body > 0: mom_bull += 1; table_data.append(["6. K線 (動能)", "🟢 看多", "長下影線支撐"])
    else: table_data.append(["6. K線 (動能)", "⚪ 中立", "一般型態"])
    # 7. 缺口
    if latest['Low'] > prev['High']: mom_bull += 1; table_data.append(["7. 缺口 (動能)", "🟢 看多", "向上跳空"])
    elif latest['High'] < prev['Low']: mom_bear += 1; table_data.append(["7. 缺口 (動能)", "🔴 看空", "向下跳空"])
    else: table_data.append(["7. 缺口 (動能)", "⚪ 中立", "無明顯缺口"])

    # 【維度 3：籌碼面】
    if has_chip:
        inst_net = latest.get('Foreign_Buy', 0) + latest.get('Trust_Buy', 0)
        # 8. 法人
        if inst_net > 500: chip_bull += 1; table_data.append(["8. 法人 (籌碼)", "🟢 看多", f"買超 {int(inst_net)}張"])
        elif inst_net < -500: chip_bear += 1; table_data.append(["8. 法人 (籌碼)", "🔴 看空", f"賣超 {int(inst_net)}張"])
        else: table_data.append(["8. 法人 (籌碼)", "⚪ 中立", "觀望/無大單"])
        # 9. 散戶 (融資)
        if latest.get('Margin_Diff', 0) > 500 and latest['Close'] < prev['Close']: chip_bear += 1; table_data.append(["9. 散戶 (籌碼)", "🔴 看空", "資增價跌(籌碼亂)"])
        else: table_data.append(["9. 散戶 (籌碼)", "⚪ 中立", "表現穩定"])
    else:
        table_data.append(["8. 法人 (籌碼)", "⚪ 未知", "無資料"])
        table_data.append(["9. 散戶 (籌碼)", "⚪ 未知", "無資料"])

    # --- 判定各維度最終狀態 ---
    trend_state = "bull" if trend_bull > trend_bear else ("bear" if trend_bear > trend_bull else "neutral")
    mom_state = "bull" if mom_bull > mom_bear else ("bear" if mom_bear > mom_bull else "neutral")
    chip_state = "bull" if chip_bull > chip_bear else ("bear" if chip_bear > chip_bull else "neutral")

    # 無籌碼時的容錯處理：若無籌碼，預設與趨勢同向以簡化邏輯
    if not has_chip or chip_state == "neutral":
        chip_state = trend_state 

    # --- 🕵️‍♂️ 撰寫辯論報告與生成劇本 ---
    rpt_html = "<h4>🔍 1. 專家偵探分析與矛盾辯論</h4>"
    
    total_bull = trend_bull + mom_bull + chip_bull
    total_bear = trend_bear + mom_bear + chip_bear
    mood = "多方佔優" if total_bull > total_bear else "空方佔優" if total_bear > total_bull else "多空拉鋸"
    rpt_html += f"<div class='status-box'><b>當前陣營清點：</b> 9 大指標中，<span style='color:blue'>看多 {total_bull} 項</span> vs <span style='color:red'>看空 {total_bear} 項</span>。整體屬 <b>{mood}</b>。</div>"
    
    scenario_type = "chaos"
    
    rpt_html += "<ul>"
    # 邏輯判斷：四種劇本寫死
    if trend_state == "bull" and mom_state == "bull" and chip_state == "bull":
        scenario_type = "scenario_1_bull"
        rpt_html += "<li><span class='logic-tag'>劇本 1：三方共振 (大多頭)</span> 目前大格局趨勢向上，短線動能同步轉強，且法人籌碼大力支持。這三個維度形成完美共振，無任何矛盾，屬於最高勝率的單邊順勢行情。</li>"
    elif trend_state == "bear" and mom_state == "bear" and chip_state == "bear":
        scenario_type = "scenario_1_bear"
        rpt_html += "<li><span class='logic-tag'>劇本 1：三方共振 (大空頭)</span> 目前大格局趨勢崩潰，短線動能極度弱勢，且法人主力籌碼持續撤退。三方邏輯一致看空，跌勢極度明確。</li>"
    elif trend_state == "bull" and chip_state == "bull" and mom_state == "bear":
        scenario_type = "scenario_2_bull"
        rpt_html += "<li><span class='logic-tag'>劇本 2：洗盤甩轎 (技術回檔)</span> 雖然短線動能 (如KD/RSI) 出現看空或轉弱的訊號，但長線均線趨勢依然穩健，且法人真金白銀持續買進。專家判定這極高機率是『主力洗盤』，短線偏空是為甩轎，大方向未變。</li>"
    elif trend_state == "bear" and chip_state == "bear" and mom_state == "bull":
        scenario_type = "scenario_2_bear"
        rpt_html += "<li><span class='logic-tag'>劇本 2：弱勢反彈 (誘多騙線)</span> 雖然短線動能指標出現超賣或黃金交叉，但大格局趨勢依然向下，且籌碼顯示法人並未進場。專家警告：這是標準的『死貓反彈』或誘多騙線，切勿輕易搶多。</li>"
    elif trend_state == "bull" and mom_state == "bull" and chip_state == "bear":
        scenario_type = "scenario_3_bull"
        rpt_html += "<li><span class='logic-tag'>劇本 3：拉高出貨 (籌碼背離)</span> 表面上技術線型與動能皆呈現強勢看多，但內部籌碼卻顯示法人正在大舉撤退。專家強烈警告！這是『拉高出貨』的背離現象，股價強勢缺乏資金燃料，極易形成多頭陷阱。</li>"
    elif trend_state == "bear" and mom_state == "bear" and chip_state == "bull":
        scenario_type = "scenario_3_bear"
        rpt_html += "<li><span class='logic-tag'>劇本 3：破底翻準備 (籌碼潛伏)</span> 技術面與短線動能極度難看，但籌碼數據卻顯示法人主力正在逢低大舉買進。專家研判這可能是『空頭陷阱』，主力正利用散戶恐慌暗中吃貨，具備隨時破底翻的潛力。</li>"
    elif trend_state == "bear" and mom_state == "bull" and chip_state == "bull":
        scenario_type = "scenario_4_bull"
        rpt_html += "<li><span class='logic-tag'>劇本 4：築底轉折 (春江水暖)</span> 大格局均線趨勢雖然仍空，但短線動能已經轉強，且觀察到主力籌碼領先悄悄進駐。這是『築底轉向』的早期訊號，資金與動能已表態。</li>"
    elif trend_state == "bull" and mom_state == "bear" and chip_state == "bear":
        scenario_type = "scenario_4_bear"
        rpt_html += "<li><span class='logic-tag'>劇本 4：高檔轉弱 (初跌段)</span> 大格局趨勢目前仍呈多頭排列，但短線動能已經衰退，且主力籌碼已領先倒貨。這是高檔反轉的早期訊號，趨勢隨時可能跟著轉弱。</li>"
    else:
        scenario_type = "chaos"
        rpt_html += "<li><span class='logic-tag'>盤整劇本：多空膠著</span> 目前趨勢、動能、籌碼三方互有勝負，未形成明顯的主導力量。盤勢處於收斂或震盪換手期，無明確方向性。</li>"
    rpt_html += "</ul>"

    # --- 🎯 撰寫具體操作策略 (綁定持倉與價位) ---
    stg_html = "<h4>🎯 2. 操作建議與具體價位規劃</h4>"
    stg_html += f"<p>基於上述專家分析，針對當前價位 (<b>{curr_price:.2f}</b>)，給予以下具體指引：</p>"
    
    stg_html += "<ul>"
    if scenario_type == "scenario_1_bull":
        stg_html += f"<li>💼 <b>【已經持有部位者】：堅定抱牢。</b> 目前為明確多頭，無需恐慌。建議以 5日均線 ({sma5:.2f}) 為強勢防守線，跌破才考慮減碼；波段停損嚴格設於 20日均線 ({sma20:.2f})。</li>"
        stg_html += f"<li>👁️ <b>【尚未持有部位者】：積極作多。</b> 現價至 5MA ({sma5:.2f}) 之間為極佳打擊區。可順勢進場，並將風險控管(停損)設於 20MA ({sma20:.2f}) 下方。</li>"
    
    elif scenario_type == "scenario_1_bear":
        stg_html += f"<li>💼 <b>【已經持有部位者】：必須果斷離場。</b> 趨勢已崩潰，切勿抱持幻想凹單。若股價無法站回 5MA ({sma5:.2f})，應盡速停損/賣出保住現金。</li>"
        stg_html += f"<li>👁️ <b>【尚未持有部位者】：絕對觀望，積極者可偏空。</b> 絕不宜在此搶反彈。保守者保持空手；積極經驗者若見股價反彈至 5MA ({sma5:.2f}) 且上攻無力，可嘗試偏空操作，停損設於 20MA ({sma20:.2f})。</li>"
    
    elif scenario_type == "scenario_2_bull":
        stg_html += f"<li>💼 <b>【已經持有部位者】：續抱，無須恐慌。</b> 目前屬短線技術性洗盤，主力並未撤退。防守底線看 20MA ({sma20:.2f})，未跌破前持股續抱。</li>"
        stg_html += f"<li>👁️ <b>【尚未持有部位者】：逢低布局好時機。</b> 利用短線動能轉弱的錯殺機會，可在靠近 10MA 或 20MA ({sma20:.2f}) 附近勇敢承接，停損設於 20MA 之下 2%。</li>"
    
    elif scenario_type == "scenario_2_bear":
        stg_html += f"<li>💼 <b>【已經持有部位者】：趁反彈逃命。</b> 短線動能雖強，但大局與籌碼皆空。若觸及 20MA ({sma20:.2f}) 附近或遇壓，應果斷賣出減碼，勿當長線投資。</li>"
        stg_html += f"<li>👁️ <b>【尚未持有部位者】：嚴格觀望，不可追高。</b> 此為誘多陷阱。保守者空手；積極者可等股價反彈靠近 20MA ({sma20:.2f}) 不過時，佈局空單。</li>"
    
    elif scenario_type == "scenario_3_bull":
        stg_html += f"<li>💼 <b>【已經持有部位者】：逢高減碼，鎖定利潤。</b> 資金正在撤退，不建議盲目抱牢。若跌破短線 5MA ({sma5:.2f}) 建議立刻獲利了結；或考慮在今日高點 ({today_high:.2f}) 附近分批賣出 50% 降風險。</li>"
        stg_html += f"<li>👁️ <b>【尚未持有部位者】：放棄做多，嚴格觀望。</b> 表面強勢缺乏資金燃料，現價買進極易成為主力倒貨對象。不建議進場，積極者反可觀察跌破 5MA ({sma5:.2f}) 時的偏空機會。</li>"
    
    elif scenario_type == "scenario_3_bear":
        stg_html += f"<li>💼 <b>【已經持有部位者】：忍耐觀望，準備解套。</b> 雖然線型極醜，但主力籌碼已在低接。目前不建議殺低，若能突破並站穩 5MA ({sma5:.2f})，有機會迎來大反彈。</li>"
        stg_html += f"<li>👁️ <b>【尚未持有部位者】：準備左側抄底。</b> 技術面雖空，但籌碼暗示底部已近。可於今日低點 ({today_low:.2f}) 不破的條件下，小資金試單做多。</li>"
    
    elif scenario_type == "scenario_4_bull":
        stg_html += f"<li>💼 <b>【已經持有部位者】：停止殺低，準備迎接轉折。</b> 動能與籌碼已翻多，股價即將挑戰 20MA ({sma20:.2f}) 壓力，等待趨勢正式扭轉。</li>"
        stg_html += f"<li>👁️ <b>【尚未持有部位者】：小部位試單做多。</b> 趨勢雖未完全翻揚，但領先指標已表態。可於 5MA ({sma5:.2f}) 附近小部位佈局，跌破今日低點 ({today_low:.2f}) 則停損。</li>"
    
    elif scenario_type == "scenario_4_bear":
        stg_html += f"<li>💼 <b>【已經持有部位者】：提高警覺，隨時停利。</b> 動能與籌碼已轉空，大趨勢隨時可能跟著轉弱。若正式跌破 5MA ({sma5:.2f})，應即刻獲利出場。</li>"
        stg_html += f"<li>👁️ <b>【尚未持有部位者】：絕不追高，逢高偏空。</b> 主力已在撤退，現價絕不建議進場做多。積極者若見反彈至今日高點 ({today_high:.2f}) 附近無力，可嘗試偏空操作。</li>"
    
    else: # chaos
        stg_html += f"<li>💼 <b>【已經持有部位者】：縮減部位。</b> 盤勢無明確方向，建議將持股降至 50% 以下，並以 20MA ({sma20:.2f}) 作為最後防守線。</li>"
        stg_html += f"<li>👁️ <b>【尚未持有部位者】：保留現金觀望。</b> 無明確方向訊號，多做多錯。建議保留 100% 現金，等待趨勢突破 {today_high:.2f} 或跌破 {today_low:.2f} 後再作定奪。</li>"
    stg_html += "</ul>"

    return pd.DataFrame(table_data, columns=["分析維度", "訊號", "技術解析"]), total_bull, total_bear, rpt_html, stg_html

# --- UI 呈現 ---
st.markdown("輸入台股代號 (如：2330, 0050)，系統將自動抓取歷史股價與近期籌碼數據，進行專家全方位辯論。")
s_code = st.text_input("📈 請輸入台股代號", "2330").strip()

if s_code:
    with st.spinner("🕵️‍♂️ AI 偵探正在搜集籌碼資料與進行多維度邏輯辯論，請稍候..."):
        df_raw, sym, s_name, has_c = fetch_complete_data(s_code)
        
        if df_raw is not None and not df_raw.empty:
            db2 = generate_db2(df_raw)
            if not db2.empty:
                db3_df, b_s, r_s, rpt_h, stg_h = generate_expert_debate_db3(db2, has_c)
                
                curr_p = db2.iloc[-1]['Close']
                diff_p = curr_p - db2.iloc[-2]['Close']
                
                st.subheader(f"📊 分析標的：{s_code} - {s_name}")
                c1, c2, c3 = st.columns(3)
                c1.metric("最新收盤價", f"{curr_p:.2f} TWD", f"{diff_p:.2f}")
                c2.metric("🟢 多方力道總計", f"{b_s} 項指標支持")
                c3.metric("🔴 空方力道總計", f"{r_s} 項指標支持")
                
                # --- Plotly K線圖設定 (台股專用紅綠色) ---
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_width=[0.3, 0.7])
                
                fig.add_trace(go.Candlestick(
                    x=db2['Date'], open=db2['Open'], high=db2['High'], low=db2['Low'], close=db2['Close'], 
                    name="K線", increasing_line_color='#EF5350', decreasing_line_color='#26A69A'
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(x=db2['Date'], y=db2['SMA_5'], line=dict(color='blue', width=1), name="5MA"), row=1, col=1)
                fig.add_trace(go.Scatter(x=db2['Date'], y=db2['SMA_20'], line=dict(color='orange', width=1.5), name="20MA"), row=1, col=1)
                
                colors = ['#EF5350' if c >= o else '#26A69A' for c, o in zip(db2['Close'], db2['Open'])]
                fig.add_trace(go.Bar(x=db2['Date'], y=db2['Volume'], marker_color=colors, name="成交量"), row=2, col=1)
                
                fig.update_layout(xaxis_rangeslider_visible=False, height=500, margin=dict(t=30, l=10, r=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
                
                # --- 報告區塊 ---
                cl, cr = st.columns([35, 65])
                with cl:
                    st.markdown("#### 🧭 9 大指標清單")
                    st.dataframe(db3_df, hide_index=True, use_container_width=True)
                with cr:
                    st.markdown(f'<div class="report-card">{rpt_h}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="strategy-card">{stg_h}</div>', unsafe_allow_html=True)
            else:
                st.error("計算技術指標時資料不足，請確認該股票上市時間是否夠長。")
        else:
            st.error(f"❌ 找不到股票代號 {s_code} 的資料，請確認代號是否正確。")
