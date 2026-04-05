import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from FinMind.data import DataLoader
from datetime import datetime, timedelta
import google.generativeai as genai  # 新增：匯入 Google AI 套件

# --- 網頁配置 ---
st.set_page_config(page_title="AI 專家偵探決策系統", layout="wide")

st.markdown("""
    <style>
    .report-card { background-color: #ffffff; padding: 30px; border-radius: 15px; border-left: 10px solid #1f77b4; box-shadow: 0 4px 15px rgba(0,0,0,0.1); line-height: 1.8; font-size: 1.05rem;}
    .status-box { background-color: #f1f8e9; padding: 15px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #c5e1a5; font-size: 1.05rem;}
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- 側邊欄：API Key 設定 ---
st.sidebar.title("🔑 AI 大腦設定")
st.sidebar.markdown("請輸入您的 Google Gemini API Key 以啟動專家系統。")
api_key = st.sidebar.text_input("Gemini API Key", type="password")
if not api_key:
    st.sidebar.warning("⚠️ 請先輸入 API Key 才可產生診斷報告！[點此免費申請](https://makersuite.google.com/app/apikey)")

st.title("🕵️‍♂️ AI 專家偵探：9 大指標全方位辯論系統")

# --- 資料抓取與預處理 (維持原樣) ---
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
                df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
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
        df_p.ffill(inplace=True); df_p.fillna(0, inplace=True)
        return df_p, actual_sym, s_name, True
    except: return df_p, actual_sym, s_name, False

def generate_db2(df):
    d = df.copy()
    d['SMA_5'] = d['Close'].rolling(5).mean()
    d['SMA_20'] = d['Close'].rolling(20).mean()
    delta = d['Close'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    d['RSI'] = 100 - (100 / (1 + rs))
    exp1 = d['Close'].ewm(span=12, adjust=False).mean()
    exp2 = d['Close'].ewm(span=26, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    d['MACD_H'] = macd_line - signal_line
    d['BB_Mid'] = d['Close'].rolling(20).mean(); std = d['Close'].rolling(20).std()
    d['BB_Up'] = d['BB_Mid'] + 2*std; d['BB_Low'] = d['BB_Mid'] - 2*std
    l9, h9 = d['Low'].rolling(9).min(), d['High'].rolling(9).max()
    rsv = (d['Close'] - l9) / (h9 - l9) * 100
    d['K'] = rsv.ewm(com=2, adjust=False).mean(); d['D'] = d['K'].ewm(com=2, adjust=False).mean()
    d['Vol_Avg'] = d['Volume'].rolling(5).mean()
    d['Margin_Diff'] = d['Margin_Bal'].diff() if 'Margin_Bal' in d.columns else 0
    return d.dropna()

# --- 新增：呼叫 Google Gemini API 產生報告 ---
def call_ai_expert(indicator_data, price_data, key):
    genai.configure(api_key=key)
    # 使用最新的 gemini-1.5-flash 模型 (速度快、免費額度高)
    model = genai.GenerativeModel('gemini-3.0-flash-preview')
    
    # 這是我們剛剛討論好的終極 Prompt！
    prompt = f"""
你是一位擁有 20 年實戰經驗的台股重量級操盤手，同時也是一位極具耐心的財經導師。你的風格是客觀、數據導向，且擅長將複雜的市場現象轉化為清晰易懂的邏輯。

現在，請你針對以下這檔股票的 9 項技術與籌碼指標，進行「教科書等級」的深度交叉辯論與實戰指導。請盡量【使用列點式呈現】，讓文章具備高度的易讀性。

### 📊 傳入的指標數據 (共9項)：
{indicator_data}

### 💰 傳入的關鍵價位數據：
{price_data}

---
### 📝 你的輸出任務必須嚴格包含以下四個段落：

#### 第一部分：盤勢全覽與指標教學解析
- 請以「導師」的口吻，挑選出目前最關鍵的 2~3 個重點指標，向使用者解釋「為什麼這個訊號在此刻很重要？」以及它背後的意義。

#### 第二部分：共識與矛盾的深度辯論 (核心價值)
- **【共識強化】：** 盤點哪些指標方向一致（如互相支持）。深入說明這些力量結合在一起時，反映了什麼樣的市場心理或主力意圖？
- **【矛盾解剖】：** 盤點哪些指標產生了衝突。請發揮專業，解釋為什麼會出現這種矛盾？這是散戶被誘多的【騙線出貨】，還是上漲過程的【技術性洗盤】？請給出你的研判。

#### 第三部分：後勢推演預判
- 基於上述辯論，推演接下來 3~5 天內該股票最有可能發展的「高機率劇本」。
- 提醒潛在的風險或需要關注的變數。

#### 第四部分：實戰操作指引與價位規劃
- 將前面的分析觀點進行整合，給出明確、有紀律的交易建議。
- **必須將傳入的【關鍵價位數據】(如 5MA、20MA、今高等) 帶入建議中，作為具體的防守或進攻點。**
- 請嚴格分為以下兩種情境，以列點方式撰寫：
  1. 💼 **【已經持有部位者】：** 該續抱、加碼、還是減碼停損？防守價位設在哪裡？為什麼？
  2. 👁️ **【尚未持有部位者】：** 該空手觀望、尋找買點、還是反手放空？進場與停損價位設在哪裡？為什麼？

請以繁體中文撰寫，排版層次分明、適當使用 Emoji，語氣必須充滿說服力且邏輯嚴密。
"""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ AI 診斷產生錯誤：{str(e)}。請確認您的 API Key 是否正確或網路是否連線。"

# --- 核心引擎：資料準備與介面呈現 ---
def generate_expert_debate_db3(db2, has_chip):
    latest, prev = db2.iloc[-1], db2.iloc[-2]
    table_data = []
    indicator_text = ""

    # 1. 均線
    is_sma_up = latest['SMA_5'] > latest['SMA_20']
    table_data.append(["1. 均線 (趨勢)", "🟢 看多" if is_sma_up else "🔴 看空", "多頭排列" if is_sma_up else "空頭排列"])
    indicator_text += f"- 均線：{'看多 (多頭排列)' if is_sma_up else '看空 (空頭排列)'}\n"

    # 2. MACD
    macd_up = latest['MACD_H'] > 0
    table_data.append(["2. MACD (趨勢)", "🟢 看多" if macd_up else "🔴 看空", "紅柱向上" if macd_up else "綠柱向下"])
    indicator_text += f"- MACD：{'看多 (紅柱)' if macd_up else '看空 (綠柱)'}\n"

    # 3. 布林通道
    bb_sig = "🟢 看多" if latest['Close'] > latest['BB_Up'] else "🔴 看空" if latest['Close'] < latest['BB_Low'] else "⚪ 中立"
    table_data.append(["3. 布林通道", bb_sig, "突破上軌" if "多" in bb_sig else "跌破下軌" if "空" in bb_sig else "通道內運行"])
    indicator_text += f"- 布林通道：{bb_sig}\n"

    # 4. RSI
    rsi_sig = "🔴 看空 (超買)" if latest['RSI'] > 75 else "🟢 看多 (超賣)" if latest['RSI'] < 25 else "⚪ 中立"
    table_data.append(["4. RSI (動能)", rsi_sig, f"目前 {latest['RSI']:.1f}"])
    indicator_text += f"- RSI：{rsi_sig} (數值 {latest['RSI']:.1f})\n"

    # 5. KD
    kd_cross = latest['K'] > latest['D']
    table_data.append(["5. KD (動能)", "🟢 看多" if kd_cross else "🔴 看空", "黃金交叉" if kd_cross else "死亡交叉"])
    indicator_text += f"- KD：{'看多 (黃金交叉)' if kd_cross else '看空 (死亡交叉)'}\n"

    # 6. K線型態
    body = abs(latest['Close'] - latest['Open'])
    lower_s = min(latest['Close'], latest['Open']) - latest['Low']
    is_shadow = lower_s > body * 1.5 and body > 0
    table_data.append(["6. K線 (動能)", "🟢 看多" if is_shadow else "⚪ 中立", "長下影線" if is_shadow else "一般型態"])
    indicator_text += f"- K線型態：{'看多 (長下影線)' if is_shadow else '中立 (一般型態)'}\n"

    # 7. 缺口
    is_gap = latest['Low'] > prev['High']
    is_gap_down = latest['High'] < prev['Low']
    gap_sig = "🟢 看多 (向上缺口)" if is_gap else "🔴 看空 (向下缺口)" if is_gap_down else "⚪ 中立"
    table_data.append(["7. 缺口 (動能)", gap_sig[:4], gap_sig[5:] if len(gap_sig)>4 else "無缺口"])
    indicator_text += f"- 缺口理論：{gap_sig}\n"

    # 8. 法人
    if has_chip:
        inst_net = latest.get('Foreign_Buy', 0) + latest.get('Trust_Buy', 0)
        inst_sig = "🟢 看多" if inst_net > 500 else "🔴 看空" if inst_net < -500 else "⚪ 中立"
        table_data.append(["8. 法人 (籌碼)", inst_sig, f"淨買賣 {int(inst_net)} 張"])
        indicator_text += f"- 法人籌碼：{inst_sig} (淨買賣 {int(inst_net)} 張)\n"
    else:
        table_data.append(["8. 法人 (籌碼)", "⚪ 未知", "無資料"])
        indicator_text += "- 法人籌碼：未知 (無資料)\n"

    # 9. 散戶
    if has_chip:
        margin_bad = latest.get('Margin_Diff', 0) > 500 and latest['Close'] < prev['Close']
        table_data.append(["9. 散戶 (籌碼)", "🔴 看空" if margin_bad else "⚪ 中立", "資增價跌" if margin_bad else "穩定"])
        indicator_text += f"- 散戶籌碼：{'🔴 看空 (資增價跌籌碼亂)' if margin_bad else '⚪ 中立 (穩定)'}\n"
    else:
        table_data.append(["9. 散戶 (籌碼)", "⚪ 未知", "無資料"])
        indicator_text += "- 散戶籌碼：未知 (無資料)\n"

    # --- 整理價位資訊送給 AI ---
    price_text = f"""
- 最新收盤價：{latest['Close']:.2f}
- 今日最高價：{latest['High']:.2f}
- 今日最低價：{latest['Low']:.2f}
- 5日均線 (5MA)：{latest['SMA_5']:.2f}
- 20日均線 (20MA/月線)：{latest['SMA_20']:.2f}
"""
    return pd.DataFrame(table_data, columns=["分析維度", "訊號", "技術解析"]), indicator_text, price_text

# --- UI 呈現與執行流程 ---
st.markdown("輸入台股代號 (如：2330, 0050)，AI 將為您進行深度的教科書級診斷。")
s_code = st.text_input("📈 請輸入台股代號", "2330").strip()

if s_code:
    if not api_key:
        st.info("👈 請先在左側邊欄輸入您的 Gemini API Key，才可啟動 AI 分析。")
    else:
        with st.spinner("🕵️‍♂️ AI 大師正在交叉比對 9 大指標，撰寫深度報告中... (約需 5~8 秒)"):
            df_raw, sym, s_name, has_c = fetch_complete_data(s_code)
            
            if df_raw is not None and not df_raw.empty:
                db2 = generate_db2(df_raw)
                if not db2.empty:
                    # 1. 取得指標狀態與字串
                    db3_df, ind_txt, prc_txt = generate_expert_debate_db3(db2, has_c)
                    
                    # 2. 呼叫 AI API 產生報告
                    ai_report = call_ai_expert(ind_txt, prc_txt, api_key)
                    
                    # 3. 畫面上方：基本資訊與圖表
                    curr_p = db2.iloc[-1]['Close']
                    diff_p = curr_p - db2.iloc[-2]['Close']
                    
                    st.subheader(f"📊 分析標的：{s_code} - {s_name}")
                    c1, c2 = st.columns(2)
                    c1.metric("最新收盤價", f"{curr_p:.2f} TWD", f"{diff_p:.2f}")
                    c2.markdown(f"**5MA:** {db2.iloc[-1]['SMA_5']:.2f} | **20MA:** {db2.iloc[-1]['SMA_20']:.2f}")
                    
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_width=[0.3, 0.7])
                    fig.add_trace(go.Candlestick(x=db2['Date'], open=db2['Open'], high=db2['High'], low=db2['Low'], close=db2['Close'], name="K線", increasing_line_color='#EF5350', decreasing_line_color='#26A69A'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=db2['Date'], y=db2['SMA_5'], line=dict(color='blue', width=1), name="5MA"), row=1, col=1)
                    fig.add_trace(go.Scatter(x=db2['Date'], y=db2['SMA_20'], line=dict(color='orange', width=1.5), name="20MA"), row=1, col=1)
                    colors = ['#EF5350' if c >= o else '#26A69A' for c, o in zip(db2['Close'], db2['Open'])]
                    fig.add_trace(go.Bar(x=db2['Date'], y=db2['Volume'], marker_color=colors, name="成交量"), row=2, col=1)
                    fig.update_layout(xaxis_rangeslider_visible=False, height=450, margin=dict(t=30, l=10, r=10, b=10))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 4. 畫面下方：左側為指標表，右側為 AI 報告
                    cl, cr = st.columns([30, 70])
                    with cl:
                        st.markdown("#### 🧭 9 大指標當前狀態")
                        st.dataframe(db3_df, hide_index=True, use_container_width=True)
                    with cr:
                        st.markdown("#### 🧠 AI 專家深度辯論與策略指導")
                        st.markdown(f'<div class="report-card">{ai_report}</div>', unsafe_allow_html=True)
                else:
                    st.error("計算技術指標時資料不足，請確認該股票上市時間是否夠長。")
            else:
                st.error(f"❌ 找不到股票代號 {s_code} 的資料，請確認代號是否正確。")
