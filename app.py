import streamlit as st
import pandas as pd
import gspread
import json
import google.generativeai as genai
from google.oauth2.service_account import Credentials
import yfinance as yf
import re

# --- 1. 視覺優化 (加大字體 & 戰情室配色) ---
st.set_page_config(page_title="台股 AI 偵探系統 v5.0", layout="wide")
st.markdown("""
    <style>
    .stMarkdown p, .stMarkdown li { font-size: 1.25rem !important; line-height: 1.7; }
    .stTable { font-size: 1.1rem !important; }
    .report-title { color: #1e3a8a; font-weight: bold; border-bottom: 2px solid #1e3a8a; padding-bottom: 5px; margin-bottom: 15px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 初始化狀態記憶 ---
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None
if 'current_id' not in st.session_state:
    st.session_state.current_id = ""

SHEET_ID = "1UH-fwxENhGUDmQjTQJq72g1DzsxML_CPIJGg39cWAgM"

# --- 3. 數據核心函式 ---
def get_data_from_sheets():
    try:
        creds_json_str = st.secrets["gcp_service_account_raw"]
        creds_info = json.loads(creds_json_str)
        scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        creds = Credentials.from_service_account_info(creds_info, scopes=scope)
        client = gspread.authorize(creds)
        sheet = client.open_by_key(SHEET_ID).sheet1 
        return pd.DataFrame(sheet.get_all_records())
    except: return pd.DataFrame()

def fetch_smart_yf(stock_id):
    suffixes = ["", ".TW", ".TWO"]
    for suf in suffixes:
        test_id = f"{stock_id}{suf}"
        try:
            df_yf = yf.download(test_id, period="2mo", progress=False)
            if not df_yf.empty and len(df_yf) > 20:
                df_yf['MA5'] = df_yf['Close'].rolling(window=5).mean()
                df_yf['MA20'] = df_yf['Close'].rolling(window=20).mean()
                delta = df_yf['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df_yf['RSI14'] = 100 - (100 / (1 + rs))
                df_yf = df_yf.reset_index().tail(5)
                return pd.DataFrame({
                    '日期': df_yf['Date'].dt.strftime('%Y-%m-%d'),
                    '股號': test_id, '股名': "即時數據",
                    '收盤價': df_yf['Close'].round(2), '成交量': df_yf['Volume'],
                    'MA5': df_yf['MA5'].round(2), 'MA20': df_yf['MA20'].round(2), 'RSI14': df_yf['RSI14'].round(2)
                })
        except: continue
    return pd.DataFrame()

def extract_block(text, block_num):
    pattern = rf"\[區塊{block_num}\](.*?)\[/區塊{block_num}\]"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""

def call_ai_detective(prompt):
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        model = genai.GenerativeModel('gemma-4-31b-it')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e: return f"AI 偵探錯誤: {str(e)}"

# --- 4. 主程式 ---
def main():
    st.title("🕵️‍♂️ 台股 AI 偵探戰情室")
    
    user_input = st.text_input("🔢 請輸入股號 (例如: 2330)", placeholder="輸入純數字即可").strip()
    if st.button("🔍 開始偵查") and user_input:
        with st.spinner("調閱檔案中..."):
            df_all = get_data_from_sheets()
            df_selected = pd.DataFrame()
            if not df_all.empty and '股號' in df_all.columns:
                df_all['股號'] = df_all['股號'].astype(str)
                df_selected = df_all[df_all['股號'].str.contains(user_input)].tail(5)
            if df_selected.empty:
                df_selected = fetch_smart_yf(user_input)
            
            if not df_selected.empty:
                st.session_state.stock_data = df_selected
                st.session_state.current_id = df_selected['股號'].iloc[-1]
            else: st.error("查無資料")

    if st.session_state.stock_data is not None:
        df_display = st.session_state.stock_data
        current_id = st.session_state.current_id
        st.subheader(f"📊 {current_id} 指標觀測站")
        st.table(df_display[['日期', '收盤價', '成交量', 'MA5', 'MA20', 'RSI14']])

        if st.button(f"🚀 啟動 {current_id} AI 深度診斷"):
            with st.spinner("偵探正在進行九項指標交叉比對..."):
                data_text = df_display.to_string(index=False)
                
                # --- 核心 Prompt：寫死九項指標與教學邏輯 ---
                prompt = f"""你現在是台股 AI 偵探。禁止英文、禁止廢話。
請嚴格遵守以下四個標籤格式輸出分析內容：

[區塊1]
【九項指標趨勢深度判讀】（請務必依序逐一條列分析）：
1. 5日均線走向
2. 20日均線走向
3. RSI14 強弱狀態
4. KDJ-K值 轉折
5. KDJ-D值 轉折
6. MACD 動能變化
7. 布林通道相對位置
8. 量價背離關係（分析價格漲跌與成交量增減之配合）
9. 近5日漲跌連續性與市場心理
[/區塊1]

[區塊2]
【指標矛盾整合與風險抓漏】（偵探教學模式）：
1. 請針對上述九項指標中，「互相支持」或「互相背離」的資料進行深入分析。
2. 以教學語氣解釋：如果某項指標看多但另一項看空，代表什麼市場意義？
3. 揭露目前數據中隱藏的誘多或誘空陷阱。
[/區塊2]

[區塊3]
【全方位操作戰略：雙重劇本】：
- 保守型劇本：(進場點、停損位、持股邏輯)
- 激進型劇本：(突破點、目標價、短線策略)
[/區塊3]

[區塊4]
【偵探總結與信心分數】：
總結：(一句話精闢評語)
信心分數：(0-100)
[/區塊4]

待分析數據來源：
{data_text}"""
                
                ans = call_ai_detective(prompt)
                
                b1 = extract_block(ans, 1)
                b2 = extract_block(ans, 2)
                b3 = extract_block(ans, 3)
                b4 = extract_block(ans, 4)

                st.markdown("---")
                st.markdown(f"### 🛡️ 偵探診斷報告：{current_id}")
                
                if b1: st.info(f"{b1}")
                if b2: st.warning(f"{b2}")
                if b3: st.success(f"{b3}")
                if b4: st.error(f"{b4}")

if __name__ == "__main__":
    main()
