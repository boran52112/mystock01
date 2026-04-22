import streamlit as st
import pandas as pd
import gspread
import json
import google.generativeai as genai
from google.oauth2.service_account import Credentials
import yfinance as yf
import re

# --- 1. 視覺優化 (手機大字體 & 專業診斷排版) ---
st.set_page_config(page_title="台股 AI 偵探系統 v5.0", layout="wide")
st.markdown("""
    <style>
    .stMarkdown p, .stMarkdown li { font-size: 1.25rem !important; line-height: 1.7; }
    .stTable { font-size: 1.1rem !important; }
    .report-card { 
        background-color: #ffffff; 
        padding: 20px; 
        border-radius: 12px; 
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 25px;
    }
    .indicator-label { font-weight: bold; color: #1e3a8a; }
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

# --- 4. 強化版：區塊內容提取邏輯 (取最後一個繁體中文區塊) ---
def extract_best_block(text, block_num):
    pattern = rf"\[區塊{block_num}\](.*?)\[/區塊{block_num}\]"
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches: return ""
    
    # 從後往前找，第一個包含中文的區塊即為正解
    for m in reversed(matches):
        if any('\u4e00' <= char <= '\u9fff' for char in m):
            return m.strip()
    return matches[-1].strip()

def call_ai_detective(prompt):
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        model = genai.GenerativeModel('gemma-4-31b-it')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e: return f"AI 偵探錯誤: {str(e)}"

# --- 5. 主程式 ---
def main():
    st.title("🕵️‍♂️ 台股 AI 偵探戰情室")
    
    user_input = st.text_input("🔢 請輸入股號", placeholder="輸入 2330 或 2356").strip()
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
            with st.spinner("偵探正在排版報告，過濾雜訊中..."):
                data_text = df_display.to_string(index=False)
                
                # 更嚴厲的 Prompt，要求 AI 絕對閉嘴，只准輸出標籤內容
                prompt = f"""你是台股 AI 偵探。輸出內容『嚴禁出現任何英文』。
禁止草擬、禁止自我檢查、禁止輸出 Input Data。
請直接填寫以下四個標籤，標籤外『不准有任何文字』：

[區塊1]
1. 5日均線走向：
2. 20日均線走向：
3. RSI14 強弱狀態：
4. KDJ-K值 轉折：
5. KDJ-D值 轉折：
6. MACD 動能變化：
7. 布林通道相對位置：
8. 量價背離關係：
9. 近5日漲跌連續性：
[/區塊1]

[區塊2]
1. 指標支持與背離分析：
2. 矛盾數據教學解釋：
3. 目前市場誘多/誘空陷阱解析：
[/區塊2]

[區塊3]
- 保守型劇本：
- 激進型劇本：
[/區塊3]

[區塊4]
總結：
信心分數：
[/區塊4]

數據來源：
{data_text}"""
                
                ans = call_ai_detective(prompt)
                
                # 提取精華內容
                b1 = extract_best_block(ans, 1)
                b2 = extract_best_block(ans, 2)
                b3 = extract_best_block(ans, 3)
                b4 = extract_best_block(ans, 4)

                st.markdown("---")
                st.subheader(f"🛡️ 偵探診斷報告：{current_id}")
                
                # 以專業卡片樣式呈現，並在 Python 端補上標題，確保 AI 漏掉標題也能正常顯示
                if b1:
                    with st.container():
                        st.markdown('<div class="report-card">', unsafe_allow_html=True)
                        st.markdown('<p class="indicator-label">【第一區塊：九項指標趨勢深度判讀】</p>', unsafe_allow_html=True)
                        st.write(b1)
                        st.markdown('</div>', unsafe_allow_html=True)

                if b2:
                    with st.container():
                        st.markdown('<div class="report-card">', unsafe_allow_html=True)
                        st.markdown('<p class="indicator-label">【第二區塊：指標矛盾整合與風險抓漏】</p>', unsafe_allow_html=True)
                        st.write(b2)
                        st.markdown('</div>', unsafe_allow_html=True)

                if b3:
                    with st.container():
                        st.markdown('<div class="report-card">', unsafe_allow_html=True)
                        st.markdown('<p class="indicator-label">【第三區塊：全方位操作戰略：雙重劇本】</p>', unsafe_allow_html=True)
                        st.write(b3)
                        st.markdown('</div>', unsafe_allow_html=True)

                if b4:
                    with st.container():
                        st.markdown('<div class="report-card">', unsafe_allow_html=True)
                        st.markdown('<p class="indicator-label">【第四區塊：偵探總結與信心分數】</p>', unsafe_allow_html=True)
                        st.write(b4)
                        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
