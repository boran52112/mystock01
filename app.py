import streamlit as st
import pandas as pd
import gspread
import json
import google.generativeai as genai
from google.oauth2.service_account import Credentials
import yfinance as yf
import re

# --- 1. 視覺優化 (加大字體 & 調整標題) ---
st.set_page_config(page_title="台股 AI 偵探系統 v5.0", layout="wide")
st.markdown("""
    <style>
    .stMarkdown p, .stMarkdown li { font-size: 1.25rem !important; line-height: 1.6; }
    .stTable { font-size: 1.1rem !important; }
    .report-block { background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #ff4b4b; }
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

# --- 4. 提取區塊內容的精確邏輯 ---
def extract_block(text, block_num):
    # 使用正規表達式抓取 [區塊X] 內容 [/區塊X]
    pattern = rf"\[區塊{block_num}\](.*?)\[/區塊{block_num}\]"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def call_ai_detective(prompt):
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        model = genai.GenerativeModel('gemma-4-31b-it')
        # 嚴格禁止思考過程的系統指令
        response = model.generate_content(prompt)
        return response.text
    except Exception as e: return f"AI 偵探錯誤: {str(e)}"

# --- 5. 主程式 ---
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
            with st.spinner("正在排版報告，請稍候..."):
                data_text = df_display.to_string(index=False)
                # 修改 Prompt，要求 AI 使用強製標籤
                prompt = f"""你現在是台股 AI 偵探。嚴格禁止輸出英文、禁止自我檢查、禁止思考過程。
請直接填寫以下四個標籤內的內容，嚴禁在標籤外寫任何字：
[區塊1] 針對九項指標(5/20MA, RSI, 量價等)進行深度判讀 [/區塊1]
[區塊2] 找出指標間的矛盾與潛在風險 [/區塊2]
[區塊3] 給出保守型與激進型操作劇本 [/區塊3]
[區塊4] 一句話總結與信心分數(0-100) [/區塊4]
數據來源：{data_text}"""
                
                ans = call_ai_detective(prompt)
                
                # 分別提取四個區塊
                b1 = extract_block(ans, 1)
                b2 = extract_block(ans, 2)
                b3 = extract_block(ans, 3)
                b4 = extract_block(ans, 4)

                st.markdown("---")
                st.subheader(f"🛡️ 偵探診斷報告：{current_id}")
                
                if b1: st.info(f"**第一區塊：【九項指標趨勢深度判讀】**\n\n{b1}")
                if b2: st.warning(f"**第二區塊：【指標矛盾整合與風險抓漏】**\n\n{b2}")
                if b3: st.success(f"**第三區塊：【全方位操作戰略：雙重劇本】**\n\n{b3}")
                if b4: st.error(f"**第四區塊：【偵探總結與信心分數】**\n\n{b4}")
                
                if not b1 and not b2: # 如果標籤提取失敗，顯示原始內容作為保險
                    st.write(ans)

if __name__ == "__main__":
    main()
