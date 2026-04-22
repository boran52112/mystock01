import streamlit as st
import pandas as pd
import gspread
import json
import google.generativeai as genai
from google.oauth2.service_account import Credentials
import yfinance as yf
from datetime import datetime, timedelta

# --- 1. 視覺優化 ---
st.set_page_config(page_title="台股 AI 偵探系統 v5.0", layout="wide")
st.markdown("""
    <style>
    .stMarkdown p, .stMarkdown li { font-size: 1.25rem !important; }
    .stTable { font-size: 1.1rem !important; }
    input { font-size: 1.2rem !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 初始化狀態記憶 (Session State) ---
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None
if 'current_id' not in st.session_state:
    st.session_state.current_id = ""

CURRENT_DATE = "2026-04-22"
SHEET_ID = "1UH-fwxENhGUDmQjTQJq72g1DzsxML_CPIJGg39cWAgM"

# --- 3. 數據讀取與 yfinance 救援 (保持不變) ---
def get_data_from_sheets():
    try:
        creds_json_str = st.secrets["gcp_service_account_raw"]
        creds_info = json.loads(creds_json_str)
        scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        creds = Credentials.from_service_account_info(creds_info, scopes=scope)
        client = gspread.authorize(creds)
        sheet = client.open_by_key(SHEET_ID).sheet1 
        return pd.DataFrame(sheet.get_all_records())
    except:
        return pd.DataFrame()

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
                    '股號': test_id,
                    '股名': "即時救援數據",
                    '收盤價': df_yf['Close'].round(2),
                    '成交量': df_yf['Volume'],
                    'MA5': df_yf['MA5'].round(2),
                    'MA20': df_yf['MA20'].round(2),
                    'RSI14': df_yf['RSI14'].round(2)
                })
        except: continue
    return pd.DataFrame()

def call_ai_detective(prompt):
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        model = genai.GenerativeModel('gemma-4-31b-it')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 偵探錯誤: {str(e)}"

# --- 4. 主程式 ---
def main():
    st.title(f"🕵️‍♂️ 台股 AI 偵探戰情室")
    
    # 輸入區
    user_input = st.text_input("🔢 請輸入股號 (例如: 2330)", placeholder="輸入純數字即可").strip()
    search_btn = st.button("🔍 開始偵查")

    # 按下查詢按鈕：更新記憶
    if search_btn and user_input:
        with st.spinner("正在調閱數據檔案..."):
            df_all = get_data_from_sheets()
            df_selected = pd.DataFrame()
            
            if not df_all.empty and '股號' in df_all.columns:
                df_all['股號'] = df_all['股號'].astype(str)
                df_selected = df_all[df_all['股號'].str.contains(user_input)].tail(5)

            if df_selected.empty:
                df_selected = fetch_smart_yf(user_input)

            if not df_selected.empty:
                # 把結果存進記憶裡
                st.session_state.stock_data = df_selected
                st.session_state.current_id = df_selected['股號'].iloc[-1]
            else:
                st.session_state.stock_data = None
                st.error(f"❌ 查無此股 ({user_input})")

    # 顯示區：如果記憶裡有資料，就顯示出來 (不論是否按下查詢按鈕)
    if st.session_state.stock_data is not None:
        df_display = st.session_state.stock_data
        current_id = st.session_state.current_id
        
        st.subheader(f"📊 {current_id} 指標觀測站")
        show_cols = ['日期', '收盤價', '成交量', 'MA5', 'MA20', 'RSI14']
        st.table(df_display[show_cols])

        # AI 診斷按鈕
        if st.button(f"🚀 啟動 {current_id} AI 深度診斷"):
            with st.spinner("偵探正在判讀指標趨勢..."):
                data_text = df_display[show_cols].to_string(index=False)
                prompt = f"""你現在是台股 AI 偵探。禁止英文。
直接從「第一區塊：【九項指標趨勢深度判讀】」開始。
數據：{data_text}"""
                ans = call_ai_detective(prompt)
                target = "第一區塊"
                if target in ans:
                    ans = ans[ans.find(target):]
                
                st.markdown("---")
                st.success(ans)

if __name__ == "__main__":
    main()
