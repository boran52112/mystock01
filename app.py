import streamlit as st
import pandas as pd
import gspread
import json
import google.generativeai as genai
from google.oauth2.service_account import Credentials
import yfinance as yf
from datetime import datetime, timedelta

# --- 1. 視覺優化 (加大字體) ---
st.set_page_config(page_title="台股 AI 偵探系統 v5.0", layout="wide")
st.markdown("""
    <style>
    .stMarkdown p, .stMarkdown li { font-size: 1.25rem !important; }
    .stTable { font-size: 1.1rem !important; }
    input { font-size: 1.2rem !important; }
    </style>
    """, unsafe_allow_html=True)

CURRENT_DATE = "2026-04-22"
SHEET_ID = "1UH-fwxENhGUDmQjTQJq72g1DzsxML_CPIJGg39cWAgM"

# --- 2. 數據讀取函式 ---
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

# --- 3. 智能 yfinance 抓取 (自動補全後綴) ---
def fetch_smart_yf(stock_id):
    # 嘗試列表：原樣、加 .TW、加 .TWO
    suffixes = ["", ".TW", ".TWO"]
    for suf in suffixes:
        test_id = f"{stock_id}{suf}"
        try:
            df_yf = yf.download(test_id, period="2mo", progress=False)
            if not df_yf.empty and len(df_yf) > 20:
                # 計算指標
                df_yf['MA5'] = df_yf['Close'].rolling(window=5).mean()
                df_yf['MA20'] = df_yf['Close'].rolling(window=20).mean()
                # RSI14
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
        except:
            continue
    return pd.DataFrame()

# --- 4. AI 診斷函式 ---
def call_ai_detective(prompt):
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 偵探錯誤: {str(e)}"

# --- 5. 主程式 ---
def main():
    st.title(f"🕵️‍♂️ 台股 AI 偵探戰情室")
    
    # 手機友善輸入佈局
    user_input = st.text_input("🔢 請輸入股號 (例如: 2330 或 2356)", placeholder="請輸入純數字股號").strip()
    search_btn = st.button("🔍 開始偵查")

    if search_btn and user_input:
        with st.spinner("正在調閱數據檔案..."):
            df_all = get_data_from_sheets()
            df_selected = pd.DataFrame()
            
            # 1. 在資料庫中搜尋 (模糊匹配)
            if not df_all.empty and '股號' in df_all.columns:
                # 找出股號中包含使用者輸入字串的資料 (忽略大小寫)
                df_all['股號'] = df_all['股號'].astype(str)
                df_selected = df_all[df_all['股號'].str.contains(user_input)].tail(5)

            # 2. 如果資料庫找不到，啟動智能補全抓取
            if df_selected.empty:
                df_selected = fetch_smart_yf(user_input)

            if not df_selected.empty:
                real_id = df_selected['股號'].iloc[-1]
                st.subheader(f"📊 {real_id} 指標觀測站")
                
                # 顯示表格
                show_cols = ['日期', '收盤價', '成交量', 'MA5', 'MA20', 'RSI14']
                st.table(df_selected[show_cols])

                # AI 診斷按鈕
                if st.button(f"🚀 啟動 {real_id} AI 深度診斷"):
                    with st.spinner("偵探正在判讀指標趨勢..."):
                        data_text = df_selected[show_cols].to_string(index=False)
                        prompt = f"""
你現在是台股 AI 偵探。禁止任何英文與廢話。
直接從「第一區塊：【九項指標趨勢深度判讀】」開始。
數據：{data_text}
區塊：第一區塊(判讀)、第二區塊(風險)、第三區塊(劇本)、第四區塊(信心分數)。
"""
                        ans = call_ai_detective(prompt)
                        # 強力切除雜訊
                        target = "第一區塊"
                        if target in ans:
                            ans = ans[ans.find(target):]
                        
                        st.markdown("---")
                        st.success(ans)
            else:
                st.error(f"❌ 查無此股 ({user_input})，請確認代碼是否正確。")

if __name__ == "__main__":
    main()
