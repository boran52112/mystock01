import streamlit as st
import pandas as pd
import gspread
import json
import google.generativeai as genai
from google.oauth2.service_account import Credentials

# --- 1. 戰情室基本設定 ---
st.set_page_config(page_title="台股 AI 偵探系統 v5.0", layout="wide")
CURRENT_DATE = "2026-04-22"
# 鎖定試算表 ID (由指揮官提供)
SHEET_ID = "1UH-fwxENhGUDmQjTQJq72g1DzsxML_CPIJGg39cWAgM"

# --- 2. 核心數據讀取函式 ---
def get_data_from_sheets():
    try:
        # 從 Secrets 讀取 JSON 字串並解析
        creds_json_str = st.secrets["gcp_service_account_raw"]
        creds_info = json.loads(creds_json_str)
        
        scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        creds = Credentials.from_service_account_info(creds_info, scopes=scope)
        client = gspread.authorize(creds)
        
        # 使用 ID 開啟，徹底解決名稱錯誤問題
        sheet = client.open_by_key(SHEET_ID).sheet1 
        data = sheet.get_all_records()
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"❌ 數據庫連線失敗。請確認 Secrets 格式與試算表權限。錯誤詳情: {e}")
        return None

# --- 3. AI 偵探診斷函式 ---
def call_ai_detective(prompt):
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    # 模擬 2026 年環境下最強模型
    model = genai.GenerativeModel('gemini-1.5-pro') 
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"偵探回報錯誤: {str(e)}"

# --- 4. 主程式介面 ---
def main():
    st.title(f"🕵️‍♂️ 台股 AI 偵探戰情室 (模擬日期: {CURRENT_DATE})")

    # 讀取並檢查數據
    df = get_data_from_sheets()
    if df is None or df.empty:
        st.warning("目前數據庫內無數據，請確認 scanner.py 是否已成功上傳資料。")
        return

    # 側邊欄：股票選擇
    if 'Stock_ID' in df.columns:
        # 取得所有股票代碼並去重
        stock_list = sorted(df['Stock_ID'].unique())
        selected_stock = st.sidebar.selectbox("🎯 選擇目標股票進行診斷", stock_list)
    else:
        st.error("試算表格式錯誤：找不到 'Stock_ID' 欄位。")
        return

    if selected_stock:
        # 篩選該股最近 5 筆數據 (代表 5 個交易日)
        df_selected = df[df['Stock_ID'] == selected_stock].tail(5)

        # --- 第一階段：數據表格化 (100% 繁體中文) ---
        st.subheader(f"📊 {selected_st
