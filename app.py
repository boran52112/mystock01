import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from google import genai
import json

# ==========================================
# 1. 基礎設定
# ==========================================
st.set_page_config(page_title="台股 AI 偵探系統 v5.4", layout="wide")

SHEET_ID = "1UH-fwxENhGUDmQjTQJq72g1DzsxML_CPIJGg39cWAgM"

def get_gspread_client():
    """連線 Google 試算表 (含私鑰修復)"""
    scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    try:
        creds_json_str = st.secrets["gcp_service_account_raw"]
        creds_info = json.loads(creds_json_str)
        if "private_key" in creds_info:
            creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")
        creds = Credentials.from_service_account_info(creds_info, scopes=scope)
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"❌ 鑰匙讀取失敗: {e}")
        return None

@st.cache_data(ttl=300)
def load_data():
    """根據截圖中的精確中文標頭進行抓取"""
    try:
        gc = get_gspread_client()
        if gc is None: return pd.DataFrame()
        sh = gc.open_by_key(SHEET_ID)
        worksheet = sh.get_worksheet(0)
        df = pd.DataFrame(worksheet.get_all_records())
        
        # 💡 根據你的截圖，精確對齊中文標頭
        rename_map = {
            '日期': 'Date', 
            '代號': 'Code', 
            '名稱': 'Name', 
            '開盤': 'OpeningPrice',
            '最高': 'HighestPrice',
            '最低': 'LowestPrice',
            '收盤': 'ClosingPrice', 
            '成交量': 'TradeVolume'
        }
        df = df.rename(columns=rename_map)
        return df
    except Exception as e:
        st.error(f"❌ 資料抓取失敗，請確認試算表第一列標頭是否正確。錯誤: {e}")
        return pd.DataFrame()

# 載入資料
df_main = load_data()

# ==========================================
# 2. 網頁分頁
# ==========================================
tab1, tab2 = st.tabs(["🔍 專家偵探深度診斷", "📡 全市場選股雷達"])

if df_main.empty:
    st.info("📡 正在從雲端試算表讀取資料，請稍候...")
else:
    # 取得最新日期
    if 'Date' in df_main.columns:
        latest_date = str(df_main['Date'].max())
    else:
        latest_date = "未知日期"

    # --- Tab 1: AI 診斷 ---
    with tab1:
        st.header(f"🕵️ AI 偵探報告 (日期: {latest_date})")
        symbol = st.text_input("請輸入股票代號 (例: 2409)", "2409")
        
        df_main['Code'] = df_main['Code'].astype(str)
        stock_info = df_main[df_main['Code'] == symbol].tail(1)
        
        if not stock_info.empty:
            row = stock_info.iloc[0].to_dict()
            col1, col2, col3 = st.columns(3)
            col1.metric("公司", row.get('Name', '未知'))
            col2.metric("最新收盤", f"{row.get('ClosingPrice', '0')} 元")
            col3.metric("成交量", f"{row.get('TradeVolume', '0')}")
            
            if st.button("開始 AI 深度分析", use_container_width=True):
                with st.spinner("AI 偵探思考中..."):
                    try:
                        ai_cl
