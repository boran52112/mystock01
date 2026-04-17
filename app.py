import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from google import genai
import json

# ==========================================
# 1. 初始化與網頁設定
# ==========================================
st.set_page_config(page_title="AI 專家偵探 5.2", layout="wide")

# AI 客戶端初始化 (從 Secrets 讀取)
API_KEY = st.secrets["GOOGLE_API_KEY"]
client = genai.Client(api_key=API_KEY)
MY_MODEL = "gemma-4-31b-it"

# Google 試算表設定
SHEET_ID = "1UH-fwxENhGUDmQjTQJq72g1DzsxML_CPIJGg39cWAgM"

def get_gspread_client():
    """自動修復私鑰格式的連線函式"""
    scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    
    try:
        # 1. 讀取你在 Secrets 裡貼的那一大坨 JSON 文字
        creds_json_str = st.secrets["gcp_service_account_raw"]
        creds_info = json.loads(creds_json_str)
        
        # 2. 【核心手術】修復私鑰中的換行符號 (解決 MalformedFraming 關鍵)
        # 有時候私鑰裡的 \n 會被誤解，這行代碼會把它修正回來
        if "private_key" in creds_info:
            creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")
        
        creds = Credentials.from_service_account_info(creds_info, scopes=scope)
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"❌ 金鑰讀取失敗！請檢查 Secrets 中的括號是否完整。詳細錯誤: {e}")
        return None

@st.cache_data(ttl=300) # 每 5 分鐘快取一次
def load_data_from_sheets():
    """從雲端試算表抓取所有資料"""
    try:
        gc = get_gspread_client()
        if gc is None: return pd.DataFrame()
        
        sh = gc.open_by_key(SHEET_ID)
        worksheet = sh.get_worksheet(0)
        data = worksheet.get_all_records()
        
        if not data:
            st.warning("⚠️ 試算表內目前沒有資料，請確認 scanner.py 是否有成功執行。")
            return pd.DataFrame()
            
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"❌ 連線資料庫失敗: {e}")
        return pd.DataFrame()

# --- 啟動時先抓取資料 ---
df_main = load_data_from_sheets()

# ==========================================
# 2. 網頁分頁佈局
# ==========================================
tab1, tab2 = st.tabs(["🔍 專家偵探深度診斷", "📡 全市場選股雷達"])

if df_main.empty:
    st.info("📡 正在等待雲端資料傳輸中... 若畫面持續空白，請檢查 Google 試算表權限。")
else:
    # 確保日期格式正確
    latest_date = str(df_main['Date'].max())
    
    # --- Tab 1: AI 診斷 ---
    with tab1:
        st.header(f"🕵️ 偵探結案報告 (資料日期: {latest_date})")
        symbol_input = st.text_input("請輸入股票代號 (例: 2330)", "2330")
        
        # 過濾該代號的資料 (處理代號可能是數字或字串的情況)
        df_main['Code'] = df_main['Code'].astype(str)
        stock_data = df_main[df_main['Code'] == symbol_input].tail(1)
        
        if not stock_data.empty:
            row = stock_data.iloc[0].to_dict()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("公司名稱", f"{row['Name']}")
            col2.metric("最新收盤價", f"{row['ClosingPrice']} 元")
            col3.metric("今日成交量", f"{row['TradeVolume']} 張")
            
            if st.button("開始 AI 深度診斷", use_container_width=True):
                with st.spinner("AI 偵探正在解讀雲端大數據..."):
                    try:
                        prompt = (
                            f"你是台股分析專家。股票 {row['Name']}({row['Code']}) 在 {row['Date']} 的表現如下：\n"
                            f"開盤 {row['OpeningPrice']}、最高 {row['HighestPrice
