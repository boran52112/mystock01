import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from google import genai
import json

# ==========================================
# 1. 初始化與網頁設定
# ==========================================
st.set_page_config(page_title="AI 專家偵探 5.1", layout="wide")

# AI 客戶端初始化 (從 Secrets 讀取)
API_KEY = st.secrets["GOOGLE_API_KEY"]
client = genai.Client(api_key=API_KEY)
MY_MODEL = "gemma-4-31b-it"

# Google 試算表設定
SHEET_ID = "1UH-fwxENhGUDmQjTQJq72g1DzsxML_CPIJGg39cWAgM"

def get_gspread_client():
    """專為『大括號 JSON 字串』設計的連線函式"""
    scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    
    try:
        # 讀取你在 Secrets 裡用三個單引號包起來的那個 raw 字串
        creds_json_str = st.secrets["gcp_service_account_raw"]
        # 將文字轉回電腦看得懂的 JSON 格式
        creds_info = json.loads(creds_json_str)
        
        creds = Credentials.from_service_account_info(creds_info, scopes=scope)
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"金鑰格式讀取失敗，請檢查 Secrets 設定。錯誤訊息: {e}")
        return None

@st.cache_data(ttl=600) # 每 10 分鐘快取一次
def load_data_from_sheets():
    """從雲端試算表抓取所有資料"""
    try:
        gc = get_gspread_client()
        if gc is None: return pd.DataFrame()
        
        sh = gc.open_by_key(SHEET_ID)
        worksheet = sh.get_worksheet(0)
        data = worksheet.get_all_records()
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"連線雲端資料庫失敗: {e}")
        return pd.DataFrame()

def get_status_ui(val):
    if val == 1: return "🔴 看多"
    if val == -1: return "🟢 看空"
    return "⚪ 中立"

# --- 啟動時先抓取資料 ---
df_main = load_data_from_sheets()

# ==========================================
# 2. 網頁分頁佈局
# ==========================================
tab1, tab2 = st.tabs(["🔍 專家偵探深度診斷", "📡 全市場選股雷達"])

if df_main.empty:
    st.warning("📡 正在等待雲端資料傳輸中... 若長時間沒出現，請確認 Google 試算表內是否有資料。")
else:
    # 找出資料庫中最新的日期
    latest_date = str(df_main['Date'].max())
    
    # --- Tab 1: AI 診斷 ---
    with tab1:
        st.header(f"🕵️ 偵探結案報告 (資料日期: {latest_date})")
        
        # 讓使用者輸入代號
        symbol_input = st.text_input("請輸入股票代號 (例: 2330)", "2330")
        
        # 從資料庫中找這檔股票
        stock_data = df_main[df_main['Code'].astype(str) == symbol_input].tail(1)
        
        if not stock_data.empty:
            row = stock_data.iloc[0].to_dict()
            
            # 顯示主要數據
            col1, col2, col3 = st.columns(3)
            col1.metric("公司名稱", f"{row['Name']}")
            col2.metric("最新收盤價", f"{row['ClosingPrice']} 元")
            col3.metric("今日成交量", f"{row['TradeVolume']} 張")
            
            if st.button("開始 AI 深度診斷", use_container_width=True):
                with st.spinner("AI 偵探正在解讀雲端大數據..."):
                    try:
                        prompt = (
                            f"你是台股分析專家。股票 {row['Name']}({row['Code']}) 在 {row['Date']} 的表現如下：\n"
                            f"開盤 {row['OpeningPrice']}、最高 {row['HighestPrice']}、最低 {row['LowestPrice']}、收盤 {row['ClosingPrice']}。\n"
                            f"請針對今日走勢給予評價，並提供信心分數與操作建議。請用繁體中文。"
                        )
                        response = client.models.generate_content(model=MY_MODEL, contents=prompt)
                        st.markdown("---")
                        st.markdown(response.text)
                    except Exception as e:
                        st.error(f"AI 診斷失敗: {e}")
        else:
            st.info(f"🔍 搜尋中... 資料庫中尚未找到代號 {symbol_input} 的資料。")

    # --- Tab 2: 選股雷達 ---
    with tab2:
        st.header(f"📡 今日熱門股排行 ({latest_date})")
        
        # 只顯示最新那一天的資料
        df_latest = df_main[df_main['Date'].astype(str) == latest_date].copy()
        
        # 簡單整理欄位給小白看
        display_df = df_latest.rename(columns={
            'Date': '日期',
            'Code': '代號',
            'Name': '名稱',
            'ClosingPrice': '收盤價',
            'TradeVolume': '成交量'
        })
        
        st.write("以下為全市場成交量前 200 名的股票：")
        st.dataframe(
            display_df[['日期', '代號', '名稱', '收盤價', '成交量']], 
            use_container_width=True, 
            hide_index=True
        )
