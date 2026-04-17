import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from google import genai
import json

# ==========================================
# 1. 網頁初始設定
# ==========================================
st.set_page_config(page_title="AI 專家偵探 5.2", layout="wide")

# 從 Secrets 讀取 AI 金鑰
API_KEY = st.secrets["GOOGLE_API_KEY"]
client = genai.Client(api_key=API_KEY)
MY_MODEL = "gemma-4-31b-it"

# Google 試算表設定
SHEET_ID = "1UH-fwxENhGUDmQjTQJq72g1DzsxML_CPIJGg39cWAgM"

def get_gspread_client():
    """專門處理 JSON 字串的連線功能"""
    scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    try:
        # 讀取 Secrets 裡那一長串原始 JSON 文字
        creds_json_str = st.secrets["gcp_service_account_raw"]
        creds_info = json.loads(creds_json_str)
        
        # 修復私鑰中的換行符號問題
        if "private_key" in creds_info:
            creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")
        
        creds = Credentials.from_service_account_info(creds_info, scopes=scope)
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"❌ 鑰匙讀取失敗：{e}")
        return None

@st.cache_data(ttl=300)
def load_data_from_sheets():
    """抓取雲端資料庫資料"""
    try:
        gc = get_gspread_client()
        if gc is None: return pd.DataFrame()
        sh = gc.open_by_key(SHEET_ID)
        worksheet = sh.get_worksheet(0)
        data = worksheet.get_all_records()
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"❌ 資料庫連線失敗：{e}")
        return pd.DataFrame()

# 啟動時先抓資料
df_main = load_data_from_sheets()

# ==========================================
# 2. 網頁介面佈局
# ==========================================
tab1, tab2 = st.tabs(["🔍 專家偵探深度診斷", "📡 全市場選股雷達"])

if df_main.empty:
    st.info("📡 偵探正在連線雲端資料庫中，請稍候...")
else:
    # 獲取最新日期
    latest_date = str(df_main['Date'].max())
    
    with tab1:
        st.header(f"🕵️ 偵探結案報告 ({latest_date})")
        symbol = st.text_input("請輸入股票代號 (例: 2330)", "2330")
        
        # 搜尋股票
        df_main['Code'] = df_main['Code'].astype(str)
        stock_data = df_main[df_main['Code'] == symbol].tail(1)
        
        if not stock_data.empty:
            row = stock_data.iloc[0].to_dict()
            col1, col2, col3 = st.columns(3)
            col1.metric("名稱", row['Name'])
            col2.metric("收盤價", f"{row['ClosingPrice']} 元")
            col3.metric("成交量", row['TradeVolume'])
            
            if st.button("開始 AI 偵探診斷", use_container_width=True):
                with st.spinner("AI 偵探解讀中..."):
                    try:
                        prompt = f"你是台股偵探。分析股票 {row['Name']}({row['Code']}) 日期 {row['Date']}：開盤{row['OpeningPrice']}、最高{row['HighestPrice']}、最低{row['LowestPrice']}、收盤{row['ClosingPrice']}。請撰寫簡短分析與信心分數。"
                        response = client.models.generate_content(model=MY_MODEL, contents=prompt)
                        st.markdown("---")
                        st.markdown(response.text)
                    except Exception as e:
                        st.error(f"AI 故障：{e}")
        else:
            st.warning(f"資料庫尚未找到代號 {symbol}")

    with tab2:
        st.header(f"📡 今日全市場排行 ({latest_date})")
        df_latest = df_main[df_main['Date'].astype(str) == latest_date].copy()
        show_df = df_latest.rename(columns={'Date':'日期','Code':'代號','Name':'名稱','ClosingPrice':'收盤價','TradeVolume':'成交量'})
        st.dataframe(show_df[['日期','代號','名稱','收盤價','成交量']], use_container_width=True, hide_index=True)
