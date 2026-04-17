import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from google import genai
import json

# ==========================================
# 1. 初始化與金鑰設定
# ==========================================
st.set_page_config(page_title="台股 AI 偵探系統 v5.5", layout="wide")

# Google 試算表 ID
SHEET_ID = "1UH-fwxENhGUDmQjTQJq72g1DzsxML_CPIJGg39cWAgM"

def get_gspread_client():
    """連線 Google 試算表"""
    scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    try:
        # 讀取 Secrets 裡的 JSON 字串
        creds_json_str = st.secrets["gcp_service_account_raw"]
        creds_info = json.loads(creds_json_str)
        
        # 修復私鑰格式
        if "private_key" in creds_info:
            creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")
            
        creds = Credentials.from_service_account_info(creds_info, scopes=scope)
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"❌ 鑰匙讀取失敗: {e}")
        return None

@st.cache_data(ttl=300)
def load_data():
    """抓取雲端數據"""
    try:
        gc = get_gspread_client()
        if gc is None: return pd.DataFrame()
        sh = gc.open_by_key(SHEET_ID)
        worksheet = sh.get_worksheet(0)
        df = pd.DataFrame(worksheet.get_all_records())
        
        # 自動對齊中文標頭
        rename_map = {
            '日期': 'Date', '代號': 'Code', '名稱': 'Name', 
            '開盤': 'Open', '最高': 'High', '最低': 'Low', 
            '收盤': 'Close', '成交量': 'Vol'
        }
        return df.rename(columns=rename_map)
    except Exception as e:
        st.error(f"❌ 資料抓取失敗: {e}")
        return pd.DataFrame()

# 載入資料
df_main = load_data()

# ==========================================
# 2. 網頁分頁佈局
# ==========================================
tab1, tab2 = st.tabs(["🔍 專家偵探深度診斷", "📡 全市場選股雷達"])

if df_main.empty:
    st.info("📡 正在讀取雲端數據，請稍候...")
else:
    # 取得最新日期
    latest_date = str(df_main['Date'].max())
    
    # --- Tab 1: AI 診斷 ---
    with tab1:
        st.header(f"🕵️ AI 偵探報告 (日期: {latest_date})")
        symbol = st.text_input("請輸入股票代號 (例: 2409)", "2409")
        
        df_main['Code'] = df_main['Code'].astype(str)
        stock_info = df_main[df_main['Code'] == symbol].tail(1)
        
        if not stock_info.empty:
            row = stock_info.iloc[0].to_dict()
            col1, col2, col3 = st.columns(3)
            col1.metric("公司", row.get('Name'))
            col2.metric("最新收盤", f"{row.get('Close')} 元")
            col3.metric("成交量", f"{row.get('Vol')}")
            
            if st.button("開始 AI 分析"):
                with st.spinner("AI 偵探思考中..."):
                    try:
                        # 呼叫 Gemini AI
                        ai_key = st.secrets["GOOGLE_API_KEY"]
                        ai_client = genai.Client(api_key=ai_key)
                        
                        prompt = f"你是台股專家。股票 {row['Name']} 在 {row['Date']} 收盤價為 {row['Close']} 元。請提供簡短的操作建議與信心分數。請用繁體中文。"
                        
                        response = ai_client.models.generate_content(model="gemma-4-31b-it", contents=prompt)
                        st.markdown("---")
                        st.write(response.text)
                    except Exception as e:
                        st.error(f"AI 分析出錯: {e}")
        else:
            st.warning(f"資料庫暫無代號 {symbol}")

    # --- Tab 2: 選股雷達 ---
    with tab2:
        st.header(f"📊 今日熱門排行 ({latest_date})")
        df_show = df_main[df_main['Date'].astype(str) == latest_date].copy()
        st.dataframe(df_show, use_container_width=True, hide_index=True)
