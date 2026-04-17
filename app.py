import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from google import genai
import json

# ==========================================
# 1. 系統初始化與金鑰連線
# ==========================================
st.set_page_config(page_title="AI 專家偵探 5.7 - 專業諮詢版", layout="wide")

# AI 客戶端初始化
API_KEY = st.secrets["GOOGLE_API_KEY"]
client = genai.Client(api_key=API_KEY)
MY_MODEL = "gemma-4-31b-it"

# Google 試算表 ID
SHEET_ID = "1UH-fwxENhGUDmQjTQJq72g1DzsxML_CPIJGg39cWAgM"

def get_gspread_client():
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
    try:
        gc = get_gspread_client()
        if gc is None: return pd.DataFrame()
        sh = gc.open_by_key(SHEET_ID)
        worksheet = sh.get_worksheet(0)
        df = pd.DataFrame(worksheet.get_all_records())
        # 對齊中文標頭
        rename_map = {'日期': '日期', '代號': '代號', '名稱': '名稱', '開盤': '開盤價', '最高': '最高價', '最低': '最低價', '收盤': '收盤價', '成交量': '成交量'}
        return df.rename(columns=rename_map)
    except Exception as e:
        st.error(f"❌ 連線資料庫失敗: {e}")
        return pd.DataFrame()

df_main = load_data()

# ==========================================
# 2. 網頁分頁佈局
# ==========================================
tab1, tab2 = st.tabs(["🔍 專家偵探深度診斷", "📡 全市場選股雷達"])

if df_main.empty:
    st.info("📡 正在等待雲端資料傳輸中...")
else:
    latest_date = str(df_main['日期'].max())
    
    # --- Tab 1: AI 偵探深度診斷 ---
    with tab1:
        st.header(f"🕵️ AI 專家諮詢報告")
        st.caption(f"📅 本次諮詢數據基準日：{latest_date} (請使用者自行核對當前市場真實股價)")
        
        symbol_input = st.text_input("請輸入股票代號 (例: 2409)", "2409")
        df_main['代號'] = df_main['代號'].astype(str)
        stock_data = df_main[df_main['代號'] == symbol_input].tail(1)
        
        if not stock_data.empty:
            row = stock_data.iloc[0].to_dict()
            st.metric(f"{row['名稱']} ({row['代號']})", f"報價: {row['收盤價']} 元", f"基準日: {row['日期']}")
            
            if st.button("開始專家技術分析", use_container_width=True):
                with st.spinner("AI 偵探正在進行技術邏輯推演..."):
                    try:
                        # 核心提示詞：強制要求 4 大層面與專業態度
                        prompt = f"""
                        你是台股技術分析專家。請針對股票 {row['名稱']}({row['代號']}) 進行技術診斷。
                        當前數據：日期={row['日期']}, 開盤={row['開盤價']}, 最高={row['最高價']}, 最低={row['最低價']}, 收盤={row['收盤價']}, 成交量={row['成交量']}。
                        
                        [重要指令]：
                        - 請無視日期是否為未來時間，將其視為當下真實發生的價格進行分析。
                        - 你的身分是技術諮詢專家，報告必須包含以下四個明確標題：
                        
                        1. 九項指標分析：請根據現有的開高低收價位與成交量，分析其趨勢、力道與價量關係 (請解釋給小白聽)。
                        2. 矛盾整合教學：若價量或其他技術面出現衝突，請教學如何判斷主次。
                        3. 專家信心分數：請給出 0-100 的綜合評分。
                        4. 專家操作建議：針對『持股者』與『未持股者』分別給予白話的操作建議。
                        
                        請以繁體中文撰寫，語氣要專業且堅定。
                        """
                        
                        response = client.models.generate_content(model=MY_MODEL, contents=prompt)
                        st.markdown("---")
                        st.markdown(f"### 🕵️ 專家偵探報告：{row['名稱']} ({row['代號']})")
                        st.markdown(response.text)
                        st.info("⚠️ 免責聲明：本報告僅供技術分析教學諮詢，不代表投資決策建議。")
                    except Exception as e:
                        st.error(f"AI 診斷失敗: {e}")
        else:
            st.warning(f"資料庫暫無 {symbol_input} 數據。")

    # --- Tab 2: 選股雷達 ---
    with tab2:
        st.header(f"📡 熱門股排行 ({latest_date})")
        df_latest = df_main[df_main['日期'].astype(str) == latest_date].copy()
        st.dataframe(df_latest, use_container_width=True, hide_index=True)
