import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from google import genai
import json

# ==========================================
# 1. 系統初始化與金鑰連線
# ==========================================
st.set_page_config(page_title="AI 專家偵探 5.6 - 雲端版", layout="wide")

# AI 客戶端初始化
API_KEY = st.secrets["GOOGLE_API_KEY"]
client = genai.Client(api_key=API_KEY)
MY_MODEL = "gemma-4-31b-it"

# Google 試算表 ID
SHEET_ID = "1UH-fwxENhGUDmQjTQJq72g1DzsxML_CPIJGg39cWAgM"

def get_gspread_client():
    """連線 Google 試算表 (支援 Secrets raw 格式)"""
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
    """從雲端抓取資料並將欄位對齊原本的功能"""
    try:
        gc = get_gspread_client()
        if gc is None: return pd.DataFrame()
        sh = gc.open_by_key(SHEET_ID)
        worksheet = sh.get_worksheet(0)
        df = pd.DataFrame(worksheet.get_all_records())
        
        # 將雲端英文標頭對齊你原本程式碼要求的中文標頭
        rename_map = {
            '日期': '日期', 'Code': '代號', 'Name': '名稱', 
            '開盤': '開盤價', '最高': '最高價', '最低': '最低價', 
            '收盤': '收盤價', '成交量': '成交量'
        }
        df = df.rename(columns=rename_map)
        return df
    except Exception as e:
        st.error(f"❌ 連線資料庫失敗: {e}")
        return pd.DataFrame()

def get_status_ui(val):
    if val == 1: return "🔴 看多"
    if val == -1: return "🟢 看空"
    return "⚪ 中立"

# --- 啟動載入 ---
df_main = load_data()

# ==========================================
# 2. 網頁佈局
# ==========================================
tab1, tab2 = st.tabs(["🔍 專家偵探深度診斷", "📡 全市場選股雷達"])

if df_main.empty:
    st.info("📡 正在等待雲端資料傳輸中...")
else:
    # 取得最新日期作為顯示基準
    latest_date = str(df_main['日期'].max())
    
    # --- Tab 1: AI 偵探深度診斷 (恢復 4 大層面分析) ---
    with tab1:
        st.header(f"🕵️ 偵探結案報告 (資料日期: {latest_date})")
        symbol_input = st.text_input("請輸入股票代號 (例: 2409)", "2409")
        
        # 比對代號 (統一轉字串)
        df_main['代號'] = df_main['代號'].astype(str)
        stock_data = df_main[df_main['代號'] == symbol_input].tail(1)
        
        if not stock_data.empty:
            row = stock_data.iloc[0].to_dict()
            st.metric(f"{row['名稱']} ({row['代號']})", f"收盤: {row['收盤價']} 元", f"日期: {row['日期']}")
            
            if st.button("開始深度偵探分析", use_container_width=True):
                with st.spinner("AI 偵探教學整合中..."):
                    try:
                        # 恢復你要求的 4 大層面 Prompt
                        prompt = f"""你是台股偵探專家。針對股票 {row['名稱']}({row['代號']})，今日數據如下：{row}。
                        請撰寫報告：
                        1.九項指標分析(解釋給小白聽的意義)
                        2.矛盾整合教學(若指標有衝突，該如何判斷)
                        3.信心分數(0-100)
                        4.針對持股與否的操作建議。
                        請用白話繁體中文，並以 Markdown 格式呈現。"""
                        
                        response = client.models.generate_content(model=MY_MODEL, contents=prompt)
                        st.markdown("---")
                        st.markdown(response.text)
                    except Exception as e:
                        st.error(f"AI 診斷失敗: {e}")
        else:
            st.warning(f"資料庫暫無 {symbol_input} 數據。")

    # --- Tab 2: 選股雷達 (保留濾網功能) ---
    with tab2:
        st.header(f"📡 全市場選股雷達 ({latest_date})")
        
        # 篩選出最新一天的資料
        df_latest = df_main[df_main['日期'].astype(str) == latest_date].copy()
        
        with st.expander("🛠️ 設定『多空偵探濾網』", expanded=True):
            opt = {"不限": "All", "看多 (🔴)": 1, "看空 (🟢)": -1}
            cols = st.columns(3)
            f_ma = cols[0].selectbox("均線趨勢", list(opt.keys()))
            f_macd = cols[1].selectbox("MACD 動能", list(opt.keys()))
            f_kd = cols[2].selectbox("KD 隨機指標", list(opt.keys()))
            
            cols2 = st.columns(3)
            f_rsi = cols2[0].selectbox("RSI 強弱", list(opt.keys()))
            f_inst = cols2[1].selectbox("法人籌碼", list(opt.keys()))
            f_mg = cols2[2].selectbox("融資籌碼", list(opt.keys()))

        # 動態過濾邏輯：只有當欄位存在於資料表時，才進行過濾
        f_df = df_latest.copy()
        check_list = {'均線': f_ma, 'MACD': f_macd, 'KD': f_kd, 'RSI': f_rsi, '法人': f_inst, '融資': f_mg}
        
        for col, val in check_list.items():
            # 只有當資料庫有這些計算後的指標列，且使用者選了非「不限」時，才過濾
            if col in f_df.columns and opt[val] != "All":
                f_df = f_df[f_df[col] == opt[val]]

        st.subheader(f"📊 篩選結果 ({len(f_df)} 檔)")
        
        # 整理顯示欄位
        base_cols = ['日期', '代號', '名稱', '收盤價', '成交量']
        # 檢查資料庫目前有哪些指標欄位可以顯示
        indicator_cols = [c for c in ['均線', 'MACD', 'KD', 'RSI', '法人', '融資', '布林', '下影線'] if c in f_df.columns]
        
        display_df = f_df[base_cols + indicator_cols].copy()
        for c in indicator_cols:
            display_df[c] = display_df[c].apply(get_status_ui)
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        if not indicator_cols:
            st.info("💡 目前雲端資料庫僅包含『價格與成交量』。明天的開發計畫將會加入 MA/MACD/KD 等 AI 指標計算，屆時濾網將會正式啟用。")
