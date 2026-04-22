import streamlit as st
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials
import google.generativeai as genai
import json
import os

# ==========================================
# 1. 初始化與金鑰設定
# ==========================================
st.set_page_config(page_title="台股 AI 偵探戰情室", layout="centered")

try:
    # 讀取試算表金鑰 (對應你的 Secrets 名稱)
    gcp_json_str = st.secrets["gcp_service_account_raw"]
    gcp_info = json.loads(gcp_json_str)
    gcp_info['private_key'] = gcp_info['private_key'].replace('\\n', '\n')
    
    # 讀取 AI 金鑰
    ai_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=ai_key)
except Exception as e:
    st.error(f"❌ 金鑰讀取失敗，請確認 Secrets 設定。錯誤: {e}")
    st.stop()

# ==========================================
# 2. 證據獲取邏輯 (加入去重與 5 日過濾)
# ==========================================
def get_detective_data(symbol):
    symbol = symbol.upper().strip()
    yf_symbol = f"{symbol}.TW" if ".TW" not in symbol and ".TWO" not in symbol else symbol

    # 方案 A：搜尋歷史資料庫
    try:
        scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        creds = Credentials.from_service_account_info(gcp_info, scopes=scope)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key("1UH-fwxENhGUDmQjTQJq72g1DzsxML_CPIJGg39cWAgM")
        wks = sh.get_worksheet(0)
        
        full_df = pd.DataFrame(wks.get_all_records())
        
        # --- 核心去重邏輯 ---
        # 依照日期與股號去重，只保留最後一筆紀錄
        full_df = full_df.drop_duplicates(subset=['日期', '股號'], keep='last')
        
        # 搜尋該股號最新的 5 筆紀錄
        target_df = full_df[full_df['股號'].astype(str).str.contains(symbol)].tail(5)
        
        if not target_df.empty:
            return target_df, "📊 數據來源：200大歷史資料庫 (已自動去重)"
    except:
        pass

    # 方案 B：現場即時偵查
    try:
        df = yf.download(yf_symbol, period="4mo", interval="1d", progress=False)
        if df.empty: return None, None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df.ta.sma(length=5, append=True)
        df.ta.sma(length=20, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.stoch(k=9, d=3, append=True)
        df.ta.macd(append=True)
        df.ta.bbands(length=20, append=True)
        df['漲跌幅%'] = df['Close'].pct_change() * 100
        
        res = df.tail(5).copy()
        res['日期'] = res.index.strftime('%Y-%m-%d')
        res['股號'] = yf_symbol
        res['股名'] = "現場偵查標的"
        
        name_map = {
            'Close': '收盤價', 'SMA_5': 'MA5', 'SMA_20': 'MA20', 'RSI_14': 'RSI14',
            'STOCHk_9_3_3': 'K', 'STOCHd_9_3_3': 'D', 'MACD_12_26_9': 'MACD',
            'BBU_20_2.0': 'BB_Upper', 'BBL_20_2.0': 'BB_Lower'
        }
        res = res.rename(columns={k: v for k, v in name_map.items() if k in res.columns})
        return res.round(2), "🔍 數據來源：現場即時偵查"
    except:
        return None, None

# ==========================================
# 3. 網頁 UI 與 AI 診斷
# ==========================================
st.title("🕵️‍♂️ 台股 AI 偵探戰情室")
st.write("2026 模擬版數據分析引擎")

stock_input = st.text_input("輸入目標股號 (如 2330):", "").strip()

if stock_input:
    with st.spinner("偵探正在彙整 5 日證據表格..."):
        data, source_info = get_detective_data(stock_input)
        
        if data is not None:
            st.success(source_info)
            # 顯示表格
            st.table(data[['日期', '收盤價', 'MA5', 'RSI14', '漲跌幅%']])
            
            # 準備 AI Prompt (修正縮進問題)
            table_md = data.to_markdown(index=False)
            prompt = f"""
            你是一位派駐在 2026 年的台股資深分析偵探。請針對以下證據進行診斷。
            你必須完全使用【繁體中文】回覆，禁止出現英文標題。

            【分析標的】：{stock_input}
            【5日證據表格】：
            {table_md}

            請嚴格依照以下格式輸出報告：

            📌 偵探報告編號：2026-{stock_id if 'stock_id' in locals() else stock_input}-SCAN
            🕵️ 偵探身份：2026年派駐資深分析偵探

            ### 第一區塊：【九項指標與趨勢判讀】
            (分析5日數據走向、目前的市場心理是恐慌還是興奮？)

            ### 第二區塊：【指標矛盾整合與風險抓漏】
            (檢查是否有指標背離或主力誘多陷阱？)

            ### 第三區塊：【全方位操作戰略：雙重劇本】
            1. 保守型劇本：(支撐點、停損位、進場邏輯)
            2. 激進型劇本：(突破點、目標價、短線追擊)

            ### 第四區塊：【偵探總結與信心分數】
            * 總結：(用一句話定調短線趨勢)
            * 信心分數：(0-100)
            """
            
            try:
                # 呼叫 2026 年 Gemma 4 模型
                model = genai.GenerativeModel('models/gemma-4-31b-it')
                response = model.generate_content(prompt)
                
                st.markdown("---")
                st.markdown("### 📝 AI 偵探深度診斷報告")
                st.markdown(response.text)
                
            except Exception as ai_e:
                st.error(f"AI 診斷中斷 (型號連線問題): {ai_e}")
                # 備援機制
                st.info("正在嘗試使用備援邏輯重啟分析...")
                model_backup = genai.GenerativeModel('gemini-1.5-flash')
                response = model_backup.generate_content(prompt)
                st.markdown(response.text)
        else:
            st.error("證據不足，無法取得資料。")

st.markdown("---")
st.caption("AI 偵探系統 v7.5 | 數據校準時間：2026-04-22")
