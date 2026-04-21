import streamlit as st
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials
import google.generativeai as genai
import json
import os
import pytz
from datetime import datetime

# ==========================================
# 1. 介面與金鑰初始化 (對應你的 Secrets 名稱)
# ==========================================
st.set_page_config(page_title="台股 AI 偵探戰情室", layout="centered")

try:
    # 1. 讀取 Google Sheets 金鑰 (對應你截圖的 gcp_service_account_raw)
    gcp_json_str = st.secrets["gcp_service_account_raw"]
    gcp_info = json.loads(gcp_json_str)
    gcp_info['private_key'] = gcp_info['private_key'].replace('\\n', '\n')
    
    # 2. 讀取 AI 金鑰 (對應你截圖的 GOOGLE_API_KEY)
    ai_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=ai_key)
    
except Exception as e:
    st.error(f"❌ 金鑰讀取失敗，請確認 Secrets 中的名稱是否正確。錯誤訊息: {e}")
    st.stop()

# ==========================================
# 2. 資料獲取邏輯
# ==========================================
def get_analysis_data(symbol):
    symbol = symbol.upper().strip()
    yf_symbol = f"{symbol}.TW" if ".TW" not in symbol and ".TWO" not in symbol else symbol

    # 優先搜尋資料庫 (方案 A)
    try:
        scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        creds = Credentials.from_service_account_info(gcp_info, scopes=scope)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key("1UH-fwxENhGUDmQjTQJq72g1DzsxML_CPIJGg39cWAgM")
        wks = sh.get_worksheet(0)
        
        full_df = pd.DataFrame(wks.get_all_records())
        # 搜尋最近 5 筆
        target_df = full_df[full_df['股號'].astype(str).str.contains(symbol)].tail(5)
        
        if not target_df.empty and len(target_df) >= 3:
            return target_df, "📊 數據來源：200大歷史資料庫"
    except:
        pass

    # 現場下載 (方案 B)
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
        res['股名'] = "現場即時偵查"
        
        # 欄位對齊
        name_map = {
            'Close': '收盤價', 'SMA_5': 'MA5', 'SMA_20': 'MA20', 'RSI_14': 'RSI14',
            'STOCHk_9_3_3': 'K', 'STOCHd_9_3_3': 'D', 'MACD_12_26_9': 'MACD'
        }
        res = res.rename(columns={k: v for k, v in name_map.items() if k in res.columns})
        return res, "🔍 數據來源：現場即時偵查"
    except:
        return None, None

# ==========================================
# 3. UI 呈現與 AI 診斷
# ==========================================
st.title("🕵️‍♂️ 台股 AI 偵探戰情室")
st.write("輸入股號後，AI 將進行 5 日趨勢與 9 項技術指標的深度診斷。")

stock_input = st.text_input("輸入股號 (如 2330):", "").strip()

if stock_input:
    with st.spinner("偵探正在翻閱證據資料..."):
        data, source_info = get_analysis_data(stock_input)
        
        if data is not None:
            st.success(source_info)
            # 顯示精簡表格
            st.table(data[['日期', '收盤價', 'MA5', 'RSI14', '漲跌幅%']].tail(5))
            
            # 準備 AI Prompt
            table_md = data.to_markdown(index=False)
            prompt = f"""
            你是一位台股資深技術偵探。請根據以下5日證據表格，進行深度診斷：
            
            【分析標的】：{stock_input}
            【證據表格】：
            {table_md}
            
            【要求】：
            1. 區塊一：九項指標與趨勢判讀（含市場心理分析）
            2. 區塊二：指標矛盾整合與風險抓漏
            3. 區塊三：操作戰略（保守與激進雙劇本）
            4. 區塊四：總結與偵探信心分數（0-100）
            
            內容要完整詳細，適合手機長滑動閱讀。
            """
            
            try:
                # 嘗試使用 Google 平台上的 Gemma 模型
                # 如果 gemma-7b-it 無法使用，會嘗試使用 gemini-1.5-flash (也是非常快速強大的)
                try:
                    model = genai.GenerativeModel('gemma-7b-it')
                    response = model.generate_content(prompt)
                except:
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    response = model.generate_content(prompt)
                
                st.markdown("---")
                st.markdown("### 📝 AI 偵探深度診斷報告")
                st.markdown(response.text)
                
            except Exception as ai_e:
                st.error(f"AI 診斷中斷: {ai_e}")
        else:
            st.error("查無資料，請確認股號。")

st.markdown("---")
st.caption("AI 偵探系統 v7.2 | 本地/雲端同步運作版")
