import streamlit as st
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials
import json
import os
from groq import Groq
import pytz
from datetime import datetime

# ==========================================
# 1. 介面初始化與金鑰檢查
# ==========================================
st.set_page_config(page_title="台股 AI 偵探系統", layout="centered")

try:
    # 讀取 Google Sheets 金鑰
    gcp_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_KEY"])
    gcp_info['private_key'] = gcp_info['private_key'].replace('\\n', '\n')
    
    # 讀取 Groq AI 金鑰
    ai_api_key = st.secrets["AI_API_KEY"]
    client = Groq(api_key=ai_api_key)
except Exception as e:
    st.error("金鑰讀取失敗，請確認 Secrets 設定中包含 GCP_SERVICE_ACCOUNT_KEY 與 AI_API_KEY。")
    st.stop()

# ==========================================
# 2. 資料獲取核心 (方案 B: 試算表優先，現場下載補位)
# ==========================================
def get_analysis_data(symbol):
    symbol = symbol.upper()
    if not symbol.endswith(".TW") and not symbol.endswith(".TWO"):
        yf_symbol = f"{symbol}.TW"
    else:
        yf_symbol = symbol

    # --- A 計劃：從 Google Sheets 資料庫搜尋歷史 ---
    try:
        scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        creds = Credentials.from_service_account_info(gcp_info, scopes=scope)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key("1UH-fwxENhGUDmQjTQJq72g1DzsxML_CPIJGg39cWAgM")
        wks = sh.get_worksheet(0)
        
        all_rows = wks.get_all_records()
        full_df = pd.DataFrame(all_rows)
        
        # 搜尋該股號最近 5 筆紀錄
        target_df = full_df[full_df['股號'].astype(str).str.contains(symbol)].tail(5)
        
        if not target_df.empty and len(target_df) >= 3:
            return target_df, "資料來源：200大歷史資料庫"
    except Exception as e:
        pass # 若試算表失敗，轉 B 計劃

    # --- B 計劃：現場下載並計算指標 ---
    try:
        df = yf.download(yf_symbol, period="4mo", interval="1d", progress=False)
        if df.empty: return None, None
        
        # 處理 MultiIndex 格式問題
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # 計算技術指標
        df.ta.sma(length=5, append=True)
        df.ta.sma(length=20, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.stoch(k=9, d=3, append=True)
        df.ta.macd(append=True)
        df.ta.bbands(length=20, append=True)
        df['漲跌幅%'] = df['Close'].pct_change() * 100
        
        latest_5 = df.tail(5).copy()
        latest_5['日期'] = latest_5.index.strftime('%Y-%m-%d')
        latest_5['股號'] = yf_symbol
        latest_5['股名'] = "現場偵查標的"
        
        # 重新整理欄位，方便 AI 閱讀
        cols_map = {
            'Close': '收盤價', 'Volume': '成交量', 
            'SMA_5': 'MA5', 'SMA_20': 'MA20', 'RSI_14': 'RSI14',
            'STOCHk_9_3_3': 'K', 'STOCHd_9_3_3': 'D',
            'MACD_12_26_9': 'MACD', 'BBU_20_2.0': 'BB_Upper', 'BBL_20_2.0': 'BB_Lower'
        }
        # 檢查欄位是否存在並改名
        existing_cols = {old: new for old, new in cols_map.items() if old in latest_5.columns}
        latest_5 = latest_5.rename(columns=existing_cols)
        
        return latest_5, "資料來源：現場即時偵查"
    except:
        return None, None

# ==========================================
# 3. 網頁 UI 設計
# ==========================================
st.title("🕵️‍♂️ 台股 AI 偵探戰情室")
st.write("這是一個基於 2026 模擬版數據與 5 日趨勢動能的深度診斷系統。")

stock_id = st.text_input("請輸入股號 (例如 2330):", "").strip()

if stock_id:
    with st.spinner('偵探正在分析趨勢表格中...'):
        df_evidence, source_text = get_analysis_data(stock_id)
        
        if df_evidence is not None:
            st.success(f"📊 證據搜尋成功！ {source_text}")
            
            # 以表格形式展示給使用者看
            st.table(df_evidence[['日期', '收盤價', 'RSI14', 'MA5', '漲跌幅%']].tail(5))
            
            # 準備發送給 AI 的完整表格字串
            # 我們精簡欄位，確保 AI 專注於關鍵指標
            table_for_ai = df_evidence.to_markdown(index=False)
            
            # --- 建立專業診斷 Prompt ---
            prompt_content = f"""
            你是一位專業的台股技術分析偵探。現在請針對下方提供的【5日技術指標趨勢表】進行深度診斷。
            你要找出趨勢的轉折，而不是只看一天的數據。

            【分析標的】：{stock_id}
            【5日歷史證據表格】：
            {table_for_ai}

            【分析指令 - 請依照以下四個區塊回覆，必須詳盡且具備洞察力】：

            區塊一：【九項指標與趨勢判讀】
            請整合 MA5/MA20、RSI、KD、MACD、布林通道的5日動向。
            並針對目前的「市場心理」進行分析：投資人現在是處於恐慌、觀望、還是過度興奮？

            區塊二：【指標矛盾整合與風險抓漏】
            是否存在指標打架的情況？(例如股價創新高但RSI下降)。找出隱藏的轉折風險或誘多陷阱。

            區塊三：【全方位操作戰略：雙重劇本】
            1. 保守型劇本：給出支撐點、停損建議與波段進場邏輯。
            2. 激進型劇本：給出突破追擊點、短線目標價與當沖風險控制。

            區塊四：【偵探總結與信心分數】
            用一段話總結此股的短線定調。並給出一個 0-100 的「偵探信心分數」。

            請使用繁體中文回覆。內容必須詳細，適合手機長滑動閱讀。
            """

            # 呼叫 Groq API (Gemma-4-31b-it)
            try:
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt_content}],
                    model="gemma2-9b-it", # 注意：若 gemma-4-31b-it 無法連線，此處可換成 Groq 目前最強的 gemma2-9b-it 或 llama-3.1-70b
                )
                response_text = chat_completion.choices[0].message.content
                
                # 輸出診斷報告
                st.markdown("---")
                st.markdown("### 📝 AI 偵探深度診斷報告")
                st.markdown(response_text)
                
            except Exception as ai_err:
                st.error(f"AI 診斷暫時無法連線: {ai_err}")
        else:
            st.error("無法取得該股資料，請確認股號是否存在。")

st.markdown("---")
st.caption("台股 AI 偵探系統 v7.0 | 數據僅供參考，請自行判斷投資風險。")
