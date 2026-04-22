import streamlit as st
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials
import google.generativeai as genai
import json
import os
from datetime import datetime

# ==========================================
# 1. 偵探系統初始化 (2026 模擬環境)
# ==========================================
st.set_page_config(page_title="台股 AI 偵探戰情室", layout="centered")

try:
    # 對應 Secrets 中的名稱
    gcp_json_str = st.secrets["gcp_service_account_raw"]
    gcp_info = json.loads(gcp_json_str)
    gcp_info['private_key'] = gcp_info['private_key'].replace('\\n', '\n')
    
    # 對應 Secrets 中的名稱
    ai_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=ai_key)
    
except Exception as e:
    st.error(f"❌ 金鑰讀取失敗，請確認 Secrets 設定。錯誤: {e}")
    st.stop()

# ==========================================
# 2. 證據獲取邏輯 (方案 B: 現場辦案模組)
# ==========================================
def get_detective_data(symbol):
    symbol = symbol.upper().strip()
    yf_symbol = f"{symbol}.TW" if ".TW" not in symbol and ".TWO" not in symbol else symbol

    # --- 方案 A：搜尋 5 日歷史資料庫 ---
    try:
        scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        creds = Credentials.from_service_account_info(gcp_info, scopes=scope)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key("1UH-fwxENhGUDmQjTQJq72g1DzsxML_CPIJGg39cWAgM")
        wks = sh.get_worksheet(0)
        
        all_df = pd.DataFrame(wks.get_all_records())
        # 搜尋最近 5 筆
        target_df = all_df[all_df['股號'].astype(str).str.contains(symbol)].tail(5)
        
        if not target_df.empty and len(target_df) >= 3:
            return target_df, "📊 數據來源：200大歷史資料庫"
    except:
        pass

    # --- 方案 B：現場即時偵查 (針對資料庫未收錄股票) ---
    try:
        df = yf.download(yf_symbol, period="4mo", interval="1d", progress=False)
        if df.empty: return None, None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # 計算 9 大核心指標
        df.ta.sma(length=5, append=True)
        df.ta.sma(length=20, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.stoch(k=9, d=3, append=True)
        df.ta.macd(append=True)
        df.ta.bbands(length=20, append=True)
        df['漲跌幅%'] = df['Close'].pct_change() * 100
        
        res = df.tail(5).copy()
        # 清理時間雜訊
        res['日期'] = res.index.strftime('%Y-%m-%d')
        res['股號'] = yf_symbol
        res['股名'] = "現場即時分析標的"
        
        # 統一欄位名稱
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
# 3. 偵探戰情室 UI
# ==========================================
st.title("🕵️‍♂️ 台股 AI 偵探戰情室")
st.caption(f"當前系統時間：2026-04-21 | AI 核心：Gemma 4-31B")

stock_id = st.text_input("輸入目標股號 (如 2330):", "").strip()

if stock_id:
    with st.spinner("偵探正在整理證據表格..."):
        data, info = get_detective_data(stock_id)
        
        if data is not None:
            st.success(info)
            # 呈現精簡證據表
            st.table(data[['日期', '收盤價', 'MA5', 'RSI14', '漲跌幅%']].tail(5))
            
            # 準備發送給 AI 的證據 (轉為 Markdown 表格)
            evidence_table = data.to_markdown(index=False)
            
            # --- 2026 版偵探指令 (Prompt) ---
            prompt = f"""
            你是一位派駐在 2026 年的台股資深偵探。請針對下方提供的【5日技術指標證據】進行深度診斷。
            你需要從數字的變化中，讀出市場參與者的「心理狀態」。

            【分析標的】：{stock_id}
            【證據表格】：
            {evidence_table}

            【分析指令 - 請依照以下四個區塊回覆，必須詳盡且具備洞察力】：

            區塊一：【九項指標與趨勢判讀】
            分析 MA5/MA20、RSI、KD、MACD、布林通道的5日走向。
            目前的「市場心理」是過熱、恐慌、還是觀望？請從價量變化給出證據。

            區塊二：【指標矛盾整合與風險抓漏】
            是否存在指標打架（如價增量縮、RSI背離）？找出隱藏的轉折風險或誘多陷阱。

            區塊三：【全方位操作戰略：雙重劇本】
            1. 保守型劇本：給出支撐點、停損建議與穩健進場邏輯。
            2. 激進型劇本：給出突破追擊點、短線目標價與當沖風險控制。

            區塊四：【偵探總結與信心分數】
            用一段話給出最終定調，並給出一個 0-100 的「偵探信心分數」。

            請使用繁體中文回覆。報告需詳細且層次分明，適合手機長滑動閱讀。
            """

            try:
                # 指定使用 2026 年 4 月最新的 Gemma 4 模型
                model = genai.GenerativeModel('models/gemma-4-31b-it')
                response = model.generate_content(prompt)
                
                st.markdown("---")
                st.markdown("### 📝 AI 偵探深度診斷報告")
                st.markdown(response.text)
                
            except Exception as ai_e:
                st.error(f"AI 偵探暫時無法辦案 (可能型號名稱在 2026 環境有微調): {ai_e}")
        else:
            st.error("證據不足，無法取得該股資料。")

st.markdown("---")
st.caption("AI 偵探系統 v7.4 | 2026 模擬版")
