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
# 1. 介面初始化 (確保這是在程式最前面)
# ==========================================
st.set_page_config(page_title="台股 AI 偵探戰情室", layout="centered")

# 讀取金鑰邏輯
if "GCP_SERVICE_ACCOUNT_KEY" not in st.secrets or "AI_API_KEY" not in st.secrets:
    st.error("❌ Secrets 遺失：請確保設定了 GCP_SERVICE_ACCOUNT_KEY 與 AI_API_KEY")
    st.stop()

try:
    gcp_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_KEY"])
    gcp_info['private_key'] = gcp_info['private_key'].replace('\\n', '\n')
    client = Groq(api_key=st.secrets["AI_API_KEY"])
except Exception as e:
    st.error(f"❌ 初始化錯誤: {e}")
    st.stop()

# ==========================================
# 2. 資料獲取邏輯 (方案 B：資料庫優先，現場為輔)
# ==========================================
def get_stock_report_data(symbol):
    symbol = symbol.upper().strip()
    yf_symbol = f"{symbol}.TW" if ".TW" not in symbol and ".TWO" not in symbol else symbol

    # 方案 A：試算表搜尋
    try:
        scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        creds = Credentials.from_service_account_info(gcp_info, scopes=scope)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key("1UH-fwxENhGUDmQjTQJq72g1DzsxML_CPIJGg39cWAgM")
        wks = sh.get_worksheet(0)
        
        full_df = pd.DataFrame(wks.get_all_records())
        target_df = full_df[full_df['股號'].astype(str).str.contains(symbol)].tail(5)
        
        if not target_df.empty and len(target_df) >= 3:
            return target_df, "📊 數據來源：200大歷史資料庫"
    except:
        pass

    # 方案 B：現場抓取
    try:
        df = yf.download(yf_symbol, period="4mo", interval="1d", progress=False)
        if df.empty: return None, None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # 計算指標
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
        res['股名'] = "即時偵查標的"
        
        # 欄位改名
        name_map = {
            'Close': '收盤價', 'SMA_5': 'MA5', 'SMA_20': 'MA20', 'RSI_14': 'RSI14',
            'STOCHk_9_3_3': 'K', 'STOCHd_9_3_3': 'D', 'MACD_12_26_9': 'MACD'
        }
        res = res.rename(columns={k: v for k, v in name_map.items() if k in res.columns})
        return res, "🔍 數據來源：現場即時偵查"
    except:
        return None, None

# ==========================================
# 3. UI 呈現
# ==========================================
st.title("🕵️‍♂️ 台股 AI 偵探戰情室")
st.write("輸入股號後，偵探會調閱最近 5 日的技術證據進行深度診斷。")

target_stock = st.text_input("輸入股號 (如 2330):", "").strip()

if target_stock:
    with st.spinner("偵探正在翻閱證據資料..."):
        data, info_text = get_stock_report_data(target_stock)
        
        if data is not None:
            st.success(info_text)
            st.dataframe(data[['日期', '收盤價', 'MA5', 'RSI14', '漲跌幅%']].tail(5))
            
            # 準備 AI 指令
            evidence_md = data.to_markdown(index=False)
            prompt = f"""
            你是一位台股資深技術偵探。請根據以下5日證據表格，進行四個區塊的深度診斷：
            
            【證據表格】：
            {evidence_md}
            
            【診斷要求】：
            區塊一：九項指標與趨勢判讀（含市場心理分析）
            區塊二：指標矛盾整合與風險抓漏
            區塊三：全方位操作戰略（保守與激進雙劇本）
            區塊四：偵探總結與信心分數（0-100）
            
            請使用繁體中文，內容詳細且適合手機閱讀。
            """
            
            try:
                # 這裡使用 gemma2-9b-it (這是 Groq 目前最穩定的 Gemma 模型)
                # 如果你想試試 gemma-7b-it 也可以，但 gemma2-9b-it 效果更好
                chat = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="gemma2-9b-it", 
                )
                st.markdown("---")
                st.markdown("### 📝 AI 偵探診斷報告")
                st.write(chat.choices[0].message.content)
            except Exception as ai_e:
                st.error(f"AI 辦案遇到阻礙: {ai_e}")
        else:
            st.error("查無資料，請確認股號正確。")

st.markdown("---")
st.caption("AI 偵探系統 v7.1 | 趨勢數據僅供參考")
