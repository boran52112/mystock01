import streamlit as st
import pandas as pd
import os
from google import genai 

# 1. 初始化 AI
API_KEY = st.secrets["GOOGLE_API_KEY"]
client = genai.Client(api_key=API_KEY)
MY_MODEL = "gemma-4-31b-it"

st.set_page_config(page_title="AI 專家偵探 - 數據連動版", layout="wide")

tab1, tab2 = st.tabs(["🔍 單檔 AI 診斷", "📡 全市場選股雷達"])

# --- Tab 1: 單檔 AI 診斷 (現在會讀取數據了！) ---
with tab1:
    st.header(f"🤖 AI 專家診斷 ({MY_MODEL})")
    symbol_input = st.text_input("輸入股票代號 (例如: 2002)", "2002")
    
    # 這裡我們自動幫代號補上後綴
    symbol = f"{symbol_input}.TW"
    
    if st.button("開始深度分析"):
        if os.path.exists("daily_scan.csv"):
            df_scan = pd.read_csv("daily_scan.csv")
            # 從 CSV 找這檔股票最新的資料
            stock_data = df_scan[df_scan['代號'] == symbol].tail(1)
            
            if not stock_data.empty:
                # 🎓 知識點：這就是「餵資料」給 AI
                row = stock_data.iloc[0]
                indicators = f"""
                股票代號: {symbol}
                日期: {row['日期']}
                收盤價: {row['收盤價']}
                指標狀態 (1為看多, -1為看空, 0為中立):
                - 均線狀態: {row['均線']}
                - 法人籌碼: {row['法人']}
                """
                
                with st.spinner("AI 正在閱讀掃描數據..."):
                    try:
                        prompt = f"你是一位台股大師。以下是我為你準備的數據，請不要質疑股票是否存在，直接針對數據進行分析與操作建議：\n{indicators}"
                        response = client.models.generate_content(model=MY_MODEL, contents=prompt)
                        st.markdown("### 📊 數據驅動分析報告")
                        st.write(response.text)
                    except Exception as e:
                        st.error(f"AI 呼叫失敗: {e}")
            else:
                st.warning(f"⚠️ 在 daily_scan.csv 中找不到 {symbol} 的數據。請確認該標的是否在掃描範圍內。")
        else:
            st.error("❌ 找不到掃描資料檔 daily_scan.csv。")

# --- Tab 2: 全市場選股雷達 (維持原樣) ---
with tab2:
    st.header("📡 全市場即時雷達")
    if os.path.exists("daily_scan.csv"):
        df = pd.read_csv("daily_scan.csv")
        st.dataframe(df, use_container_width=True)
