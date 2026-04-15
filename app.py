import streamlit as st
import pandas as pd
import os
# 🎓 知識點：這是 Google 最新的 SDK 引入方式
from google import genai 

# 1. 初始化 AI 客戶端
# 我們從 Streamlit Secrets 讀取 Key
API_KEY = st.secrets["GOOGLE_API_KEY"]
client = genai.Client(api_key=API_KEY)

# 🎓 知識點：寫死你指定的模型名稱
MY_MODEL = "gemma-4-31b-it"

st.set_page_config(page_title="AI 專家偵探 - 全新進化版", layout="wide")

# 建立分頁
tab1, tab2 = st.tabs(["🔍 單檔 AI 診斷", "📡 全市場選股雷達"])

# --- Tab 1: 單檔 AI 診斷 ---
with tab1:
    st.header(f"🤖 AI 專家診斷 (模型: {MY_MODEL})")
    symbol = st.text_input("輸入股票代號", "2330")
    
    if st.button("開始深度分析"):
        with st.spinner("AI 正在分析 9 大指標資料..."):
            try:
                # 這裡假設抓取了 9 指標的數據 (暫時用文字代替)
                # 實際運作時，這裡會帶入我們掃描出來的數據
                prompt_content = f"你是一位台股分析專家。請針對股票 {symbol} 的技術面與籌碼面進行辯論分析，並給予投資建議。"
                
                # 🎓 知識點：這是最新版 google-genai 的呼叫方式
                response = client.models.generate_content(
                    model=MY_MODEL,
                    contents=prompt_content
                )
                
                st.markdown("### 📊 分析報告")
                st.write(response.text)
                
            except Exception as e:
                st.error(f"❌ 發生錯誤：{e}")
                st.info("💡 如果出現 'Model not found'，代表此模型名稱可能需要特定權限或名稱微調。")

# --- Tab 2: 全市場選股雷達 (讀取 CSV) ---
with tab2:
    st.header("📡 全市場即時雷達")
    if os.path.exists("daily_scan.csv"):
        df = pd.read_csv("daily_scan.csv")
        
        # 簡單篩選器佈局
        st.subheader("快速篩選")
        c1, c2 = st.columns(2)
        with c1:
            f_ma = st.checkbox("📈 均線看多")
        with c2:
            f_inst = st.checkbox("🏢 法人買超")
            
        # 執行過濾
        temp_df = df.copy()
        if f_ma: temp_df = temp_df[temp_df['均線'] == 1]
        if f_inst: temp_df = temp_df[temp_df['法人'] == 1]
        
        st.dataframe(temp_df, use_container_width=True)
    else:
        st.warning("尚未找到 daily_scan.csv，請等待自動掃描執行。")
