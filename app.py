import streamlit as st
import pandas as pd
import os
from google import genai 

# 1. AI 客戶端設定
API_KEY = st.secrets["GOOGLE_API_KEY"]
client = genai.Client(api_key=API_KEY)
MY_MODEL = "gemma-4-31b-it"

st.set_page_config(page_title="AI 專家偵探 3.0", layout="wide")

# 2. 側邊欄：全市場雷達的篩選器
st.sidebar.title("🕵️ 偵探篩選面板")
st.sidebar.info("在此設定你的選股策略，結果將顯示在『選股雷達』頁籤。")

opt = {"不限": "All", "看多 (1)": 1, "看空 (-1)": -1}
f_ma = st.sidebar.selectbox("均線趨勢", list(opt.keys()))
f_macd = st.sidebar.selectbox("MACD 動能", list(opt.keys()))
f_kd = st.sidebar.selectbox("KD 隨機指標", list(opt.keys()))
f_rsi = st.sidebar.selectbox("RSI 強弱", list(opt.keys()))
f_bb = st.sidebar.selectbox("布林位置", list(opt.keys()))
f_shadow = st.sidebar.selectbox("下影線訊號", list(opt.keys()))
f_gap = st.sidebar.selectbox("跳空缺口", list(opt.keys()))
f_inst = st.sidebar.selectbox("法人籌碼", list(opt.keys()))
f_margin = st.sidebar.selectbox("融資籌碼", list(opt.keys()))

# 3. 主頁面內容
tab1, tab2 = st.tabs(["🔍 單檔 AI 偵探診斷", "📡 全市場選股雷達"])

# --- Tab 1: AI 深度診斷 ---
with tab1:
    st.header(f"🤖 AI 專家診斷系統 ({MY_MODEL})")
    symbol_input = st.text_input("輸入股票代號 (例: 2330)", "2330")
    
    if st.button("開始偵探分析"):
        if os.path.exists("daily_scan.csv"):
            df_scan = pd.read_csv("daily_scan.csv")
            stock_id = f"{symbol_input}.TW" if "." not in symbol_input else symbol_input
            stock_data = df_scan[df_scan['代號'] == stock_id].tail(1)
            
            if not stock_data.empty:
                row = stock_data.iloc[0].to_dict()
                
                # 🎓 知識點：偵探式提詞 (Prompt Engineering)
                prompt = f"""
                你是一位毒舌但極其專業的台股投資偵探。請針對以下數據進行『推理分析』：
                股票標的：{row['代號']}
                數據內容：{row}

                請提供以下架構的報告：
                1. **數據衝突與矛盾**：如果均線看多但RSI超買，或籌碼不動但技術面噴發，請指出其中的詭異之處。
                2. **多空辯論迴路**：分別代表多頭與空頭專家，進行激烈的觀點攻防。
                3. **信心分數**：請給出 0-100 的綜合看多信心分。
                4. **給小白的最終結案**：用最白話的方式告訴新手，現在該買、該賣還是睡覺？
                """
                
                with st.spinner("偵探正在翻閱卷宗..."):
                    try:
                        response = client.models.generate_content(model=MY_MODEL, contents=prompt)
                        st.markdown("### 🔍 偵探結案報告")
                        st.write(response.text)
                    except Exception as e:
                        st.error(f"AI 罷工了: {e}")
            else:
                st.warning("資料庫中找不到此代號，請先確認它在全市場掃描清單中。")

# --- Tab 2: 選股雷達 ---
with tab2:
    st.header("📡 全市場即時選股雷達")
    if os.path.exists("daily_scan.csv"):
        df = pd.read_csv("daily_scan.csv")
        
        # 執行過濾邏輯
        f_df = df.copy()
        filters = {
            '均線': f_ma, 'MACD': f_macd, 'KD': f_kd, 'RSI': f_rsi, 
            '布林': f_bb, '下影線': f_shadow, '缺口': f_gap, 
            '法人': f_inst, '融資': f_margin
        }
        
        for col, val in filters.items():
            if opt[val] != "All":
                f_df = f_df[f_df[col] == opt[val]]
        
        st.subheader(f"📊 篩選結果：共 {len(f_df)} 檔符合條件")
        st.dataframe(f_df, use_container_width=True)
    else:
        st.error("尚未偵測到 daily_scan.csv 檔案。")
