import streamlit as st
import pandas as pd
import os
from google import genai 

# 1. AI 客戶端初始化
API_KEY = st.secrets["GOOGLE_API_KEY"]
client = genai.Client(api_key=API_KEY)
MY_MODEL = "gemma-4-31b-it"

# 2. 網頁基本設定 (設定為自動適應螢幕寬度)
st.set_page_config(page_title="AI 專家偵探", layout="wide")

# --- 自定義小工具：將數字轉為手機友善圖示 ---
def icon_map(val):
    if val == 1: return "🔴" # 多頭用紅色 (台股習慣)
    if val == -1: return "🟢" # 空頭用綠色
    return "⚪" # 中立

# 3. 頁面分頁
tab1, tab2 = st.tabs(["🔍 單檔診斷", "📡 選股雷達"])

# --- Tab 1: 單檔 AI 診斷 ---
with tab1:
    st.header("🤖 AI 偵探分析")
    symbol_input = st.text_input("輸入代號", "2330", key="single_search")
    
    if st.button("開始偵探分析", use_container_width=True): # 按鈕變大，手機好按
        if os.path.exists("daily_scan.csv"):
            df_scan = pd.read_csv("daily_scan.csv")
            stock_id = f"{symbol_input}.TW" if "." not in symbol_input else symbol_input
            stock_data = df_scan[df_scan['代號'] == stock_id].tail(1)
            
            if not stock_data.empty:
                row = stock_data.iloc[0].to_dict()
                
                # AI 提詞
                prompt = f"你是一位專業台股偵探。針對 {row['代號']} 數據：{row}。請給出：1.數據矛盾點 2.多空辯論 3.百分比信心分數 4.小白建議。"
                
                with st.spinner("偵探分析中..."):
                    try:
                        response = client.models.generate_content(model=MY_MODEL, contents=prompt)
                        st.success("分析完成！")
                        # 顯示分析結果
                        st.markdown(response.text)
                    except Exception as e:
                        st.error(f"AI 故障：{e}")
            else:
                st.warning("找不到該股票數據。")

# --- Tab 2: 全市場選股雷達 (手機優化版) ---
with tab2:
    st.header("📡 選股雷達")
    
    if os.path.exists("daily_scan.csv"):
        df = pd.read_csv("daily_scan.csv")
        
        # 🎓 知識點：在手機版，我們用 Expander 取代 Sidebar
        with st.expander("🛠️ 點我設定『多空濾網』", expanded=False):
            st.write("設定篩選條件 (不限即代表略過該指標)")
            opt = {"不限": "All", "看多 (🔴)": 1, "看空 (🟢)": -1}
            
            # 手機版佈局：兩兩一組
            c1, c2 = st.columns(2)
            with c1: f_ma = st.selectbox("均線", list(opt.keys()))
            with c2: f_inst = st.selectbox("法人", list(opt.keys()))
            
            c3, c4 = st.columns(2)
            with c3: f_kd = st.selectbox("KD", list(opt.keys()))
            with c4: f_rsi = st.selectbox("RSI", list(opt.keys()))

        # 執行過濾
        f_df = df.copy()
        # (這裡簡化示範，僅放四個最常用的)
        if opt[f_ma] != "All": f_df = f_df[f_df['均線'] == opt[f_ma]]
        if opt[f_inst] != "All": f_df = f_df[f_df['法人'] == opt[f_inst]]
        if opt[f_kd] != "All": f_df = f_df[f_df['KD'] == opt[f_kd]]
        if opt[f_rsi] != "All": f_df = f_df[f_df['RSI'] == opt[f_rsi]]

        st.subheader(f"📊 結果 ({len(f_df)} 檔)")
        
        # 🎓 知識點：表格美化
        # 我們只顯示關鍵欄位，並將 1/-1 轉為圖示，節省手機空間
        display_df = f_df[['代號', '收盤價', '均線', '法人', 'KD', 'RSI']].copy()
        for col in ['均線', '法人', 'KD', 'RSI']:
            display_df[col] = display_df[col].apply(icon_map)
            
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
    else:
        st.error("找不到資料檔。")
