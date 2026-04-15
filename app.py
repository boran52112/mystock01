import streamlit as st
import pandas as pd
import os
from google import genai

# 1. AI 初始化
API_KEY = st.secrets["GOOGLE_API_KEY"]
client = genai.Client(api_key=API_KEY)
MY_MODEL = "gemma-4-31b-it"

st.set_page_config(page_title="AI 專家偵探 3.2", layout="wide")

# 圖示轉換
def icon_map(val):
    if val == 1: return "🔴"
    if val == -1: return "🟢"
    return "⚪"

tab1, tab2 = st.tabs(["🔍 單檔 AI 深度診斷", "📡 9 大指標選股雷達"])

# --- Tab 1: AI 診斷 ---
with tab1:
    st.header(f"🕵️ 專家偵探分析 ({MY_MODEL})")
    symbol_input = st.text_input("輸入股票代號 (例: 2002)", "2002")
    symbol = f"{symbol_input}.TW" if "." not in symbol_input else symbol_input

    if st.button("開始深度分析", use_container_width=True):
        if os.path.exists("daily_scan.csv"):
            df_scan = pd.read_csv("daily_scan.csv")
            stock_data = df_scan[df_scan['代號'] == symbol].tail(1)
            
            if not stock_data.empty:
                row = stock_data.iloc[0].to_dict()
                with st.spinner("偵探正在拼湊線索..."):
                    try:
                        prompt = f"你是台股偵探專家。請針對數據：{row} 進行深度辯論。必須包含：1.各指標矛盾點分析 2.給小白的白話建議 3.最後給出一個 0-100 的『看多信心分數』。"
                        response = client.models.generate_content(model=MY_MODEL, contents=prompt)
                        st.markdown(response.text)
                    except Exception as e:
                        st.error(f"AI 呼叫失敗: {e}")
            else:
                st.warning("找不到數據，請確認代號正確或已完成掃描。")

# --- Tab 2: 選股雷達 (9 指標全開) ---
with tab2:
    st.header("📡 全市場選股雷達")
    if os.path.exists("daily_scan.csv"):
        df = pd.read_csv("daily_scan.csv")
        
        with st.expander("🛠️ 設定 9 大指標過濾條件", expanded=True):
            opt = {"不限": "All", "看多 (🔴)": 1, "看空 (🟢)": -1}
            # 手機版佈局：三行三列
            c1, c2, c3 = st.columns(3)
            with c1: f_ma = st.selectbox("均線", list(opt.keys()))
            with c2: f_macd = st.selectbox("MACD", list(opt.keys()))
            with c3: f_bb = st.selectbox("布林", list(opt.keys()))
            
            c4, c5, c6 = st.columns(3)
            with c4: f_rsi = st.selectbox("RSI", list(opt.keys()))
            with c5: f_kd = st.selectbox("KD", list(opt.keys()))
            with c6: f_sh = st.selectbox("下影線", list(opt.keys()))
            
            c7, c8, c9 = st.columns(3)
            with c7: f_gp = st.selectbox("缺口", list(opt.keys()))
            with c8: f_in = st.selectbox("法人", list(opt.keys()))
            with c9: f_mg = st.selectbox("融資", list(opt.keys()))

        # 安全過濾邏輯 (避免 KeyError)
        f_df = df.copy()
        filters = {
            '均線': f_ma, 'MACD': f_macd, '布林': f_bb, 'RSI': f_rsi, 
            'KD': f_kd, '下影線': f_sh, '缺口': f_gp, '法人': f_in, '融資': f_mg
        }
        for col, val in filters.items():
            if col in f_df.columns and opt[val] != "All":
                f_df = f_df[f_df[col] == opt[val]]

        st.subheader(f"📊 篩選結果 ({len(f_df)} 檔)")
        # 轉換圖示並顯示
        display_cols = ['代號', '收盤價'] + list(filters.keys())
        # 只顯示現有的欄位，防止報錯
        actual_cols = [c for c in display_cols if c in f_df.columns]
        display_df = f_df[actual_cols].copy()
        for c in filters.keys():
            if c in display_df.columns:
                display_df[c] = display_df[c].apply(icon_map)
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.error("尚未偵測到 daily_scan.csv，請執行掃描腳本。")
