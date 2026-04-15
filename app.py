import streamlit as st
import pandas as pd
import os
from google import genai

# 1. AI 客戶端初始化
API_KEY = st.secrets["GOOGLE_API_KEY"]
client = genai.Client(api_key=API_KEY)
MY_MODEL = "gemma-4-31b-it"

st.set_page_config(page_title="AI 專家偵探 3.6", layout="wide")

def get_status_ui(val):
    if val == 1: return "🔴 看多"
    if val == -1: return "🟢 看空"
    return "⚪ 中立"

@st.cache_data
def load_stock_names():
    for f in ["股票清單.csv", "stock_list.csv"]:
        if os.path.exists(f):
            df_n = pd.read_csv(f)
            return dict(zip(df_n['stock_id'].astype(str), df_n['stock_name']))
    return {}

stock_names = load_stock_names()
tab1, tab2 = st.tabs(["🔍 專家偵探深度診斷", "📡 全市場選股雷達"])

# --- Tab 1: AI 診斷 ---
with tab1:
    st.header(f"🕵️ 偵探結案報告")
    symbol_input = st.text_input("請輸入股票代號 (例: 2002)", "2002")
    pure_id = symbol_input.split('.')[0]
    stock_id = f"{pure_id}.TW" if "." not in symbol_input else symbol_input
    name = stock_names.get(pure_id, "未知企業")

    if st.button("開始深度偵探分析", use_container_width=True):
        if os.path.exists("daily_scan.csv"):
            df_scan = pd.read_csv("daily_scan.csv")
            stock_data = df_scan[df_scan['代號'] == stock_id].tail(1)
            
            if not stock_data.empty:
                row = stock_data.iloc[0].to_dict()
                st.metric("公司名稱", f"{pure_id} {name}", f"收盤: {row.get('收盤價', 'N/A')} 元")
                
                with st.spinner("AI 偵探教學整合中..."):
                    try:
                        prompt = f"你是台股偵探專家。針對股票 {pure_id}({name})，數據如下：{row}。請撰寫報告：1.九項指標分析(解釋小白意義) 2.矛盾整合教學 3.信心分數(0-100) 4.針對持股與否的操作建議。請用白話繁體中文。"
                        response = client.models.generate_content(model=MY_MODEL, contents=prompt)
                        st.markdown(response.text)
                    except Exception as e:
                        st.error(f"AI 診斷失敗: {e}")
            else:
                st.warning(f"資料庫暫無 {stock_id} 數據。")

# --- Tab 2: 選股雷達 (防彈版) ---
with tab2:
    st.header("📡 全市場選股雷達")
    if os.path.exists("daily_scan.csv"):
        df = pd.read_csv("daily_scan.csv")
        df = df.sort_values('日期').drop_duplicates('代號', keep='last')
        
        with st.expander("🛠️ 設定『多空偵探濾網』", expanded=True):
            opt = {"不限": "All", "看多 (🔴)": 1, "看空 (🟢)": -1}
            cols = st.columns(3)
            f_ma = cols[0].selectbox("均線趨勢", list(opt.keys()))
            f_macd = cols[1].selectbox("MACD 動能", list(opt.keys()))
            f_kd = cols[2].selectbox("KD 隨機指標", list(opt.keys()))
            
            cols2 = st.columns(3)
            f_rsi = cols2[0].selectbox("RSI 強弱", list(opt.keys()))
            f_inst = cols2[1].selectbox("法人籌碼", list(opt.keys()))
            f_mg = cols2[2].selectbox("融資籌碼", list(opt.keys()))

        # 動態過濾：只針對「CSV 裡有的欄位」進行過濾
        f_df = df.copy()
        check_list = {'均線': f_ma, 'MACD': f_macd, 'KD': f_kd, 'RSI': f_rsi, '法人': f_inst, '融資': f_mg}
        for col, val in check_list.items():
            if col in f_df.columns and opt[val] != "All":
                f_df = f_df[f_df[col] == opt[val]]

        st.subheader(f"📊 篩選結果 ({len(f_df)} 檔)")
        
        # 🎓 防彈顯示邏輯：只抓 CSV 裡真的有的欄位來顯示
        f_df['名稱'] = f_df['代號'].apply(lambda x: stock_names.get(x.split('.')[0], "未知"))
        base_cols = ['日期', '代號', '名稱', '收盤價']
        indicator_cols = [c for c in ['均線', 'MACD', 'KD', 'RSI', '法人', '融資', '布林', '下影線', '缺口'] if c in f_df.columns]
        
        display_df = f_df[base_cols + indicator_cols].copy()
        for c in indicator_cols:
            display_df[c] = display_df[c].apply(get_status_ui)
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.error("尚未偵測到掃描資料。")
