import streamlit as st
import pandas as pd
import os
from google import genai

# 1. AI 客戶端初始化
API_KEY = st.secrets["GOOGLE_API_KEY"]
client = genai.Client(api_key=API_KEY)
MY_MODEL = "gemma-4-31b-it"

st.set_page_config(page_title="AI 專家偵探 3.5", layout="wide")

# 圖示與狀態轉換 (讓介面不出現 1, 0, -1)
def get_status_ui(val):
    if val == 1: return "🔴 看多"
    if val == -1: return "🟢 看空"
    return "⚪ 中立"

# 載入股票名稱點名簿
@st.cache_data
def load_stock_names():
    for f in ["股票清單.csv", "stock_list.csv"]:
        if os.path.exists(f):
            df_n = pd.read_csv(f)
            return dict(zip(df_n['stock_id'].astype(str), df_n['stock_name']))
    return {}

stock_names = load_stock_names()

tab1, tab2 = st.tabs(["🔍 專家偵探深度診斷", "📡 全市場選股雷達"])

# --- Tab 1: 單檔 AI 深度診斷 (教學策略版) ---
with tab1:
    st.header(f"🕵️ 偵探結案報告 ({MY_MODEL})")
    symbol_input = st.text_input("請輸入股票代號 (例: 2002)", "2002")
    
    # 處理代號
    pure_id = symbol_input.split('.')[0]
    stock_id = f"{pure_id}.TW" if "." not in symbol_input else symbol_input
    name = stock_names.get(pure_id, "未知企業")

    if st.button("開始深度偵探分析", use_container_width=True):
        if os.path.exists("daily_scan.csv"):
            df_scan = pd.read_csv(df_scan := "daily_scan.csv")
            stock_data = df_scan[df_scan['代號'] == stock_id].tail(1)
            
            if not stock_data.empty:
                row = stock_data.iloc[0].to_dict()
                
                # 呈現基本資料區
                c1, c2, c3 = st.columns(3)
                c1.metric("公司名稱", f"{pure_id} {name}")
                c2.metric("最新收盤", f"{row['收盤價']} 元")
                c3.metric("資料日期", row['日期'])

                with st.spinner("偵探正在進行邏輯推理與教學整合..."):
                    try:
                        # 教學式 Prompt
                        prompt = f"""
                        你是一位台股偵探專家。針對股票 {pure_id}({name})，數據如下：{row}。
                        請撰寫一份深度報告，內容必須包含：
                        1. 【九項指標分析】：逐一簡述各指標目前訊號對小白的意義。若數值為 0 請解釋為數據暫缺或中立。
                        2. 【矛盾分析整合】：找出指標間互相支持或互相矛盾之處，並教導小白如何判斷。
                        3. 【具體策略建議】：給予信心分數(0-100)，並針對『已持有』與『未持有』的人給予具體買賣價位與操作建議。
                        請使用專業但白話的繁體中文。
                        """
                        response = client.models.generate_content(model=MY_MODEL, contents=prompt)
                        st.markdown("---")
                        st.markdown(response.text)
                    except Exception as e:
                        st.error(f"AI 診斷失敗: {e}")
            else:
                st.warning(f"在資料庫中找不到 {stock_id}。請確認代號正確。")

# --- Tab 2: 選股雷達 (去除數字、中文名稱版) ---
with tab2:
    st.header("📡 全市場選股雷達")
    if os.path.exists("daily_scan.csv"):
        df = pd.read_csv("daily_scan.csv")
        # 只取每一檔股票最新的一筆資料，避免重複顯示
        df = df.sort_values('日期').drop_duplicates('代號', keep='last')
        
        with st.expander("🛠️ 點我設定『多空偵探濾網』", expanded=True):
            opt = {"不限": "All", "看多 (🔴)": 1, "看空 (🟢)": -1}
            c = st.columns(3)
            f_ma = c[0].selectbox("均線趨勢", list(opt.keys()))
            f_macd = c[1].selectbox("MACD 動能", list(opt.keys()))
            f_kd = c[2].selectbox("KD 隨機指標", list(opt.keys()))
            
            c2 = st.columns(3)
            f_rsi = c2[0].selectbox("RSI 強弱", list(opt.keys()))
            f_inst = c2[1].selectbox("法人籌碼(目前建置中)", list(opt.keys()))
            f_mg = c2[2].selectbox("融資籌碼(目前建置中)", list(opt.keys()))

        # 過濾邏輯
        f_df = df.copy()
        for col, val in zip(['均線', 'MACD', 'KD', 'RSI', '法人', '融資'], [f_ma, f_macd, f_kd, f_rsi, f_inst, f_mg]):
            if col in f_df.columns and opt[val] != "All":
                f_df = f_df[f_df[col] == opt[val]]

        st.subheader(f"📊 篩選結果：符合條件共 {len(f_df)} 檔")
        
        # 整理顯示用表格
        f_df['名稱'] = f_df['代號'].apply(lambda x: stock_names.get(x.split('.')[0], "未知"))
        
        # 轉換狀態為圖示文字，不再出現 1, 0, -1
        display_df = f_df[['日期', '代號', '名稱', '收盤價', '均線', 'MACD', 'KD', '法人']].copy()
        for col in ['均線', 'MACD', 'KD', '法人']:
            display_df[col] = display_df[col].apply(get_status_ui)
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.error("尚未偵測到掃描資料。")
