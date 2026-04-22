import streamlit as st
import pandas as pd
import gspread
import json
import google.generativeai as genai
from google.oauth2.service_account import Credentials

# --- 1. 戰情室基本設定 ---
st.set_page_config(page_title="台股 AI 偵探系統 v5.0", layout="wide")
CURRENT_DATE = "2026-04-22"
SHEET_ID = "1UH-fwxENhGUDmQjTQJq72g1DzsxML_CPIJGg39cWAgM"

# --- 2. 核心數據讀取函式 ---
def get_data_from_sheets():
    try:
        creds_json_str = st.secrets["gcp_service_account_raw"]
        creds_info = json.loads(creds_json_str)
        scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        creds = Credentials.from_service_account_info(creds_info, scopes=scope)
        client = gspread.authorize(creds)
        
        # 精確鎖定 ID
        sheet = client.open_by_key(SHEET_ID).sheet1 
        data = sheet.get_all_records()
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"❌ 數據庫連線失敗: {e}")
        return None

# --- 3. AI 偵探診斷函式 ---
def call_ai_detective(prompt):
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        # 使用 2026 指定模型
        model = genai.GenerativeModel('models/gemma-4-31b-it') 
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        # 如果模型名稱在測試環境不支援，自動切換回相容模式
        try:
            model = genai.GenerativeModel('gemini-1.5-pro')
            response = model.generate_content(prompt)
            return response.text
        except:
            return f"偵探回報錯誤: {str(e)}"

# --- 4. 主程式介面 ---
def main():
    st.title(f"🕵️‍♂️ 台股 AI 偵探戰情室 (模擬日期: {CURRENT_DATE})")

    df = get_data_from_sheets()
    if df is None or df.empty:
        st.warning("目前數據庫內無數據。")
        return

    if 'Stock_ID' in df.columns:
        stock_list = sorted(df['Stock_ID'].unique())
        selected_stock = st.sidebar.selectbox("🎯 選擇目標股票", stock_list)
    else:
        st.error("找不到 Stock_ID 欄位")
        return

    if selected_stock:
        # 過濾該股最近 5 筆數據
        df_selected = df[df['Stock_ID'] == selected_stock].tail(5)

        # --- 數據表格區 ---
        st.subheader(f"📊 {selected_stock} 數據觀測站")
        
        column_mapping = {
            'Date': '日期',
            'Close': '收盤價',
            'Volume': '成交量',
            'RSI': 'RSI',
            'KDJ_K': 'K值',
            'KDJ_D': 'D值',
            'MACD': 'MACD',
            'BB_Upper': '布林上軌',
            'BB_Lower': '布林下軌',
            'Daily_Return': '漲跌幅%'
        }
        
        existing_cols = [c for c in column_mapping.keys() if c in df_selected.columns]
        display_df = df_selected[existing_cols].rename(columns=column_mapping)
        
        st.dataframe(display_df, use_container_width=True)

        # --- AI 診斷按鈕 ---
        if st.button(f"🚀 啟動 {selected_stock} AI 深度診斷"):
            with st.spinner("AI 偵探正在分析..."):
                data_summary = display_df.to_string(index=False)
                
                prompt = f"""
你現在是「台股 AI 偵探系統 v5.0」。
請針對 {selected_stock} 數據進行診斷，嚴格遵守格式，禁止英文。

數據：
{data_summary}

第一區塊：【九項指標趨勢深度判讀】
第二區塊：【指標矛盾整合與風險抓漏】
第三區塊：【全方位操作戰略：雙重劇本】
第四區塊：【偵探總結與信心分數】
"""
                raw_response = call_ai_detective(prompt)
                
                # 簡單過濾
                target = "第一區塊"
                clean_response = raw_response[raw_response.find(target):] if target in raw_response else raw_response

                st.markdown("---")
                st.info(clean_response)

if __name__ == "__main__":
    main()
