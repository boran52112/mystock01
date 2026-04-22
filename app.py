import streamlit as st
import pandas as pd
import gspread
import json
import google.generativeai as genai
from google.oauth2.service_account import Credentials

# --- 1. 戰情室基本設定 ---
st.set_page_config(page_title="台股 AI 偵探系統 v5.0", layout="wide")
CURRENT_DATE = "2026-04-22"
# 您的試算表 ID
SHEET_ID = "1UH-fwxENhGUDmQjTQJq72g1DzsxML_CPIJGg39cWAgM"

# --- 2. 核心數據讀取函式 ---
def get_data_from_sheets():
    try:
        # 從 Secrets 讀取 JSON 並轉換
        creds_json_str = st.secrets["gcp_service_account_raw"]
        creds_info = json.loads(creds_json_str)
        scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        creds = Credentials.from_service_account_info(creds_info, scopes=scope)
        client = gspread.authorize(creds)
        
        # 使用 ID 開啟
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
        # 嘗試使用 2026 指定模型，若失敗則切換至 pro 版
        try:
            model = genai.GenerativeModel('models/gemma-4-31b-it') 
            response = model.generate_content(prompt)
        except:
            model = genai.GenerativeModel('gemini-1.5-pro')
            response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"偵探回報錯誤: {str(e)}"

# --- 4. 主程式介面 ---
def main():
    st.title(f"🕵️‍♂️ 台股 AI 偵探戰情室 (模擬日期: {CURRENT_DATE})")

    # 讀取數據
    df = get_data_from_sheets()
    
    if df is None or df.empty:
        st.warning("目前數據庫內無數據。")
        return

    # 【修正點】改為匹配您試算表中的中文欄位「股號」
    if '股號' in df.columns:
        stock_list = sorted(df['股號'].unique())
        selected_stock = st.sidebar.selectbox("🎯 選擇目標股票", stock_list)
    else:
        st.error("❌ 找不到『股號』欄位。請確認試算表第一列是否有『股號』二字。")
        # 顯示目前的欄位名稱供指揮官除錯
        st.write("目前偵測到的欄位有：", list(df.columns))
        return

    if selected_stock:
        # 篩選該股最近 5 筆數據
        df_selected = df[df['股號'] == selected_stock].tail(5)

        # --- 第一階段：數據表格展示 ---
        st.subheader(f"📊 {selected_stock} 數據觀測站 (最近 5 日)")
        
        # 【修正點】根據您的截圖，定義要顯示的中文欄位
        target_columns = ['日期', '股號', '股名', '收盤價', '成交量', 'MA5', 'MA20', 'RSI14']
        
        # 自動過濾掉不存在的欄位，防止程式崩潰
        existing_cols = [c for c in target_columns if c in df_selected.columns]
        display_df = df_selected[existing_cols]
        
        # 顯示美化表格 (移除索引)
        st.table(display_df)

        # --- 第二階段：強制 AI 四區塊分析 ---
        if st.button(f"🚀 啟動 {selected_stock} AI 深度診斷"):
            with st.spinner("AI 偵探正在分析指標..."):
                
                # 數據文字化
                data_summary = display_df.to_string(index=False)
                
                prompt = f"""
你現在是「台股 AI 偵探系統 v5.0」，禁止輸出英文、禁止前言。
直接從「第一區塊」開始按照以下格式診斷 {selected_stock}：

數據內容：
{data_summary}

第一區塊：【九項指標趨勢深度判讀】
(針對 5/20日線、RSI、KDJ、MACD、布林、量價背離、漲跌連續性、支撐壓力、市場心理進行條列式判讀)

第二區塊：【指標矛盾整合與風險抓漏】
(分析長短線矛盾或量價陷阱)

第三區塊：【全方位操作戰略：雙重劇本】
- 保守型劇本：進場點、停損位、持股邏輯
- 激進型劇本：突破點、目標價、短線策略

第四區塊：【偵探總結與信心分數】
總結：(一句話精闢評語)
信心分數：(0-100)
"""
                raw_response = call_ai_detective(prompt)
                
                # 廢話切除邏輯
                target = "第一區塊"
                if target in raw_response:
                    clean_response = raw_response[raw_response.find(target):]
                else:
                    clean_response = raw_response

                st.markdown("---")
                st.markdown(f"### 🛡️ 偵探診斷報告：{selected_stock}")
                st.info(clean_response)

if __name__ == "__main__":
    main()
