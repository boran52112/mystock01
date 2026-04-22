import streamlit as st
import pandas as pd
import gspread
import json
import google.generativeai as genai
from google.oauth2.service_account import Credentials

# --- 1. 初始化與環境設定 ---
st.set_page_config(page_title="台股 AI 偵探系統 v5.0", layout="wide")

# 設定 2026 模擬環境時間
CURRENT_DATE = "2026-04-22"

# --- 2. Google 試算表連線函式 (對應您的 Secrets 名稱) ---
def get_data_from_sheets():
    # 從 Secrets 讀取原始 JSON 字串
    creds_json_str = st.secrets["gcp_service_account_raw"]
    # 將字串轉換為 Python 字典
    creds_info = json.loads(creds_json_str)
    
    scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    creds = Credentials.from_service_account_info(creds_info, scopes=scope)
    client = gspread.authorize(creds)
    
    # 【注意】請確保您的試算表名稱正確，若改名請修改這裡
    sheet = client.open("台股AI偵探數據庫").sheet1 
    data = sheet.get_all_records()
    return pd.DataFrame(data)

# --- 3. AI 模型調用函式 ---
def call_ai_detective(prompt):
    # 設定 API Key
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    # 使用 2026 環境指定的模型
    model = genai.GenerativeModel('gemini-1.5-pro') # 這裡建議先用穩定版，若環境有特定模型再調整
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 偵探回報錯誤: {str(e)}"

# --- 4. 主程式邏輯 ---
def main():
    st.title(f"🕵️‍♂️ 台股 AI 偵探戰情室 (模擬日期: {CURRENT_DATE})")

    # 讀取數據
    try:
        df = get_data_from_sheets()
    except Exception as e:
        st.error(f"無法讀取試算表。請檢查：1. Secrets 名稱是否正確 2. 試算表是否已分享給 Service Account Email。錯誤回報: {e}")
        return

    # 側邊欄：選擇股票
    if 'Stock_ID' in df.columns:
        stock_list = df['Stock_ID'].unique()
        selected_stock = st.sidebar.selectbox("選擇要診斷的股票", stock_list)
    else:
        st.error("試算表中找不到 'Stock_ID' 欄位，請檢查數據格式。")
        return

    if selected_stock:
        # 過濾該股最近 5 天數據
        df_selected = df[df['Stock_ID'] == selected_stock].tail(5)

        # --- 第一階段：顯示中文化表格 (方案 A) ---
        st.subheader(f"🔍 {selected_stock} 數據戰情盤後 (5日歷史趨勢)")
        
        column_mapping = {
            'Date': '日期',
            'Close': '收盤價',
            'Volume': '成交量',
            'RSI': 'RSI強弱',
            'KDJ_K': 'K值',
            'KDJ_D': 'D值',
            'MACD': 'MACD動能',
            'BB_Upper': '布林上軌',
            'BB_Lower': '布林下軌',
            'Daily_Return': '漲跌幅%'
        }
        
        # 只取現有的欄位進行映射
        existing_cols = [col for col in column_mapping.keys() if col in df_selected.columns]
        display_df = df_selected[existing_cols].rename(columns=column_mapping)
        
        # 顯示美化表格
        st.dataframe(display_df.style.format(subset=['收盤價', '漲跌幅%'], formatter="{:.2f}"), use_container_width=True)

        # --- 第二階段：AI 診斷啟動器 ---
        if st.button(f"🚀 啟動 {selected_stock} AI 深度偵探診斷"):
            with st.spinner("偵探正在判讀指標..."):
                
                # 準備 AI 資料字串
                data_summary = display_df.to_string(index=False)
                
                # 建構強效 Prompt (按照指揮官要求的四個區塊)
                prompt = f"""
你現在是「台股 AI 偵探系統 v5.0」，禁止輸出英文、禁止前言與廢話。
請針對 {selected_stock} 過去 5 日的數據進行診斷，嚴格遵守以下格式：

數據內容：
{data_summary}

第一區塊：【九項指標趨勢深度判讀】
(針對 5/20日線、RSI、KDJ、MACD、布林、量價背離、漲跌連續性、支撐壓力、市場心理進行條列式判讀)

第二區塊：【指標矛盾整合與風險抓漏】
(分析數據中的矛盾點，例如價量關係、多空對峙狀況)

第三區塊：【全方位操作戰略：雙重劇本】
- 保守型劇本：(進場點、停損位、持股邏輯)
- 激進型劇本：(突破點、目標價、短線策略)

第四區塊：【偵探總結與信心分數】
總結：(一句話精闢總結)
信心分數：(0-100)
"""
                # 取得 AI 回應
                raw_response = call_ai_detective(prompt)
                
                # 廢話過濾邏輯
                target_start = "第一區塊"
                if target_start in raw_response:
                    clean_response = raw_response[raw_response.find(target_start):]
                else:
                    clean_response = raw_response

                st.markdown("---")
                st.markdown(clean_response)

# 執行主程式
if __name__ == "__main__":
    main()
