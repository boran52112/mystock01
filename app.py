import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

# --- 1. 初始化與環境設定 ---
st.set_page_config(page_title="台股 AI 偵探系統 v5.0", layout="wide")

# 設定 2026 模擬環境模型與時間
CURRENT_DATE = "2026-04-22"
MODEL_NAME = "models/gemma-4-31b-it"

# --- 2. Google 試算表連線函式 ---
def get_data_from_sheets():
    # 這裡請確保您的憑證檔案名稱正確
    scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    creds = Credentials.from_service_account_file('credentials.json', scopes=scope)
    client = gspread.authorize(creds)
    
    # 開啟試算表 (請填入您的試算表名稱)
    sheet = client.open("台股AI偵探數據庫").sheet1 
    data = sheet.get_all_records()
    return pd.DataFrame(data)

# --- 3. AI 模型調用函式 ---
def call_ai_detective(prompt):
    # 這裡串接您原本的 AI 調用邏輯
    # 假設使用 st.secrets 或直接調用
    try:
        # 模擬呼叫模型 (請替換為您實際的呼叫代碼)
        # response = model.generate_content(prompt)
        # return response.text
        return "（這是模擬回應，請確保您的 AI API 密鑰已設定）"
    except Exception as e:
        return f"AI 偵探回報錯誤: {str(e)}"

# --- 4. 主程式邏輯 ---
def main():
    st.title(f"🕵️‍♂️ 台股 AI 偵探戰情室 (模擬日期: {CURRENT_DATE})")

    # 讀取數據
    try:
        df = get_data_from_sheets()
    except Exception as e:
        st.error(f"無法讀取試算表，請檢查憑證或檔名。錯誤: {e}")
        return

    # 側邊欄：選擇股票
    stock_list = df['Stock_ID'].unique()
    selected_stock = st.sidebar.selectbox("選擇要診斷的股票", stock_list)

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
        
        # 顯示表格
        st.dataframe(display_df, use_container_width=True)

        # --- 第二階段：AI 診斷啟動器 ---
        if st.button(f"🚀 啟動 {selected_stock} AI 深度偵探診斷"):
            with st.spinner("偵探正在判讀指標..."):
                
                # 準備 AI 資料字串
                data_summary = display_df.to_string(index=False)
                
                # 建構強效 Prompt
                prompt = f"""
你現在是「台股 AI 偵探系統 v5.0」，禁止輸出英文前言與廢話。
直接從「第一區塊」開始按照以下格式分析 {selected_stock}：

數據內容：
{data_summary}

第一區塊：【九項指標趨勢深度判讀】
(分析 5/20日線、RSI、KDJ、MACD、布林、量價背離、漲跌連續性、支撐壓力、市場心理)

第二區塊：【指標矛盾整合與風險抓漏】
(分析長短線矛盾或量價陷阱)

第三區塊：【全方位操作戰略：雙重劇本】
- 保守型劇本：進場點、停損位、持股邏輯
- 激進型劇本：突破點、目標價、短線策略

第四區塊：【偵探總結與信心分數】
總結：(一句話)
信心分數：(0-100)
"""
                # 取得 AI 回應
                raw_response = call_ai_detective(prompt)
                
                # 過濾廢話 (只保留「第一區塊」之後的內容)
                if "第一區塊" in raw_response:
                    clean_response = raw_response[raw_response.find("第一區塊"):]
                else:
                    clean_response = raw_response

                st.markdown("---")
                st.markdown(clean_response)

# 執行主程式
if __name__ == "__main__":
    main()
