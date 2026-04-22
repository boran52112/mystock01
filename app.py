import streamlit as st
import pandas as pd
# 假設您已有的 Google Sheets 讀取邏輯與模型調用邏輯已存在
# 這裡重點展示「表格中文化」與「四區塊診斷控制」的優化代碼

def run_v85_optimization(df_selected, stock_id):
    """
    df_selected: 從試算表抓取出的該股 5 日數據 DataFrame
    stock_id: 股票代碼 (例如 "2330")
    """
    
    # --- 第一階段：數據表格化 (方案 A) ---
    st.subheader(f"🔍 {stock_id} 數據戰情盤後 (5日歷史趨勢)")
    
    # 1. 建立中文映射表 (確保完全無英文)
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
    
    # 2. 轉換 DataFrame 欄位並過濾
    display_df = df_selected[list(column_mapping.keys())].rename(columns=column_mapping)
    
    # 3. 呈現表格 (Streamlit 原生組件)
    st.dataframe(display_df.style.highlight_max(axis=0, color='#2ecc71').highlight_min(axis=0, color='#e74c3c'), use_container_width=True)

    # --- 第二階段：準備餵給 AI 的資料 (字串化) ---
    # 將 5 日數據轉為簡潔字串，讓 AI 讀取但不直接顯示
    data_summary = display_df.to_string(index=False)

    # --- 第三階段：建構「四區塊」強制指令 (Prompt) ---
    prompt = f"""
你現在是「台股 AI 偵探系統 v5.0」，專精於技術指標與市場心理學。
當前模擬時間：2026-04-22。

【嚴格規範】
1. 禁止輸出任何英文前言（如 Sure, here is...）。
2. 禁止輸出結尾語。
3. 必須且只能輸出以下四個診斷區塊。
4. 語言：繁體中文。

以下是 {stock_id} 過去 5 日的技術指標數據：
{data_summary}

請根據數據執行深度診斷，格式如下：

第一區塊：【九項指標趨勢深度判讀】
(請針對 5/20日線、RSI、KDJ、MACD、布林、量價背離、漲跌連續性、支撐壓力、市場心理進行條列式判讀)

第二區塊：【指標矛盾整合與風險抓漏】
(請找出數據中互相衝突的地方，例如價格漲但量能縮，並分析是否為陷阱)

第三區塊：【全方位操作戰略：雙重劇本】
- 保守型劇本：(進場點、停損位、持股邏輯)
- 激進型劇本：(突破點、目標價、短線策略)

第四區塊：【偵探總結與信心分數】
總結：(一句話精闢點評)
信心分數：(0-100)
"""

    # --- 第四階段：呼叫 AI 並執行「廢話過濾」 ---
    if st.button(f"🚀 啟動 {stock_id} AI 深度偵探診斷"):
        with st.spinner("偵探正在分析指標細節..."):
            # 這裡調用您模擬環境的 gemma-4-31b-it
            raw_response = call_gemma_model(prompt) 
            
            # 廢話過濾邏輯：強行切除第一個區塊標題之前的任何內容
            target_start = "第一區塊"
            if target_start in raw_response:
                clean_response = raw_response[raw_response.find(target_start):]
            else:
                clean_response = raw_response

            # 最終呈現
            st.markdown("---")
            st.markdown(clean_response)

# 註：call_gemma_model 為您專案中既有的 API 呼叫函數
