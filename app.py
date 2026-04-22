import streamlit as st
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import gspread
from google.oauth2.service_account import Credentials
import google.generativeai as genai
import json

st.set_page_config(page_title="台股 AI 偵探戰情室", layout="centered")

# 金鑰初始化
try:
    gcp_info = json.loads(st.secrets["gcp_service_account_raw"])
    gcp_info['private_key'] = gcp_info['private_key'].replace('\\n', '\n')
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except:
    st.error("金鑰設定錯誤")
    st.stop()

def get_clean_data(symbol):
    symbol = symbol.upper().strip()
    try:
        scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        creds = Credentials.from_service_account_info(gcp_info, scopes=scope)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key("1UH-fwxENhGUDmQjTQJq72g1DzsxML_CPIJGg39cWAgM")
        wks = sh.get_worksheet(0)
        df = pd.DataFrame(wks.get_all_records())
        # 去重，只留最後一筆
        df = df.drop_duplicates(subset=['日期', '股號'], keep='last')
        target = df[df['股號'].astype(str).str.contains(symbol)].sort_values('日期').tail(5)
        if not target.empty: return target, "5日趨勢資料庫"
    except: pass
    return None, None

st.title("🕵️‍♂️ 台股 AI 偵探戰情室")
stock_id = st.text_input("輸入目標股號 (如 2330):", "").strip()

if stock_id:
    with st.spinner("偵探正在依照九項指標進行深度辦案..."):
        data, source = get_clean_data(stock_id)
        if data is not None:
            st.success(f"📊 5日證據鎖定成功")
            # 格式化表格
            st.table(data[['日期', '收盤價', 'MA5', 'RSI14', '漲跌幅%']])
            
            evidence_md = data.to_markdown(index=False)
            
            # 【核心修正】: 鎖死九項指標分析，禁止任何英文前言
            prompt = f"""
            [系統絕對指令：禁止輸出任何英文，禁止解釋指令，禁止顯示分析過程。請直接輸出繁體中文報告內容。]
            
            你是一位派駐在 2026 年的台股資深分析偵探。請針對以下證據產出深度診斷報告。

            【5日歷史證據表格】：
            {evidence_md}

            請嚴格依照以下格式輸出報告：

            📌 偵探報告編號：2026-{stock_id}-SCAN
            🕵️ 偵探身份：2026年派駐資深分析偵探

            ### 第一區塊：【九項指標趨勢深度判讀】
            你必須針對以下九項指標進行逐一分析：
            1. 週線與月線(MA5/MA20)走向。
            2. RSI強弱指標狀態。
            3. KD隨機指標轉折。
            4. MACD趨勢動能。
            5. 布林通道相對位置。
            6. 成交量與價格之背離關係。
            7. 近5日漲跌幅%之連續性。
            8. 關鍵支撐與壓力位判斷。
            9. 市場心理分析（恐慌、亢奮或觀望）。

            ### 第二區塊：【指標矛盾整合與風險抓漏】
            (分析各指標間是否有衝突，是否存在誘多或誘空陷阱。)

            ### 第三區塊：【全方位操作戰略：雙重劇本】
            1. 保守型劇本：(進場點、停損位、持股邏輯)
            2. 激進型劇本：(突破點、目標價、短線策略)

            ### 第四區塊：【偵探總結與信心分數】
            * 總結：(一句話定調)
            * 信心分數：(0-100)
            """
            
            try:
                model = genai.GenerativeModel('models/gemma-4-31b-it')
                response = model.generate_content(prompt)
                st.markdown("---")
                st.markdown("### 📝 AI 偵探深度診斷報告")
                # 再次確保過濾掉可能的 AI 碎碎念
                report = response.text
                if "Persona" in report or "Constraint" in report:
                    st.warning("偵探報告含有過多系統訊息，請重新嘗試。")
                st.write(report)
            except:
                st.error("AI 偵探目前無法連線。")
        else:
            st.warning("資料庫尚無此股之5日完整數據。")

st.markdown("---")
st.caption("AI 偵探系統 v8.2 | 強制九項指標診斷版")
