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
        df = df.drop_duplicates(subset=['日期', '股號'], keep='last')
        target = df[df['股號'].astype(str).str.contains(symbol)].sort_values('日期').tail(5)
        if not target.empty: return target, "5日趨勢資料庫"
    except: pass
    return None, None

st.title("🕵️‍♂️ 台股 AI 偵探戰情室")
stock_id = st.text_input("輸入目標股號 (如 2330):", "").strip()

if stock_id:
    with st.spinner("偵探正在翻譯數據並產出中文報告..."):
        data, source = get_clean_data(stock_id)
        if data is not None:
            st.success(f"📊 5日證據鎖定成功")
            # 讓表格數字好看一點
            st.table(data[['日期', '收盤價', 'MA5', 'RSI14', '漲跌幅%']])
            
            evidence_md = data.to_markdown(index=False)
            
            # 強化版 Prompt：將英文摘要強制轉中文
            prompt = f"""
            [系統令：嚴格禁止使用英文。所有技術分析、數據標題必須轉換為繁體中文。]
            
            你是一位派駐在 2026 年的台股資深分析偵探。請針對以下證據表格產出【全繁體中文】報告。

            【證據表格】：
            {evidence_md}

            請嚴格依照以下結構輸出報告，不要輸出任何前言或後記：

            📌 偵探報告編號：2026-{stock_id}-SCAN
            🕵️ 偵探身份：2026年派駐資深分析偵探

            ### 🔍 證據深度拆解 (數據摘要)
            (請將表格中的價格變動、均線走向、RSI14、KD、MACD、布林通道與成交量的具體數據變化，以「繁體中文」條列式逐一說明。)

            ### 第一區塊：【技術指標與趨勢判讀】
            (在此分析這5天的情緒演變，目前的市場心理是恐慌還是興奮？)

            ### 第二區塊：【指標矛盾整合與風險抓漏】
            (檢查是否有指標背離、誘多或誘空陷阱？)

            ### 第三區塊：【全方位操作戰略：雙重劇本】
            1. 保守型劇本：(提供穩健進場點、停損建議)
            2. 激進型劇本：(提供突破追擊點、目標價建議)

            ### 第四區塊：【偵探總結與信心分數】
            * 總結：
            * 信心分數：(0-100)
            """
            
            try:
                # 呼叫 2026 年 Gemma 4 模型
                model = genai.GenerativeModel('models/gemma-4-31b-it')
                response = model.generate_content(prompt)
                st.markdown("---")
                st.markdown("### 📝 AI 偵探深度診斷報告")
                st.write(response.text)
            except:
                st.error("AI 偵探目前連線繁忙，請稍後再試。")
        else:
            st.warning("查無 5 日資料庫紀錄。")

st.markdown("---")
st.caption("AI 偵探系統 v8.1 | 數據校準日期：2026-04-22")
