import streamlit as st
import pandas as pd
import gspread
import json
import google.generativeai as genai
from google.oauth2.service_account import Credentials
import yfinance as yf
import re

# --- 1. 視覺優化 ---
st.set_page_config(page_title="台股 AI 偵探系統 v5.0", layout="wide")
st.markdown("""
    <style>
    .stMarkdown p, .stMarkdown li { font-size: 1.25rem !important; line-height: 1.7; }
    .report-card { 
        background-color: #ffffff; padding: 20px; border-radius: 12px; 
        border: 1px solid #e0e0e0; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 25px;
    }
    .indicator-label { font-weight: bold; color: #1e3a8a; font-size: 1.3rem; }
    </style>
    """, unsafe_allow_html=True)

if 'stock_data' not in st.session_state: st.session_state.stock_data = None
if 'current_id' not in st.session_state: st.session_state.current_id = ""

SHEET_ID = "1UH-fwxENhGUDmQjTQJq72g1DzsxML_CPIJGg39cWAgM"

# --- 2. 數據核心 ---
def get_data_from_sheets():
    try:
        creds_json_str = st.secrets["gcp_service_account_raw"]
        creds_info = json.loads(creds_json_str)
        scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        creds = Credentials.from_service_account_info(creds_info, scopes=scope)
        client = gspread.authorize(creds)
        sheet = client.open_by_key(SHEET_ID).sheet1 
        return pd.DataFrame(sheet.get_all_records())
    except: return pd.DataFrame()

def fetch_smart_yf(stock_id):
    suffixes = ["", ".TW", ".TWO"]
    for suf in suffixes:
        test_id = f"{stock_id}{suf}"
        try:
            df_yf = yf.download(test_id, period="2mo", progress=False)
            if not df_yf.empty and len(df_yf) > 20:
                df_yf['MA5'] = df_yf['Close'].rolling(window=5).mean()
                df_yf['MA20'] = df_yf['Close'].rolling(window=20).mean()
                delta = df_yf['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df_yf['RSI14'] = 100 - (100 / (1 + rs))
                df_yf = df_yf.reset_index().tail(5)
                return pd.DataFrame({
                    '日期': df_yf['Date'].dt.strftime('%Y-%m-%d'),
                    '股號': test_id, '股名': "即時數據",
                    '收盤價': df_yf['Close'].round(2), '成交量': df_yf['Volume'],
                    'MA5': df_yf['MA5'].round(2), 'MA20': df_yf['MA20'].round(2), 'RSI14': df_yf['RSI14'].round(2)
                })
        except: continue
    return pd.DataFrame()

# --- 3. 強化版：內容清洗器 ---
def clean_ai_content(text):
    # 移除 AI 內心獨白的特徵行 (例如以 * 或 - 開頭的英文行)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # 如果這一行包含超過 3 個英文單字，且有 "Wait" "prompt" "Labels" 等字眼，就剔除
        if re.search(r'(Wait|prompt|label|verify|check|English)', line, re.IGNORECASE) and len(re.findall(r'[a-zA-Z]+', line)) > 3:
            continue
        # 移除純英文字元佔比過高的行 (排除掉數據行)
        if len(line) > 10 and len(re.findall(r'[a-zA-Z]', line)) / len(line) > 0.8:
            continue
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines).strip()

def extract_best_block(text, block_num):
    pattern = rf"\[區塊{block_num}\](.*?)\[/區塊{block_num}\]"
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches: return ""
    # 取最後一個標籤內容並清洗雜訊
    raw_content = matches[-1].strip()
    return clean_ai_content(raw_content)

def call_ai_detective(prompt):
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        model = genai.GenerativeModel('gemma-4-31b-it')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e: return f"偵探連線錯誤: {str(e)}"

# --- 4. 主程式 ---
def main():
    st.title("🕵️‍♂️ 台股 AI 偵探戰情室")
    
    user_input = st.text_input("🔢 請輸入股號", placeholder="例如: 2330").strip()
    if st.button("🔍 開始偵查") and user_input:
        with st.spinner("調閱檔案中..."):
            df_all = get_data_from_sheets()
            df_selected = pd.DataFrame()
            if not df_all.empty and '股號' in df_all.columns:
                df_all['股號'] = df_all['股號'].astype(str)
                df_selected = df_all[df_all['股號'].str.contains(user_input)].tail(5)
            if df_selected.empty: df_selected = fetch_smart_yf(user_input)
            
            if not df_selected.empty:
                st.session_state.stock_data = df_selected
                st.session_state.current_id = df_selected['股號'].iloc[-1]
            else: st.error("查無資料")

    if st.session_state.stock_data is not None:
        df_display = st.session_state.stock_data
        current_id = st.session_state.current_id
        st.subheader(f"📊 {current_id} 指標觀測站")
        st.table(df_display[['日期', '收盤價', '成交量', 'MA5', 'MA20', 'RSI14']])

        if st.button(f"🚀 啟動 {current_id} AI 深度診斷"):
            with st.spinner("偵探正在排版專業報告..."):
                data_text = df_display.to_string(index=False)
                
                # 改用「範例導向」的 Prompt，減少禁令，增加模仿對象
                prompt = f"""你是台股 AI 偵探，請依照針對每一項內容對數據進行繁體中文診斷，並給予教學說明為何這樣判斷。
禁止輸出英文思考過程。請直接將內容填入標籤。

範例格式：
[區塊1]
1. 5日均線：走向平穩...
2. 20日均線：持續上揚...
(以此類推九項)
[/區塊1]

待分析數據：
{data_text}

請開始分析並填入以下標籤：
[區塊1] 九項指標條列分析 [/區塊1]
[區塊2] 指標矛盾與教學 [/區塊2]
[區塊3] 雙重操作劇本 [/區塊3]
[區塊4] 總結與信心分數 [/區塊4]
"""
                ans = call_ai_detective(prompt)
                
                blocks = {
                    1: ("第一區塊：【九項指標趨勢深度判讀】", extract_best_block(ans, 1)),
                    2: ("第二區塊：【指標矛盾整合與風險抓漏】", extract_best_block(ans, 2)),
                    3: ("第三區塊：【全方位操作戰略：雙重劇本】", extract_best_block(ans, 3)),
                    4: ("第四區塊：【偵探總結與信心分數】", extract_best_block(ans, 4))
                }

                st.markdown("---")
                st.subheader(f"🛡️ 偵探診斷報告：{current_id}")
                
                for i in range(1, 5):
                    title, content = blocks[i]
                    if content:
                        st.markdown(f"""
                        <div class="report-card">
                            <p class="indicator-label">{title}</p>
                            {content.replace('\n', '<br>')}
                        </div>
                        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
