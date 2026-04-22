import streamlit as st
import pandas as pd
import gspread
import json
import google.generativeai as genai
from google.oauth2.service_account import Credentials
import yfinance as yf
import re
import numpy as np

# --- 1. 視覺優化 (手機版加大字體) ---
st.set_page_config(page_title="台股 AI 偵探系統 v5.0", layout="wide")
st.markdown("""
    <style>
    .stMarkdown p, .stMarkdown li { font-size: 1.25rem !important; line-height: 1.7; }
    .report-card { 
        background-color: #ffffff; padding: 20px; border-radius: 12px; 
        border: 1px solid #e0e0e0; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 25px;
    }
    .indicator-label { font-weight: bold; color: #1e3a8a; font-size: 1.3rem; margin-bottom: 10px; display: block; }
    </style>
    """, unsafe_allow_html=True)

if 'stock_data' not in st.session_state: st.session_state.stock_data = None
if 'current_id' not in st.session_state: st.session_state.current_id = ""
if 'current_name' not in st.session_state: st.session_state.current_name = ""

SHEET_ID = "1UH-fwxENhGUDmQjTQJq72g1DzsxML_CPIJGg39cWAgM"

# --- 2. 數據核心：讀取試算表 ---
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

# --- 3. 後端計算引擎：即時算齊九大指標 ---
def fetch_and_calculate_all(stock_id):
    suffixes = ["", ".TW", ".TWO"]
    for suf in suffixes:
        test_id = f"{stock_id}{suf}"
        try:
            df = yf.download(test_id, period="3mo", progress=False) # 抓3個月以利計算MA20和MACD
            if df.empty or len(df) < 25: continue
            
            # 1. 均線 MA
            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA20'] = df['Close'].rolling(window=20).mean()
            
            # 2. RSI14
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            df['RSI14'] = 100 - (100 / (1 + (gain / loss)))
            
            # 3. KDJ (9,3,3)
            low_list = df['Low'].rolling(9).min()
            high_list = df['High'].rolling(9).max()
            rsv = (df['Close'] - low_list) / (high_list - low_list) * 100
            df['KDJ_K'] = rsv.ewm(com=2).mean()
            df['KDJ_D'] = df['KDJ_K'].ewm(com=2).mean()
            
            # 4. MACD (12,26,9)
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            
            # 5. 布林通道 (20, 2)
            df['BB_Mid'] = df['MA20']
            df['BB_Std'] = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
            
            # 抓取股名 (yfinance info)
            try:
                name = yf.Ticker(test_id).info.get('longName', '未知')
            except: name = "即時偵測"

            df = df.reset_index().tail(5)
            return pd.DataFrame({
                '日期': df['Date'].dt.strftime('%Y-%m-%d'),
                '股號': test_id, '股名': name,
                '收盤價': df['Close'], '成交量': df['Volume'],
                'MA5': df['MA5'], 'MA20': df['MA20'], 'RSI14': df['RSI14'],
                'K值': df['KDJ_K'], 'D值': df['KDJ_D'], 'MACD': df['MACD'],
                '布林上軌': df['BB_Upper']
            })
        except: continue
    return pd.DataFrame()

# --- 4. 內容清洗與 AI 調用 ---
def clean_ai_content(text):
    text = text.replace(r'$\rightarrow$', ' → ').replace(r'\rightarrow', ' → ').replace('$', '')
    lines = text.split('\n')
    cleaned = [l for l in lines if not re.search(r'(Wait|prompt|label|verify|check|English|Input|template)', l, re.IGNORECASE)]
    return '\n'.join(cleaned).strip()

def extract_best_block(text, block_num):
    pattern = rf"\[區塊{block_num}\](.*?)\[/區塊{block_num}\]"
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches: return ""
    return clean_ai_content(matches[-1].strip())

def call_ai_detective(prompt):
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        model = genai.GenerativeModel('gemma-4-31b-it')
        return model.generate_content(prompt).text
    except Exception as e: return f"偵探連線錯誤: {str(e)}"

# --- 5. 主程式 ---
def main():
    st.title("🕵️‍♂️ 台股 AI 偵探戰情室 (全市場版)")
    
    user_input = st.text_input("🔢 請輸入股號 (例如: 2330)", placeholder="輸入代碼，系統自動計算全指標").strip()
    if st.button("🔍 開始偵查") and user_input:
        with st.spinner("正在調閱數據並計算指標..."):
            df_all = get_data_from_sheets()
            df_selected = pd.DataFrame()
            # 優先找試算表
            if not df_all.empty and '股號' in df_all.columns:
                df_all['股號'] = df_all['股號'].astype(str)
                df_selected = df_all[df_all['股號'].str.contains(user_input)].tail(5)
            
            # 若試算表沒資料，啟動後端計算引擎
            if df_selected.empty:
                st.toast(f"資料庫查無 {user_input}，已啟動即時計算引擎...")
                df_selected = fetch_and_calculate_all(user_input)
            
            if not df_selected.empty:
                st.session_state.stock_data = df_selected
                st.session_state.current_id = df_selected['股號'].iloc[-1]
                st.session_state.current_name = df_selected['股名'].iloc[-1]
            else: st.error("查無此股，請確認代碼。")

    if st.session_state.stock_data is not None:
        df_display = st.session_state.stock_data
        full_name = f"{st.session_state.current_id} {st.session_state.current_name}"
        st.subheader(f"📊 {full_name} 指標觀測站")
        
        # 統一表格顯示
        show_cols = ['日期', '收盤價', '成交量', 'MA5', 'MA20', 'RSI14']
        st.table(df_display[show_cols].style.format({
            '收盤價': '{:,.2f}', 'MA5': '{:,.2f}', 'MA20': '{:,.2f}', 
            'RSI14': '{:,.2f}', '成交量': '{:,.0f}'
        }))

        if st.button(f"🚀 啟動 {full_name} 深度診斷"):
            with st.spinner("AI 偵探正在判讀全市場數據..."):
                # 把所有計算出來的指標都餵給 AI
                data_text = df_display.to_string(index=False)
                prompt = f"""你是台股 AI 偵探。請呈現分析結果就好，並且以繁體中文顯示，如果是英文也請自動翻譯。請對分析結果與診斷以繁體中文說明，內容請以詳細教學內容為主，讓我可以清楚推斷的依據。
【關鍵指令】：嚴禁使用 LaTeX 數學符號（如 $ 符號）、英文僅用在專有名詞部分。使用一般箭頭 →。
請直接填入標籤，不要有任何標籤外的文字。
多一點教學內容，讓使用者更清楚判斷的依據相關的理論。

[區塊1]
【九項指標趨勢深度判讀】：
1. 5/20日均線走向與排列：根據某某理論...，所以.....
2. RSI14強弱與動能：根據某某理論...，所以...
3. KDJ(K/D值)轉折分析：根據某某理論...，所以...
4. MACD柱狀體動能變化：根據某某理論...，所以...
5. 布林通道相對位置與擠壓：根據某某理論...，所以...
6. 量價背離與動能支撐關係：根據某某理論...，所以...
7. 近5日漲跌連續性與市場情緒：根據某某理論...，所以...
8. 關鍵支撐與壓力位判斷：根據某某理論...，所以...
9. 整體多空力道權重：根據某某理論...，所以...
[/區塊1]

[區塊2]
【指標矛盾整合與風險抓漏】：
(針對上述指標間的背離進行教學解釋，根據區塊1的九項指標可能互相呼應，也可能互相矛盾，請給予說明與陷阱警示，並最後說明那些可信度較高，另外針對假突破、假跌破或者騙線等等股市中的陷阱也做一些推論，多一點教學理論。根據什麼理論？)
[/區塊2]

[區塊3]
【全方位操作戰略：雙重劇本】：
- 保守型劇本：(建議進場/停損/持股，請提供價位或者條件參考，並請提醒需要哪些條件配合？例如某些指標的支持或者數字多少)
- 激進型劇本：(建議突破/目標/短線，請提供價位或者條件參考，並請提醒需要哪些條件配合？例如某些指標的支持或者數字多少)
[/區塊3]

[區塊4]
【偵探總結與信心分數】：
總結：請根據以上三個區塊的內容，做一個總體性的報告，內容除了對整體股票的看法，也包括未來在什麼價位要做什麼動作，作多的條件為何？放空的條件？需要哪些指標支持？或者應該在什麼訊號要賣出？什麼訊號要買進？
信心分數：請說明信心分數是針對什麼信心
[/區塊4]

數據來源：
{data_text}"""
                ans = call_ai_detective(prompt)
                blocks = {1: "九項指標判讀", 2: "指標矛盾與教學", 3: "雙重劇本", 4: "總結與信心"}
                st.markdown("---")
                st.subheader(f"🛡️ 偵探診斷報告：{full_name}")
                for i in range(1, 5):
                    content = extract_best_block(ans, i)
                    if content:
                        st.markdown(f'<div class="report-card"><span class="indicator-label">第{i}區塊：{blocks[i]}</span>{content.replace("\n", "<br>")}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
