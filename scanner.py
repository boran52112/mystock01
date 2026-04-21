import yfinance as yf
import pandas as pd
import pandas_ta as ta
import gspread
from google.oauth2.service_account import Credentials
import os
import json
import pytz
from datetime import datetime

# ==========================================
# 1. 獲取精準台灣時間 (國家標準時間邏輯)
# ==========================================
def get_taiwan_date():
    # 設定台灣時區
    tz = pytz.timezone('Asia/Taipei')
    # 直接獲取目前的台灣時間
    tw_now = datetime.now(tz)
    return tw_now.strftime('%Y-%m-%d')

# ==========================================
# 2. Google 試算表連線設定
# ==========================================
def setup_google_sheets():
    # 對應 YAML 檔案中的環境變數名稱
    creds_json = os.environ.get('GCP_SERVICE_ACCOUNT_KEY')
    if not creds_json:
        raise ValueError("找不到 GCP_SERVICE_ACCOUNT_KEY，請檢查 GitHub Secrets 設定。")
    
    info = json.loads(creds_json)
    info['private_key'] = info['private_key'].replace('\\n', '\n')
    
    scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    creds = Credentials.from_service_account_info(info, scopes=scope)
    client = gspread.authorize(creds)
    
    # 試算表 ID
    sheet_id = "1UH-fwxENhGUDmQjTQJq72g1DzsxML_CPIJGg39cWAgM"
    sh = client.open_by_key(sheet_id)
    return sh.get_worksheet(0)

# ==========================================
# 3. 核心運算邏輯
# ==========================================
def run_scanner():
    wks = setup_google_sheets()
    today_date = get_taiwan_date()
    
    # 定義新版標題欄位 (共 17 欄)
    headers = [
        "日期", "股號", "股名", "開盤價", "最高價", "最低價", "收盤價", "成交量",
        "MA5", "MA20", "RSI14", "K", "D", "MACD", "BB_Upper", "BB_Lower", "漲跌幅%"
    ]
    
    # 自動整地：檢查第一列標題，若不符則初始化
    try:
        existing_headers = wks.row_values(1)
        if not existing_headers or existing_headers[0] != "日期":
            print("初始化試算表欄位...")
            wks.clear()
            wks.insert_row(headers, 1)
    except:
        wks.insert_row(headers, 1)

    # 監測清單
    stock_list = {
        "2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科",
        "2308.TW": "台達電", "2382.TW": "廣達", "2881.TW": "富邦金",
        "2882.TW": "國泰金", "3711.TW": "日月光投控"
    }

    final_rows = []
    
    for symbol, name in stock_list.items():
        try:
            print(f"正在分析: {symbol}...")
            # 抓取 3 個月資料計算指標
            df = yf.download(symbol, period="3mo", interval="1d", progress=False)
            if df.empty: continue
            
            # --- 技術指標計算 ---
            df['MA5'] = ta.sma(df['Close'], length=5)
            df['MA20'] = ta.sma(df['Close'], length=20)
            df['RSI14'] = ta.rsi(df['Close'], length=14)
            kd = ta.stoch(df['High'], df['Low'], df['Close'], k=9, d=3)
            df['K'] = kd['STOCHk_9_3_3']
            df['D'] = kd['STOCHd_9_3_3']
            macd = ta.macd(df['Close'])
            df['MACD'] = macd['MACD_12_26_9']
            bb = ta.bbands(df['Close'], length=20)
            df['BB_U'] = bb['BBU_20_2.0']
            df['BB_L'] = bb['BBL_20_2.0']
            df['Chg'] = df['Close'].pct_change() * 100
            
            # 取得最新一筆數據
            latest = df.iloc[-1]
            
            row = [
                today_date, symbol, name,
                round(float(latest['Open']), 2),
                round(float(latest['High']), 2),
                round(float(latest['Low']), 2),
                round(float(latest['Close']), 2),
                int(latest['Volume']),
                round(float(latest['MA5']), 2),
                round(float(latest['MA20']), 2),
                round(float(latest['RSI14']), 2),
                round(float(latest['K']), 2),
                round(float(latest['D']), 2),
                round(float(latest['MACD']), 2),
                round(float(latest['BB_U']), 2),
                round(float(latest['BB_L']), 2),
                f"{round(float(latest['Chg']), 2)}%"
            ]
            final_rows.append(row)
        except Exception as e:
            print(f"{symbol} 出錯: {e}")

    # 寫入試算表
    if final_rows:
        wks.append_rows(final_rows)
        print(f"成功更新 {len(final_rows)} 筆台灣標準時間數據！")

if __name__ == "__main__":
    run_scanner()
