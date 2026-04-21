import yfinance as yf
import pandas as pd
import pandas_ta as ta
import gspread
from google.oauth2.service_account import Credentials
import os
import json
import requests
from datetime import datetime, timedelta

# ==========================================
# 1. 權威時間校準 (2026 模擬版)
# ==========================================
def get_simulated_2026_date():
    try:
        # 優先嘗試 WorldTimeAPI
        response = requests.get("http://worldtimeapi.org/api/timezone/Asia/Taipei", timeout=5)
        current_time = datetime.fromisoformat(response.json()['datetime'])
    except:
        # 備援：使用 Google 伺服器時間
        response = requests.get("https://www.google.com", timeout=5)
        date_str = response.headers.get('date')
        current_time = datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %Z') + timedelta(hours=8)
    
    # 強制轉換為 2026 年 (模擬 2026-04-16 基底)
    simulated_date = current_time.replace(year=2026)
    return simulated_date.strftime('%Y-%m-%d')

# ==========================================
# 2. Google 試算表連線設定
# ==========================================
def setup_google_sheets():
    # 從 GitHub Secrets 讀取 JSON 字串
    creds_json = os.environ.get('GCP_KEY_JSON')
    if not creds_json:
        raise ValueError("找不到 GCP_KEY_JSON，請檢查 GitHub Secrets 設定。")
    
    # 修正私鑰換行符號問題
    info = json.loads(creds_json)
    info['private_key'] = info['private_key'].replace('\\n', '\n')
    
    scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    creds = Credentials.from_service_account_info(info, scopes=scope)
    client = gspread.authorize(creds)
    
    # 開啟試算表 (ID 已更新)
    sheet_id = "1UH-fwxENhGUDmQjTQJq72g1DzsxML_CPIJGg39cWAgM"
    sh = client.open_by_key(sheet_id)
    return sh.get_worksheet(0)

# ==========================================
# 3. 核心運算邏輯
# ==========================================
def run_scanner():
    wks = setup_google_sheets()
    sim_date = get_simulated_2026_date()
    
    # 定義新版標題欄位
    headers = [
        "日期", "股號", "股名", "開盤價", "最高價", "最低價", "收盤價", "成交量",
        "MA5", "MA20", "RSI14", "K", "D", "MACD", "BB_Upper", "BB_Lower", "漲跌幅%"
    ]
    
    # 自動整地：檢查第一列標題，若不符則重寫
    existing_headers = wks.row_values(1)
    if not existing_headers or existing_headers[0] != "日期" or len(existing_headers) < 10:
        print("檢測到舊版或空白標題，正在初始化新版欄位...")
        wks.clear()
        wks.append_row(headers)

    # 選股清單 (可自行增加)
    stock_list = {
        "2330.TW": "台積電",
        "2317.TW": "鴻海",
        "2454.TW": "聯發科",
        "2308.TW": "台達電",
        "2382.TW": "廣達",
        "2881.TW": "富邦金",
        "2882.TW": "國泰金",
        "3711.TW": "日月光投控"
    }

    final_rows = []
    
    for symbol, name in stock_list.items():
        try:
            print(f"正在分析: {symbol} {name}...")
            # 抓取 3 個月資料確保指標準確
            df = yf.download(symbol, period="3mo", interval="1d", progress=False)
            
            if df.empty: continue
            
            # --- 計算技術指標 ---
            # 1. 均線
            df['MA5'] = ta.sma(df['Close'], length=5)
            df['MA20'] = ta.sma(df['Close'], length=20)
            
            # 2. RSI
            df['RSI14'] = ta.rsi(df['Close'], length=14)
            
            # 3. KD (9, 3, 3)
            kd = ta.stoch(df['High'], df['Low'], df['Close'], k=9, d=3, smooth_k=3)
            df['K'] = kd['STOCHk_9_3_3']
            df['D'] = kd['STOCHd_9_3_3']
            
            # 4. MACD (12, 26, 9)
            macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
            df['MACD'] = macd['MACD_12_26_9']
            
            # 5. 布林通道 (20, 2)
            bb = ta.bbands(df['Close'], length=20, std=2)
            df['BB_Upper'] = bb['BBU_20_2.0']
            df['BB_Lower'] = bb['BBL_20_2.0']
            
            # 6. 漲跌幅
            df['Chg'] = df['Close'].pct_change() * 100
            
            # 擷取最後一筆 (最新數據)
            latest = df.iloc[-1]
            
            # 格式化資料列 (對應 headers)
            row = [
                sim_date,
                symbol,
                name,
                round(float(latest['Open']), 2),
                round(float(latest['High']), 2),
                round(float(latest['Low']), 2),
                round(float(latest['Close']), 2),
                int(latest['Volume']),
                round(float(latest['MA5']), 2) if not pd.isna(latest['MA5']) else 0,
                round(float(latest['MA20']), 2) if not pd.isna(latest['MA20']) else 0,
                round(float(latest['RSI14']), 2) if not pd.isna(latest['RSI14']) else 0,
                round(float(latest['K']), 2) if not pd.isna(latest['K']) else 0,
                round(float(latest['D']), 2) if not pd.isna(latest['D']) else 0,
                round(float(latest['MACD']), 2) if not pd.isna(latest['MACD']) else 0,
                round(float(latest['BB_Upper']), 2) if not pd.isna(latest['BB_Upper']) else 0,
                round(float(latest['BB_Lower']), 2) if not pd.isna(latest['BB_Lower']) else 0,
                f"{round(float(latest['Chg']), 2)}%"
            ]
            final_rows.append(row)
            
        except Exception as e:
            print(f"處理 {symbol} 時發生錯誤: {e}")

    # 一次性寫入試算表 (減少 API 呼叫次數)
    if final_rows:
        wks.append_rows(final_rows)
        print(f"成功存入 {len(final_rows)} 筆 2026 模擬指標數據！")

if __name__ == "__main__":
    run_scanner()
