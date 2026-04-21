import yfinance as yf
import pandas as pd
import pandas_ta as ta
import gspread
from google.oauth2.service_account import Credentials
import os
import json
import pytz
import requests
import time
from datetime import datetime

# ==========================================
# 1. 取得台灣時間與證交所清單
# ==========================================
def get_top_200_stocks():
    print("正在從證交所獲取全市場成交量排行...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    url = "https://www.twse.com.tw/exchangeReport/STOCK_DAY_ALL?response=json"
    
    try:
        res = requests.get(url, headers=headers, timeout=15)
        data = res.json()
        if 'data' not in data:
            return {"2330.TW": "台積電", "2317.TW": "鴻海"}
        
        df_all = pd.DataFrame(data['data'])
        df_all[2] = df_all[2].str.replace(',', '').astype(float)
        df_top = df_all.sort_values(by=2, ascending=False).head(200)
        
        stock_dict = {}
        for _, row in df_top.iterrows():
            if len(str(row[0])) <= 5:
                stock_dict[f"{row[0]}.TW"] = row[1]
        print(f"成功取得 {len(stock_dict)} 支熱門股清單！")
        return stock_dict
    except Exception as e:
        print(f"清單獲取失敗: {e}")
        return {"2330.TW": "台積電", "2317.TW": "鴻海"}

# ==========================================
# 2. Google 試算表連線設定
# ==========================================
def setup_google_sheets():
    creds_json = os.environ.get('GCP_SERVICE_ACCOUNT_KEY')
    info = json.loads(creds_json)
    info['private_key'] = info['private_key'].replace('\\n', '\n')
    scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    creds = Credentials.from_service_account_info(info, scopes=scope)
    client = gspread.authorize(creds)
    sh = client.open_by_key("1UH-fwxENhGUDmQjTQJq72g1DzsxML_CPIJGg39cWAgM")
    return sh.get_worksheet(0)

# ==========================================
# 3. 核心運算邏輯 (強化版)
# ==========================================
def run_scanner():
    wks = setup_google_sheets()
    tz = pytz.timezone('Asia/Taipei')
    today_date = datetime.now(tz).strftime('%Y-%m-%d')
    
    stock_list = get_top_200_stocks()
    headers = ["日期", "股號", "股名", "開盤價", "最高價", "最低價", "收盤價", "成交量", "MA5", "MA20", "RSI14", "K", "D", "MACD", "BB_Upper", "BB_Lower", "漲跌幅%"]
    
    # 修正 update 語法警告
    wks.update(values=[headers], range_name='A1')

    final_rows = []
    print(f"開始分析指標...")
    
    # 這次只測試前 170 支
    for symbol, name in stock_list.items():
        try:
            # 關鍵修正：使用 group_by='ticker' 並處理 MultiIndex
            df = yf.download(symbol, period="4mo", interval="1d", progress=False, timeout=15)
            
            if df.empty or len(df) < 30:
                continue

            # 確保欄位是單層標題 (處理 yfinance 新版 MultiIndex 問題)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # 排除重複欄位並計算指標
            close = df['Close'].dropna()
            high = df['High'].dropna()
            low = df['Low'].dropna()

            df['MA5'] = ta.sma(close, length=5)
            df['MA20'] = ta.sma(close, length=20)
            df['RSI14'] = ta.rsi(close, length=14)
            kd = ta.stoch(high, low, close, k=9, d=3)
            df['K'], df['D'] = kd['STOCHk_9_3_3'], kd['STOCHd_9_3_3']
            macd = ta.macd(close)
            df['MACD_VAL'] = macd['MACD_12_26_9']
            bb = ta.bbands(close, length=20)
            df['BBU'], df['BBL'] = bb['BBU_20_2.0'], bb['BBL_20_2.0']
            df['Chg'] = close.pct_change() * 100
            
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
                round(float(latest['MACD_VAL']), 2),
                round(float(latest['BBU']), 2),
                round(float(latest['BBL']), 2),
                f"{round(float(latest['Chg']), 2)}%"
            ]
            final_rows.append(row)
            
            if len(final_rows) % 20 == 0:
                print(f"已成功計算 {len(final_rows)} 支股票...")

        except Exception as e:
            print(f"跳過 {symbol}: {e}")
            continue

    if final_rows:
        wks.batch_clear(['A2:Q500'])
        wks.append_rows(final_rows)
        print(f"✨ 成功更新 {len(final_rows)} 筆數據！")
    else:
        print("❌ 依然無法獲取數據，請檢查 yfinance 連線。")

if __name__ == "__main__":
    run_scanner()
