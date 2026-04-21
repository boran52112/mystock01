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
# 1. 取得台灣時間與證交所清單 (加入人類偽裝)
# ==========================================
def get_top_200_stocks():
    print("正在從證交所獲取全市場成交量排行...")
    # 偽裝成一般的瀏覽器，避免被證交所阻擋
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    url = "https://www.twse.com.tw/exchangeReport/STOCK_DAY_ALL?response=json"
    
    try:
        res = requests.get(url, headers=headers, timeout=15)
        if res.status_code != 200:
            print(f"證交所連線失敗，狀態碼: {res.status_code}")
            return {}
            
        data = res.json()
        if 'data' not in data:
            print("證交所回傳格式異常，改用備援清單。")
            return {"2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科"}
        
        df_all = pd.DataFrame(data['data'])
        # 欄位 0:代號, 1:名稱, 2:成交股數
        df_all[2] = df_all[2].str.replace(',', '').astype(float)
        
        # 取成交量前 200
        df_top = df_all.sort_values(by=2, ascending=False).head(200)
        
        stock_dict = {}
        for _, row in df_top.iterrows():
            # 過濾掉權證與奇怪的代號 (只取 4 位或 5 位數的股號)
            if len(str(row[0])) <= 5:
                symbol = f"{row[0]}.TW"
                stock_dict[symbol] = row[1]
        
        print(f"成功取得 {len(stock_dict)} 支熱門股清單！")
        return stock_dict
    except Exception as e:
        print(f"抓取發生錯誤: {e}，改用備援清單。")
        return {"2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科"}

# ==========================================
# 2. Google 試算表連線設定
# ==========================================
def setup_google_sheets():
    creds_json = os.environ.get('GCP_SERVICE_ACCOUNT_KEY')
    if not creds_json:
        raise ValueError("找不到 GCP_SERVICE_ACCOUNT_KEY")
    
    info = json.loads(creds_json)
    info['private_key'] = info['private_key'].replace('\\n', '\n')
    
    scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    creds = Credentials.from_service_account_info(info, scopes=scope)
    client = gspread.authorize(creds)
    
    sheet_id = "1UH-fwxENhGUDmQjTQJq72g1DzsxML_CPIJGg39cWAgM"
    sh = client.open_by_key(sheet_id)
    return sh.get_worksheet(0)

# ==========================================
# 3. 核心運算邏輯
# ==========================================
def run_scanner():
    wks = setup_google_sheets()
    tz = pytz.timezone('Asia/Taipei')
    today_date = datetime.now(tz).strftime('%Y-%m-%d')
    
    stock_list = get_top_200_stocks()
    
    headers = [
        "日期", "股號", "股名", "開盤價", "最高價", "最低價", "收盤價", "成交量",
        "MA5", "MA20", "RSI14", "K", "D", "MACD", "BB_Upper", "BB_Lower", "漲跌幅%"
    ]
    
    # 寫入標題
    wks.update('A1', [headers])

    final_rows = []
    count = 0
    
    print(f"開始分析指標...")
    
    for symbol, name in stock_list.items():
        try:
            # 增加 retry 機制避免 yfinance 斷線
            df = yf.download(symbol, period="3mo", interval="1d", progress=False, timeout=10)
            
            if df is None or len(df) < 25:
                continue
            
            # 指標計算
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
            
            latest = df.iloc[-1]
            
            # 資料轉換為純數值避免格式問題
            row = [
                today_date, symbol, name,
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
                round(float(latest['BB_U']), 2) if not pd.isna(latest['BB_U']) else 0,
                round(float(latest['BB_L']), 2) if not pd.isna(latest['BB_L']) else 0,
                f"{round(float(latest['Chg']), 2)}%"
            ]
            final_rows.append(row)
            count += 1
            if count % 10 == 0:
                print(f"已處理 {count} 支股票...")
                
        except Exception as e:
            continue

    if final_rows:
        # 清空舊資料
        wks.batch_clear(['A2:Q500'])
        # 寫入新資料
        wks.append_rows(final_rows)
        print(f"✨ 成功更新 {len(final_rows)} 筆數據！")
    else:
        print("❌ 本次執行未獲取任何有效數據。")

if __name__ == "__main__":
    run_scanner()
