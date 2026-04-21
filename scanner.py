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
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    url = "https://www.twse.com.tw/exchangeReport/STOCK_DAY_ALL?response=json"
    
    try:
        res = requests.get(url, headers=headers, timeout=15)
        data = res.json()
        df_all = pd.DataFrame(data['data'])
        df_all[2] = df_all[2].str.replace(',', '').astype(float) # 成交股數
        df_top = df_all.sort_values(by=2, ascending=False).head(200)
        
        stock_dict = {}
        for _, row in df_top.iterrows():
            if len(str(row[0])) <= 5:
                symbol = f"{row[0]}.TW"
                stock_dict[symbol] = row[1]
        return stock_dict
    except Exception as e:
        print(f"清單獲取失敗: {e}，改用基本清單。")
        return {"2330.TW": "台積電", "2317.TW": "鴻海"}

# ==========================================
# 2. Google 試算表連線
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
# 3. 核心計算引擎 (鋼鐵人 v6.1 版)
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
    wks.update('A1', [headers])

    final_rows = []
    print(f"成功取得 {len(stock_list)} 支清單，開始深度分析指標...")

    for symbol, name in stock_list.items():
        try:
            # 1. 下載數據
            df = yf.download(symbol, period="4mo", interval="1d", progress=False)
            if df.empty or len(df) < 30: continue

            # --- 關鍵修正：徹底粉碎雙層標題 ---
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # 2. 確保數據是乾淨的 Series
            close = df['Close'].squeeze()
            high = df['High'].squeeze()
            low = df['Low'].squeeze()

            # 3. 計算指標 (使用附加模式)
            df.ta.sma(length=5, append=True)
            df.ta.sma(length=20, append=True)
            df.ta.rsi(length=14, append=True)
            df.ta.stoch(high=high, low=low, close=close, k=9, d=3, append=True)
            df.ta.macd(close=close, append=True)
            df.ta.bbands(close=close, length=20, std=2, append=True)
            df['CHG'] = df['Close'].pct_change() * 100

            # 4. 智慧提取：不管欄位名稱變什麼，只要包含關鍵字就抓
            def get_col(keyword):
                cols = [c for c in df.columns if keyword in c]
                return df[cols[0]].iloc[-1] if cols else 0

            latest = df.iloc[-1]
            
            # 依照標題順序組合
            row = [
                today_date, symbol, name,
                round(float(latest['Open']), 2),
                round(float(latest['High']), 2),
                round(float(latest['Low']), 2),
                round(float(latest['Close']), 2),
                int(latest['Volume']),
                round(get_col('SMA_5'), 2),
                round(get_col('SMA_20'), 2),
                round(get_col('RSI_14'), 2),
                round(get_col('STOCHk_9_3_3'), 2),
                round(get_col('STOCHd_9_3_3'), 2),
                round(get_col('MACD_12_26_9'), 2),
                round(get_col('BBU_20_2.0'), 2),
                round(get_col('BBL_20_2.0'), 2),
                f"{round(float(latest['CHG']), 2)}%"
            ]
            final_rows.append(row)
            if len(final_rows) % 20 == 0:
                print(f"已完成 {len(final_rows)} 支...")

        except Exception as e:
            # print(f"{symbol} 跳過: {e}") # 偵錯用，平時可關閉
            continue

    # 5. 批次存入
    if final_rows:
        wks.batch_clear(['A2:Q500'])
        wks.append_rows(final_rows)
        print(f"✨ 大功告成！試算表已更新 {len(final_rows)} 支熱門股！")
    else:
        print("❌ 失敗：未能計算出任何數據，請檢查 yfinance 連線。")

if __name__ == "__main__":
    run_scanner()
