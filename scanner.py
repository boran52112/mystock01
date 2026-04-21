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
# 1. 取得證交所成交量排行 (Top 200)
# ==========================================
def get_top_200_stocks():
    print("正在獲取成交量排行清單...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    url = "https://www.twse.com.tw/exchangeReport/STOCK_DAY_ALL?response=json"
    
    try:
        res = requests.get(url, headers=headers, timeout=15)
        data = res.json()
        df_all = pd.DataFrame(data['data'])
        df_all[2] = df_all[2].str.replace(',', '').astype(float) # 第2欄是成交股數
        df_top = df_all.sort_values(by=2, ascending=False).head(200)
        
        stock_dict = {}
        for _, row in df_top.iterrows():
            if len(str(row[0])) <= 5: # 過濾掉權證
                stock_dict[f"{row[0]}.TW"] = row[1]
        return stock_dict
    except Exception as e:
        print(f"清單獲取失敗: {e}，使用備援清單")
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
# 3. 滾動歷史維護 (保留最近 5 天)
# ==========================================
def maintenance_rolling_history(wks, max_days=5):
    print(f"檢查資料庫容量 (目標保留 {max_days} 天)...")
    all_dates = wks.col_values(1)[1:] # 取得日期欄，跳過標題
    if not all_dates: return

    unique_dates = sorted(list(set(all_dates))) # 取得不重複日期並排序
    
    if len(unique_dates) > max_days:
        # 找出最老的日期
        oldest_dates = unique_dates[:(len(unique_dates) - max_days)]
        print(f"檢測到資料已超過 {max_days} 天，正在移除舊資料: {oldest_dates}")
        
        # 獲取所有資料重新過濾 (這是對 Google Sheets 最安全的做法)
        all_data = wks.get_all_values()
        header = all_data[0]
        rows = all_data[1:]
        
        # 只保留不屬於 oldest_dates 的資料
        new_rows = [row for row in rows if row[0] not in oldest_dates]
        
        wks.clear()
        wks.append_row(header)
        wks.append_rows(new_rows)
        print("資料庫清理完成。")

# ==========================================
# 4. 主掃描程式
# ==========================================
def run_scanner():
    wks = setup_google_sheets()
    stock_list = get_top_200_stocks()
    
    headers = [
        "日期", "股號", "股名", "開盤價", "最高價", "最低價", "收盤價", "成交量",
        "MA5", "MA20", "RSI14", "K", "D", "MACD", "BB_Upper", "BB_Lower", "漲跌幅%"
    ]
    # 如果試算表是空的，先寫入標題
    if not wks.row_values(1):
        wks.append_row(headers)

    final_rows = []
    print(f"開始分析 {len(stock_list)} 支熱門股指標...")

    for symbol, name in stock_list.items():
        try:
            df = yf.download(symbol, period="4mo", interval="1d", progress=False)
            if df.empty or len(df) < 30: continue

            # 修正 MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # 獲取真實交易日期 (Data Date)
            real_date = df.index[-1].strftime('%Y-%m-%d')
            
            # 指標計算
            close = df['Close'].squeeze()
            high = df['High'].squeeze()
            low = df['Low'].squeeze()
            
            df.ta.sma(length=5, append=True)
            df.ta.sma(length=20, append=True)
            df.ta.rsi(length=14, append=True)
            df.ta.stoch(high=high, low=low, close=close, k=9, d=3, append=True)
            df.ta.macd(close=close, append=True)
            df.ta.bbands(close=close, length=20, std=2, append=True)
            df['CHG'] = df['Close'].pct_change() * 100

            def get_col(keyword):
                cols = [c for c in df.columns if keyword in c]
                return df[cols[0]].iloc[-1] if cols else 0

            latest = df.iloc[-1]
            
            row = [
                real_date, symbol, name,
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
            
            if len(final_rows) % 30 == 0:
                print(f"進度: {len(final_rows)} 支...")

        except Exception:
            continue

    if final_rows:
        # 寫入新資料 (不刪除舊的，直接往後貼)
        wks.append_rows(final_rows)
        print(f"✨ 成功寫入 {len(final_rows)} 筆今日數據。")
        
        # 執行維護邏輯：保持 5 天歷史
        maintenance_rolling_history(wks, max_days=5)
    else:
        print("❌ 本次執行未抓取到有效數據。")

if __name__ == "__main__":
    run_scanner()
