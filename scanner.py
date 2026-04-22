import yfinance as yf
import pandas as pd
import pandas_ta as ta
import gspread
from google.oauth2.service_account import Credentials
import os
import json
import requests
from datetime import datetime

def get_top_200_stocks():
    print("正在獲取成交量排行清單...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    url = "https://www.twse.com.tw/exchangeReport/STOCK_DAY_ALL?response=json"
    try:
        res = requests.get(url, headers=headers, timeout=15)
        df_all = pd.DataFrame(res.json()['data'])
        df_all[2] = df_all[2].str.replace(',', '').astype(float)
        df_top = df_all.sort_values(by=2, ascending=False).head(200)
        return {f"{row[0]}.TW": row[1] for _, row in df_top.iterrows() if len(str(row[0])) <= 5}
    except:
        return {"2330.TW": "台積電", "2317.TW": "鴻海"}

def setup_google_sheets():
    creds_json = os.environ.get('GCP_SERVICE_ACCOUNT_KEY') or os.environ.get('gcp_service_account_raw')
    info = json.loads(creds_json)
    info['private_key'] = info['private_key'].replace('\\n', '\n')
    scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    creds = Credentials.from_service_account_info(info, scopes=scope)
    client = gspread.authorize(creds)
    return client.open_by_key("1UH-fwxENhGUDmQjTQJq72g1DzsxML_CPIJGg39cWAgM").get_worksheet(0)

def run_scanner():
    wks = setup_google_sheets()
    stock_list = get_top_200_stocks()
    
    headers = ["日期", "股號", "股名", "開盤價", "最高價", "最低價", "收盤價", "成交量", "MA5", "MA20", "RSI14", "K", "D", "MACD", "BB_Upper", "BB_Lower", "漲跌幅%"]
    
    final_batch = []
    print(f"開始執行 5 日歷史回溯分析...")

    for symbol, name in stock_list.items():
        try:
            df = yf.download(symbol, period="4mo", interval="1d", progress=False)
            if len(df) < 30: continue
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            # 技術指標運算
            df.ta.sma(length=5, append=True)
            df.ta.sma(length=20, append=True)
            df.ta.rsi(length=14, append=True)
            df.ta.stoch(k=9, d=3, append=True)
            df.ta.macd(append=True)
            df.ta.bbands(length=20, append=True)
            df['CHG%'] = df['Close'].pct_change() * 100
            df = df.fillna(0) # 徹底消滅 NaN

            # 擷取最後 5 天
            latest_5 = df.tail(5)
            for idx, row in latest_5.iterrows():
                def gv(kw):
                    c = [col for col in df.columns if kw in col]
                    return round(float(row[c[0]]), 2) if c else 0

                data_row = [
                    idx.strftime('%Y-%m-%d'), symbol, name,
                    round(float(row['Open']), 2), round(float(row['High']), 2),
                    round(float(row['Low']), 2), round(float(row['Close']), 2),
                    int(row['Volume']),
                    gv('SMA_5'), gv('SMA_20'), gv('RSI_14'),
                    gv('STOCHk_9_3_3'), gv('STOCHd_9_3_3'), gv('MACD_12_26_9'),
                    gv('BBU_20_2.0'), gv('BBL_20_2.0'),
                    f"{round(float(row['CHG%']), 2)}%"
                ]
                final_batch.append(data_row)
        except:
            continue

    if final_batch:
        wks.clear()
        wks.append_row(headers)
        wks.append_rows(final_batch)
        print(f"✨ 歷史回溯完成！已存入 {len(final_batch)} 筆數據 (200支 x 5日)。")

if __name__ == "__main__":
    run_scanner()
