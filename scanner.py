import yfinance as yf
import pandas as pd
import pandas_ta as ta
import gspread
from google.oauth2.service_account import Credentials
import os
import json
import pytz
import requests
from datetime import datetime

# ==========================================
# 1. 取得成交量排行 (Top 200) - 加入人類偽裝
# ==========================================
def get_top_200_stocks():
    print("正在獲取成交量排行清單...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    url = "https://www.twse.com.tw/exchangeReport/STOCK_DAY_ALL?response=json"
    
    try:
        res = requests.get(url, headers=headers, timeout=15)
        data = res.json()
        if 'data' not in data: return {"2330.TW": "台積電"}
        
        df_all = pd.DataFrame(data['data'])
        df_all[2] = df_all[2].str.replace(',', '').astype(float)
        df_top = df_all.sort_values(by=2, ascending=False).head(200)
        
        stock_dict = {}
        for _, row in df_top.iterrows():
            if len(str(row[0])) <= 5: # 只取股號不取權證
                stock_dict[f"{row[0]}.TW"] = row[1]
        return stock_dict
    except Exception as e:
        print(f"清單獲取失敗: {e}")
        return {"2330.TW": "台積電", "2317.TW": "鴻海"}

# ==========================================
# 2. Google 試算表連線
# ==========================================
def setup_google_sheets():
    # 對應你 Secrets 中的金鑰名稱 gcp_service_account_raw
    creds_json = os.environ.get('GCP_SERVICE_ACCOUNT_KEY') # 此處若你在 YAML 設為此名，請維持
    if not creds_json:
        # 備援：若開發環境名稱不同
        creds_json = os.environ.get('gcp_service_account_raw')

    info = json.loads(creds_json)
    info['private_key'] = info['private_key'].replace('\\n', '\n')
    scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    creds = Credentials.from_service_account_info(info, scopes=scope)
    client = gspread.authorize(creds)
    sh = client.open_by_key("1UH-fwxENhGUDmQjTQJq72g1DzsxML_CPIJGg39cWAgM")
    return sh.get_worksheet(0)

# ==========================================
# 3. 核心清理邏輯 (去重 + 滾動 5 天)
# ==========================================
def clean_sheet_data(wks, target_date, max_days=5):
    print(f"正在清理資料庫，移除日期：{target_date} (若存在)...")
    all_values = wks.get_all_values()
    if len(all_values) <= 1: return # 只有標題就不處理
    
    header = all_values[0]
    rows = all_values[1:]
    
    # 1. 移除與本次寫入日期相同的舊資料 (去重)
    rows = [row for row in rows if row[0] != target_date]
    
    # 2. 滾動天數控制 (保留 5 天)
    unique_dates = sorted(list(set([row[0] for row in rows])))
    if len(unique_dates) >= max_days:
        oldest_dates = unique_dates[:(len(unique_dates) - max_days + 1)]
        print(f"資料庫超過 {max_days} 天，移除最老日期: {oldest_dates}")
        rows = [row for row in rows if row[0] not in oldest_dates]
        
    # 3. 回寫清空後的資料
    wks.clear()
    wks.append_row(header)
    if rows:
        wks.append_rows(rows)

# ==========================================
# 4. 主掃描程式
# ==========================================
def run_scanner():
    wks = setup_google_sheets()
    stock_list = get_top_200_stocks()
    
    final_rows = []
    real_data_date = ""

    print("開始下載並計算技術指標...")
    for symbol, name in stock_list.items():
        try:
            df = yf.download(symbol, period="4mo", interval="1d", progress=False)
            if df.empty or len(df) < 25: continue

            # 粉碎 MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # 取得真實交易日期 (這支股票最後一次成交的日期)
            real_data_date = df.index[-1].strftime('%Y-%m-%d')
            
            # 指標計算
            df.ta.sma(length=5, append=True)
            df.ta.sma(length=20, append=True)
            df.ta.rsi(length=14, append=True)
            df.ta.stoch(k=9, d=3, append=True)
            df.ta.macd(append=True)
            df.ta.bbands(length=20, append=True)
            df['CHG%'] = df['Close'].pct_change() * 100

            # 智慧欄位選取
            def gv(kw):
                c = [col for col in df.columns if kw in col]
                val = df[c[0]].iloc[-1] if c else 0
                return 0 if pd.isna(val) or val == float('inf') or val == float('-inf') else val

            latest = df.iloc[-1]
            row = [
                real_data_date, symbol, name,
                round(float(latest['Open']), 2),
                round(float(latest['High']), 2),
                round(float(latest['Low']), 2),
                round(float(latest['Close']), 2),
                int(latest['Volume']),
                round(gv('SMA_5'), 2),
                round(gv('SMA_20'), 2),
                round(gv('RSI_14'), 2),
                round(gv('STOCHk_9_3_3'), 2),
                round(gv('STOCHd_9_3_3'), 2),
                round(gv('MACD_12_26_9'), 2),
                round(gv('BBU_20_2.0'), 2),
                round(gv('BBL_20_2.0'), 2),
                f"{round(float(latest['CHG%']), 2)}%" if not pd.isna(latest['CHG%']) else "0%"
            ]
            final_rows.append(row)
        except:
            continue

    if final_rows and real_data_date:
        # 在寫入新資料前，先執行清理去重
        clean_sheet_data(wks, real_data_date, max_days=5)
        
        # 寫入最新資料
        wks.append_rows(final_rows)
        print(f"✨ 成功更新 {real_data_date} 的 {len(final_rows)} 筆熱門股數據！")
    else:
        print("❌ 執行失敗，未獲取任何有效數據。")

if __name__ == "__main__":
    run_scanner()
