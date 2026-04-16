import os
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import time
from datetime import datetime, timedelta
from FinMind.data import DataLoader

# --- 1. 設定區 ---
FINMIND_TOKEN = os.getenv("FINMIND_TOKEN", "")
dl = DataLoader()
if FINMIND_TOKEN:
    try:
        dl.login_token(FINMIND_TOKEN)
    except:
        print("FinMind Token 登入失敗，將以無 Token 模式執行。")

# 硬性規定的 13 個欄位
COLUMNS = [
    'Date', 'StockID', 'StockName', 'Close', 
    'MA_Signal', 'MACD_Signal', 'RSI_Signal', 'KD_Signal', 'BB_Signal',
    'Shadow_Signal', 'Gap_Signal', 'Inst_Signal', 'Margin_Signal'
]

# --- 2. 內建 200 檔核心股票清單 (確保 API 斷線也能跑) ---
# 包含 0050, 0056 與熱門權值股
CORE_STOCKS = [
    "2330", "2317", "2454", "2308", "2382", "2303", "2881", "2882", "3008", "2603",
    "2609", "2615", "2357", "3231", "2376", "6669", "2408", "2409", "3481", "3037",
    "2324", "2353", "2356", "2377", "2379", "2383", "2401", "2449", "2451", "3034",
    "3035", "3044", "3231", "3443", "3532", "3711", "4919", "4938", "4958", "4961",
    "6176", "6213", "6239", "6415", "8046", "8210", "1101", "1102", "1216", "1301",
    "1303", "1326", "1402", "1503", "1504", "1513", "1519", "1605", "1722", "1802",
    "2002", "2006", "2105", "2201", "2204", "2206", "2347", "2501", "2542", "2606",
    "2610", "2618", "2707", "2801", "2809", "2812", "2834", "2880", "2883", "2884",
    "2885", "2886", "2887", "2888", "2890", "2891", "2892", "2912", "5871", "5876",
    "5880", "6505", "8046", "8454", "9904", "9910", "9921", "9945", "1476", "9933"
    # 此處已縮減，實務上可放滿 200 檔
]

# 模擬名稱對照 (避免 FinMind 沒抓到名稱)
STOCK_NAMES = {
    "2330": "台積電", "2317": "鴻海", "2454": "聯發科", "2308": "台達電", "2382": "廣達",
    "2881": "富邦金", "2882": "國泰金", "2002": "中鋼", "2603": "長榮", "2303": "聯電"
}

# --- 3. 核心邏輯 ---

def compute_signals(df, stock_id, inst_data, margin_data):
    res = {col: 0 for col in COLUMNS}
    res['Date'] = datetime.now().strftime('%Y-%m-%d')
    res['StockID'] = stock_id
    res['StockName'] = STOCK_NAMES.get(stock_id, f"個股 {stock_id}")
    
    if len(df) < 30: return res
    
    # 這裡的 df 是 yfinance 抓到的
    latest_close = float(df['Close'].iloc[-1])
    res['Close'] = round(latest_close, 2)
    
    try:
        # 1. MA
        ma5 = ta.sma(df['Close'], length=5)
        ma20 = ta.sma(df['Close'], length=20)
        res['MA_Signal'] = 1 if (ma5.iloc[-1] > ma20.iloc[-1] and latest_close > ma20.iloc[-1]) else -1

        # 2. MACD
        macd = ta.macd(df['Close'])
        res['MACD_Signal'] = 1 if macd['MACDs_12_26_9'].iloc[-1] > 0 else -1

        # 3. RSI
        rsi = ta.rsi(df['Close'], length=14)
        res['RSI_Signal'] = 1 if rsi.iloc[-1] > 50 else -1

        # 4. KD
        kd = ta.stoch(df['High'], df['Low'], df['Close'])
        res['KD_Signal'] = 1 if kd['STOCHk_14_3_3'].iloc[-1] > kd['STOCHd_14_3_3'].iloc[-1] else -1

        # 5. BB
        bb = ta.bbands(df['Close'], length=20)
        res['BB_Signal'] = 1 if latest_close > bb['BBM_20_2.0'].iloc[-1] else -1

        # 6. 下影線
        body = abs(df['Close'].iloc[-1] - df['Open'].iloc[-1])
        low_shadow = min(df['Open'].iloc[-1], df['Close'].iloc[-1]) - df['Low'].iloc[-1]
        res['Shadow_Signal'] = 1 if low_shadow > (body * 2) and body > 0 else 0

        # 7. 跳空
        res['Gap_Signal'] = 1 if df['Low'].iloc[-1] > df['High'].iloc[-2] else (-1 if df['High'].iloc[-1] < df['Low'].iloc[-2] else 0)

        # 8. 法人 (FinMind)
        if not inst_data.empty:
            net_buy = inst_data['buy'].sum() - inst_data['sell'].sum()
            res['Inst_Signal'] = 1 if net_buy > 0 else -1

        # 9. 融資 (FinMind)
        if len(margin_data) >= 2:
            res['Margin_Signal'] = 1 if margin_data['MarginPurchaseStock'].iloc[-1] < margin_data['MarginPurchaseStock'].iloc[-2] else -1
            
    except Exception as e:
        print(f"指標計算錯誤 {stock_id}: {e}")
        
    return res

def main():
    print(f"啟動掃描器 v3.1... 預計掃描 {len(CORE_STOCKS)} 檔股票")
    all_results = []
    today_str = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=45)).strftime('%Y-%m-%d')

    for sid in CORE_STOCKS:
        try:
            print(f"正在分析: {sid}")
            # 下載技術面資料 (yfinance - 免費)
            df = yf.download(f"{sid}.TW", start=start_date, progress=False)
            
            # 下載籌碼資料 (FinMind)
            inst_data = pd.DataFrame()
            margin_data = pd.DataFrame()
            if FINMIND_TOKEN:
                try:
                    inst_data = dl.taiwan_stock_institutional_investors(stock_id=sid, start_date=today_str)
                    margin_data = dl.taiwan_stock_margin_purchase_short_sale(stock_id=sid, start_date=start_date)
                    time.sleep(0.3)
                except:
                    pass # Token 失效時不報錯，填 0 繼續執行

            # 計算訊號
            res = compute_signals(df, sid, inst_data, margin_data)
            all_results.append(res)
            
        except Exception as e:
            print(f"跳過 {sid}: {e}")

    # 合併與存檔
    new_df = pd.DataFrame(all_results)
    file_path = 'daily_scan.csv'
    
    if os.path.exists(file_path):
        try:
            old_df = pd.read_csv(file_path)
            # 確保舊資料格式正確，如果不正確就直接捨棄
            if 'Date' in old_df.columns:
                final_df = pd.concat([old_df, new_df]).drop_duplicates(subset=['Date', 'StockID'], keep='last')
            else:
                final_df = new_df
        except:
            final_df = new_df
    else:
        final_df = new_df

    # 只保留最近 3 天
    valid_dates = sorted(final_df['Date'].unique(), reverse=True)[:3]
    final_df = final_df[final_df['Date'].isin(valid_dates)]
    
    final_df.to_csv(file_path, index=False, encoding='utf-8-sig')
    print(f"成功! 檔案已更新，共 {len(final_df)} 筆資料。")

if __name__ == "__main__":
    main()
