import os
import pandas as pd
import yfinance as yf
import pandas_ta as ta

def run_scanner():
    print("🚀 啟動「全方位」掃描儀 (含技術指標 + 預留籌碼位)...")
    
    if not os.path.exists("stock_list.csv"): return
    df_list = pd.read_csv("stock_list.csv")
    
    def format_id(row):
        s_id = str(row['stock_id'])
        m_type = str(row['type']).lower()
        # 只要是 4-6 碼的我們都試試看（包含 B 結尾的 ETF）
        if len(s_id) > 6: return None
        if 'twse' in m_type: return f"{s_id}.TW"
        if 'tpex' in m_type: return f"{s_id}.TWO"
        return None

    df_list['full_id'] = df_list.apply(format_id, axis=1)
    df_list = df_list.dropna(subset=['full_id'])
    
    # 目前測試：我們先跑前 100 檔（你可以視情況增加，或是全跑）
    all_tickers = df_list['full_id'].tolist()
    test_tickers = all_tickers[:100] 
    
    print(f"✅ 準備掃描 {len(test_tickers)} 檔標的...")

    # 下載股價
    data = yf.download(test_tickers, period="60d", group_by='ticker', threads=True)
    
    results = []

    for ticker in test_tickers:
        try:
            # 檢查資料是否存在
            if ticker not in data or data[ticker].empty:
                print(f"⏩ {ticker}: 無股價資料，跳過")
                continue
                
            df = data[ticker].copy().dropna()
            if len(df) < 30: continue

            # --- 計算 5 個技術指標 ---
            df['MA5'] = ta.sma(df['Close'], length=5)
            df['MA20'] = ta.sma(df['Close'], length=20)
            df['K'] = ta.stoch(df['High'], df['Low'], df['Close'])['STOCKk_14_3_3']
            df['D'] = ta.stoch(df['High'], df['Low'], df['Close'])['STOCKd_14_3_3']
            macd = ta.macd(df['Close'])
            df['MACD'] = macd['MACDH_12_26_9'] # 拿柱狀圖

            for i in range(-3, 0): # 抓最後三天
                day_data = df.iloc[i]
                
                # 多空判定邏輯
                ma_status = 1 if (day_data['Close'] > day_data['MA20'] and day_data['MA5'] > day_data['MA20']) else (-1 if (day_data['Close'] < day_data['MA20'] and day_data['MA5'] < day_data['MA20']) else 0)
                kd_status = 1 if day_data['K'] > day_data['D'] else -1
                macd_status = 1 if day_data['MACD'] > 0 else -1
                gap_status = 1 if df.iloc[i]['Low'] > df.iloc[i-1]['High'] else (-1 if df.iloc[i]['High'] < df.iloc[i-1]['Low'] else 0)

                results.append({
                    '日期': df.index[i].strftime('%Y-%m-%d'),
                    '代號': ticker,
                    '均線': ma_status,
                    'KD': kd_status,
                    'MACD': macd_status,
                    '缺口': gap_status,
                    '法人': 0, # 預留位，明天補
                    '融資': 0, # 預留位，明天補
                    '收盤價': round(day_data['Close'], 2)
                })
            print(f"✅ {ticker}: 技術指標計算完成")
        except:
            continue

    if results:
        final_df = pd.DataFrame(results)
        final_df.to_csv("daily_scan.csv", index=False, encoding="utf-8-sig")
        print(f"🎉 完成！daily_scan.csv 已產出，共 {len(final_df)} 筆。")

if __name__ == "__main__":
    run_scanner()
