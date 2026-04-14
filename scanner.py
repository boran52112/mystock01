import os
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from datetime import datetime

def run_scanner():
    print("🚀 啟動 9 大指標掃描儀...")
    
    # 1. 讀取清單
    if not os.path.exists("stock_list.csv"): return
    df_list = pd.read_csv("stock_list.csv")
    
    # 2. 修正代號邏輯 (根據你的截圖：tpex 與 twse)
    def format_id(row):
        m_type = str(row['type']).lower()
        s_id = str(row['stock_id'])
        if 'twse' in m_type: return f"{s_id}.TW"
        if 'tpex' in m_type: return f"{s_id}.TWO"
        return None

    df_list['full_id'] = df_list.apply(format_id, axis=1)
    df_list = df_list.dropna(subset=['full_id'])
    
    # 測試階段：我們先抓前 100 檔，確保穩定
    test_tickers = df_list['full_id'].tolist()[:100]
    print(f"✅ 準備掃描 {len(test_tickers)} 檔標的...")

    # 3. 抓取股價
    data = yf.download(test_tickers, period="60d", group_by='ticker', threads=True)
    
    results = []

    # 4. 進入核心計算迴圈
    for ticker in test_tickers:
        try:
            df = data[ticker].copy().dropna()
            if len(df) < 30: continue # 資料太少就跳過

            # --- A. 技術指標計算 (使用 pandas_ta) ---
            df['MA5'] = ta.sma(df['Close'], length=5)
            df['MA20'] = ta.sma(df['Close'], length=20)
            macd = ta.macd(df['Close'])
            df['MACD_hist'] = macd['MACDH_12_26_9']
            rsi = ta.rsi(df['Close'], length=14)
            kd = ta.stoch(df['High'], df['Low'], df['Close'])
            df['K'] = kd['STOCKk_14_3_3']
            df['D'] = kd['STOCKd_14_3_3']
            bbands = ta.bbands(df['Close'], length=20)
            df['BB_upper'] = bbands['BBU_20_2.0']
            df['BB_lower'] = bbands['BBL_20_2.0']

            # --- B. 判斷多空 (我們只拿最後三天的結果) ---
            for i in range(-3, 0): # 倒數第3天, 第2天, 第1天
                day_data = df.iloc[i]
                prev_day = df.iloc[i-1]
                
                # 均線判斷
                ma_status = 1 if (day_data['Close'] > day_data['MA20'] and day_data['MA5'] > day_data['MA20']) else (-1 if (day_data['Close'] < day_data['MA20'] and day_data['MA5'] < day_data['MA20']) else 0)
                
                # KD 判斷
                kd_status = 1 if day_data['K'] > day_data['D'] else -1
                
                # 跳空缺口判斷
                gap_status = 1 if df.iloc[i]['Low'] > df.iloc[i-1]['High'] else (-1 if df.iloc[i]['High'] < df.iloc[i-1]['Low'] else 0)

                results.append({
                    '日期': df.index[i].strftime('%Y-%m-%d'),
                    '代號': ticker,
                    '均線': ma_status,
                    'KD': kd_status,
                    '缺口': gap_status,
                    '收盤價': round(day_data['Close'], 2)
                })
        except:
            continue

    # 5. 存檔
    final_df = pd.DataFrame(results)
    final_df.to_csv("daily_scan.csv", index=False, encoding="utf-8-sig")
    print(f"🎉 掃描完成！已存入 daily_scan.csv，共 {len(final_df)} 筆資料。")

if __name__ == "__main__":
    run_scanner()
