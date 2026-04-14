import os
import pandas as pd
import yfinance as yf
import pandas_ta as ta

def run_scanner():
    print("🚀 啟動指標大滿貫掃描儀...")
    
    if not os.path.exists("stock_list.csv"): return
    df_list = pd.read_csv("stock_list.csv")
    
    # 1. 整理代號 (測試時我們跑前 100 檔，含 2330, 2317)
    test_list = ["2330.TW", "2317.TW", "2454.TW"]
    all_tickers = []
    for _, row in df_list.head(100).iterrows():
        s_id = str(row['stock_id'])
        m_type = str(row['type']).lower()
        if 'twse' in m_type: all_tickers.append(f"{s_id}.TW")
        elif 'tpex' in m_type: all_tickers.append(f"{s_id}.TWO")
    final_tickers = list(set(test_list + all_tickers))

    # 2. 下載股價
    data = yf.download(final_tickers, period="60d", group_by='ticker')
    
    results = []
    for ticker in final_tickers:
        try:
            df = data[ticker].copy().dropna()
            if len(df) < 35: continue

            # --- 計算指標 ---
            df['MA5'] = ta.sma(df['Close'], length=5)
            df['MA20'] = ta.sma(df['Close'], length=20)
            macd = ta.macd(df['Close'])
            df['MACD'] = macd['MACDH_12_26_9']
            rsi = ta.rsi(df['Close'], length=14)
            df['RSI'] = rsi
            kd = ta.stoch(df['High'], df['Low'], df['Close'])
            df['K'], df['D'] = kd['STOCKk_14_3_3'], kd['STOCKd_14_3_3']
            bb = ta.bbands(df['Close'], length=20)
            df['BBU'], df['BBL'] = bb['BBU_20_2.0'], bb['BBL_20_2.0']

            # --- 判斷 3 天的狀態 ---
            for i in range(-3, 0): # 倒數 3 天
                day = df.iloc[i]
                prev = df.iloc[i-1]
                
                res = {'日期': df.index[i].strftime('%Y-%m-%d'), '代號': ticker}
                res['均線'] = 1 if (day['Close'] > day['MA20'] and day['MA5'] > day['MA20']) else (-1 if (day['Close'] < day['MA20'] and day['MA5'] < day['MA20']) else 0)
                res['MACD'] = 1 if day['MACD'] > 0 else -1
                res['布林'] = 1 if day['Close'] > day['MA20'] else -1 # 簡化版：在中軸以上
                res['RSI'] = 1 if day['RSI'] > 50 else -1
                res['KD'] = 1 if day['K'] > day['D'] else -1
                res['下影線'] = 1 if (day['Low'] < min(day['Open'], day['Close'])) and (abs(day['Low'] - min(day['Open'], day['Close'])) > abs(day['Open'] - day['Close']) * 2) else 0
                res['缺口'] = 1 if day['Low'] > prev['High'] else (-1 if day['High'] < prev['Low'] else 0)
                res['法人'], res['融資'] = 0, 0 # 預留
                res['收盤價'] = round(float(day['Close']), 2)
                results.append(res)
            print(f"✅ {ticker} 完成")
        except: continue

    # 3. 存檔
    if results:
        pd.DataFrame(results).to_csv("daily_scan.csv", index=False, encoding="utf-8-sig")
        print("🎉 7 大技術指標（3日份）掃描完成！")

if __name__ == "__main__":
    run_scanner()
