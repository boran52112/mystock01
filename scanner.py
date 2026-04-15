import os
import pandas as pd
import yfinance as yf
import pandas_ta as ta

def run_scanner():
    print("🚀 啟動 9 大指標完整版掃描儀...")
    
    # 1. 讀取點名簿
    filename = "股票清單.csv"
    if not os.path.exists(filename):
        if os.path.exists("stock_list.csv"): filename = "stock_list.csv"
        else: return

    df_list = pd.read_csv(filename)
    
    def format_id(row):
        s_id, m_type = str(row['stock_id']), str(row['type']).lower()
        if len(s_id) > 4: return None
        return f"{s_id}.TW" if 'twse' in m_type else f"{s_id}.TWO"

    df_list['full_id'] = df_list.apply(format_id, axis=1)
    df_list = df_list.dropna(subset=['full_id'])
    
    # 跑前 200 檔確保效能與資料完整性
    all_tickers = df_list['full_id'].tolist()[:200]
    data = yf.download(all_tickers, period="60d", group_by='ticker', threads=True)
    
    results = []
    for ticker in all_tickers:
        try:
            if ticker not in data or data[ticker].empty: continue
            df = data[ticker].copy().dropna()
            if len(df) < 35: continue

            # 計算指標
            df['MA5'] = ta.sma(df['Close'], length=5)
            df['MA20'] = ta.sma(df['Close'], length=20)
            macd = ta.macd(df['Close'])
            df['MACD'] = macd['MACDH_12_26_9']
            df['RSI'] = ta.rsi(df['Close'], length=14)
            kd = ta.stoch(df['High'], df['Low'], df['Close'])
            df['K'], df['D'] = kd['STOCKk_14_3_3'], kd['STOCKd_14_3_3']
            bb = ta.bbands(df['Close'], length=20)
            df['BBU'] = bb['BBU_20_2.0']

            # 抓最後一天
            day = df.iloc[-1]
            prev = df.iloc[-2]
            
            # 判定邏輯 (1=多, -1=空, 0=中立)
            res = {'日期': df.index[-1].strftime('%Y-%m-%d'), '代號': ticker}
            res['均線'] = 1 if (day['Close'] > day['MA20'] and day['MA5'] > day['MA20']) else -1
            res['MACD'] = 1 if day['MACD'] > 0 else -1
            res['布林'] = 1 if day['Close'] > day['MA20'] else -1
            res['RSI'] = 1 if day['RSI'] > 50 else -1
            res['KD'] = 1 if day['K'] > day['D'] else -1
            res['下影線'] = 1 if (day['Low'] < min(day['Open'], day['Close'])) and (abs(day['Low'] - min(day['Open'], day['Close'])) > abs(day['Open'] - day['Close']) * 2) else 0
            res['缺口'] = 1 if day['Low'] > prev['High'] else (-1 if day['High'] < prev['Low'] else 0)
            res['法人'], res['融資'] = 0, 0 # 籌碼面預留位
            res['收盤價'] = round(float(day['Close']), 2)
            results.append(res)
        except: continue

    if results:
        pd.DataFrame(results).to_csv("daily_scan.csv", index=False, encoding="utf-8-sig")
        print("🎉 掃描成功，CSV 已更新。")

if __name__ == "__main__":
    run_scanner()
