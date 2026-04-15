import os
import pandas as pd
import yfinance as yf
import pandas_ta as ta

def run_scanner():
    print("🚀 啟動 9 大指標完整版掃描儀...")
    
    filename = "股票清單.csv"
    if not os.path.exists(filename):
        filename = "stock_list.csv" if os.path.exists("stock_list.csv") else None
        if not filename: return

    df_list = pd.read_csv(filename)
    def format_id(row):
        s_id, m_type = str(row['stock_id']), str(row['type']).lower()
        if len(s_id) > 4: return None
        return f"{s_id}.TW" if 'twse' in m_type else f"{s_id}.TWO"

    df_list['full_id'] = df_list.apply(format_id, axis=1)
    df_list = df_list.dropna(subset=['full_id'])
    
    # 掃描前 200 檔標的
    all_tickers = df_list['full_id'].tolist()[:200]
    data = yf.download(all_tickers, period="60d", group_by='ticker', threads=True)
    
    results = []
    for ticker in all_tickers:
        try:
            if ticker not in data or data[ticker].empty: continue
            df = data[ticker].copy().dropna()
            if len(df) < 35: continue

            # --- 計算指標 ---
            df['MA5'] = ta.sma(df['Close'], length=5)
            df['MA20'] = ta.sma(df['Close'], length=20)
            macd = ta.macd(df['Close'])
            df['MACD'] = macd['MACDH_12_26_9']
            df['RSI'] = ta.rsi(df['Close'], length=14)
            stoch = ta.stoch(df['High'], df['Low'], df['Close'])
            df['K'], df['D'] = stoch['STOCKk_14_3_3'], stoch['STOCKd_14_3_3']
            bb = ta.bbands(df['Close'], length=20)
            df['布林'] = bb['BBU_20_2.0']

            day, prev = df.iloc[-1], df.iloc[-2]
            
            # 建立 9 個指標的 1/-1/0 狀態
            res = {
                '日期': df.index[-1].strftime('%Y-%m-%d'),
                '代號': ticker,
                '均線': 1 if (day['Close'] > day['MA20'] and day['MA5'] > day['MA20']) else -1,
                'MACD': 1 if day['MACD'] > 0 else -1,
                '布林': 1 if day['Close'] > day['MA20'] else -1,
                'RSI': 1 if day['RSI'] > 50 else -1,
                'KD': 1 if day['K'] > day['D'] else -1,
                '下影線': 1 if (day['Low'] < min(day['Open'], day['Close'])) and (abs(day['Low'] - min(day['Open'], day['Close'])) > abs(day['Open'] - day['Close']) * 2) else 0,
                '缺口': 1 if day['Low'] > prev['High'] else (-1 if day['High'] < prev['Low'] else 0),
                '法人': 0, # 預留位
                '融資': 0, # 預留位
                '收盤價': round(float(day['Close']), 2)
            }
            results.append(res)
        except: continue

    if results:
        pd.DataFrame(results).to_csv("daily_scan.csv", index=False, encoding="utf-8-sig")
        print("🎉 掃描完成！")

if __name__ == "__main__":
    run_scanner()
