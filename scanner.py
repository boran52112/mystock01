import os
import pandas as pd
import yfinance as yf
import pandas_ta as ta

def run_scanner():
    print("🚀 --- 偵錯版掃描儀啟動 ---")
    
    # 1. 檢查檔案
    if not os.path.exists("stock_list.csv"):
        print("❌ 找不到 stock_list.csv")
        return
    
    # 2. 強制加入測試標的，確保一定有資料可以跑
    # 我們抓：台積電, 鴻海, 聯發科, 加上原清單的前 10 檔
    df_list = pd.read_csv("stock_list.csv")
    all_tickers = []
    
    # 這裡手動加入確保成功的代號
    test_list = ["2330.TW", "2317.TW", "2454.TW"]
    print(f"💡 加入核心測試標的: {test_list}")
    
    # 解析其餘代號 (增加診斷印出)
    for _, row in df_list.head(20).iterrows():
        s_id = str(row['stock_id'])
        m_type = str(row['type']).lower()
        if 'twse' in m_type: all_tickers.append(f"{s_id}.TW")
        elif 'tpex' in m_type: all_tickers.append(f"{s_id}.TWO")
    
    final_tickers = list(set(test_list + all_tickers))
    print(f"📋 最終點名清單: {final_tickers}")

    # 3. 抓取股價
    print("⏳ 正在下載股價資料...")
    data = yf.download(final_tickers, period="60d", group_by='ticker')
    
    results = []
    for ticker in final_tickers:
        try:
            # 這裡用比較安全的方式拿資料
            df = data[ticker].copy()
            df = df.dropna()
            
            if df.empty or len(df) < 10:
                print(f"⏩ {ticker}: 資料不足，跳過")
                continue

            # 計算一個最簡單的均線做測試
            df['MA5'] = df['Close'].rolling(window=5).mean()
            
            # 拿最後一天的資料
            last_day = df.iloc[-1]
            status = 1 if last_day['Close'] > last_day['MA5'] else -1
            
            results.append({
                '日期': df.index[-1].strftime('%Y-%m-%d'),
                '代號': ticker,
                '均線狀態': status,
                '收盤價': round(float(last_day['Close']), 2)
            })
            print(f"✅ {ticker}: 計算成功！(收盤價: {last_day['Close']})")
        except Exception as e:
            print(f"❌ {ticker}: 出錯了 -> {e}")

    # 4. 存檔
    if results:
        final_df = pd.DataFrame(results)
        final_df.to_csv("daily_scan.csv", index=False, encoding="utf-8-sig")
        print(f"🎉 成功！產出 {len(final_df)} 筆資料到 daily_scan.csv")
    else:
        print("⚠️ 警告：所有的標全都失敗了，請檢查 yfinance 連線。")

if __name__ == "__main__":
    run_scanner()
