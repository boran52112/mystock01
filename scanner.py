import os
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from FinMind.data import DataLoader
from datetime import datetime, timedelta
import time

def run_scanner():
    print("🚀 啟動「防撞強化版」掃描儀...")
    
    # 1. 讀取名單
    if not os.path.exists("stock_list.csv"): return
    df_list = pd.read_csv("stock_list.csv")
    
    def format_id(row):
        s_id, m_type = str(row['stock_id']), str(row['type']).lower()
        if len(s_id) > 4: return None
        return f"{s_id}.TW" if 'twse' in m_type else f"{s_id}.TWO"

    df_list['full_id'] = df_list.apply(format_id, axis=1)
    df_list = df_list.dropna(subset=['full_id'])
    all_tickers = df_list['full_id'].tolist()
    
    # 2. 準備抓取股價 (yfinance 通常很穩，先抓)
    print(f"⏳ 正在抓取 {len(all_tickers)} 檔股價資料...")
    data = yf.download(all_tickers, period="60d", group_by='ticker', threads=True)

    # 3. 準備籌碼資料箱子
    token = os.getenv('FINMIND_TOKEN')
    dl = DataLoader()
    if token: dl.login(token)
    
    # 我們試著抓最近 3 個交易日的籌碼
    # 這裡我們用「安全模式」：一小塊一小塊要
    target_dates = []
    # 簡單找出最近的有交易的 3 天 (從 yfinance 的資料找最準)
    sample_df = data[all_tickers[0]]
    if not sample_df.empty:
        target_dates = sample_df.index[-3:].strftime('%Y-%m-%d').tolist()

    # 建立一個大存錢筒來放籌碼資料
    all_inst_data = pd.DataFrame()
    
    print(f"⏳ 嘗試抓取籌碼日期: {target_dates}")
    for d in target_dates:
        try:
            print(f"  正在抓取 {d} 的法人資料...")
            # 這裡我們只抓「那一天」的資料
            d_data = dl.taiwan_stock_institutional_investors(start_date=d, end_date=d)
            all_inst_data = pd.concat([all_inst_data, d_data])
            time.sleep(2) # 休息一下，不要讓 API 覺得我們很煩
        except:
            print(f"  ⚠️ {d} 籌碼資料抓取失敗，跳過...")

    # 4. 開始合併計算
    results = []
    for ticker in all_tickers:
        try:
            df = data[ticker].copy().dropna()
            if len(df) < 35: continue
            
            # 計算技術指標 (維持昨天的公式)
            df['MA20'] = ta.sma(df['Close'], length=20)
            df['MA5'] = ta.sma(df['Close'], length=5)
            
            stock_id = ticker.split('.')[0]
            for i in range(-3, 0):
                d_str = df.index[i].strftime('%Y-%m-%d')
                
                # 籌碼判斷 (如果剛才沒抓到資料，就預設為 0)
                inst_status = 0
                if not all_inst_data.empty:
                    match = all_inst_data[(all_inst_data['date'] == d_str) & (all_inst_data['stock_id'] == stock_id)]
                    if not match.empty:
                        inst_sum = match['buy'].sum() - match['sell'].sum()
                        inst_status = 1 if inst_sum > 0 else -1

                results.append({
                    '日期': d_str,
                    '代號': ticker,
                    '均線': 1 if df.iloc[i]['Close'] > df.iloc[i]['MA20'] else -1,
                    '法人': inst_status,
                    '收盤價': round(float(df.iloc[i]['Close']), 2)
                })
        except: continue

    # 5. 存檔
    if results:
        pd.DataFrame(results).to_csv("daily_scan.csv", index=False, encoding="utf-8-sig")
        print(f"🎉 成功！產出 {len(results)} 筆資料。")
    else:
        print("❌ 失敗：完全沒有資料產出。")

if __name__ == "__main__":
    run_scanner()
