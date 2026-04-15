import os
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from FinMind.data import DataLoader
from datetime import datetime, timedelta

def run_scanner():
    print("🚀 啟動「全指標」雲端掃描儀...")
    
    # 1. 讀取清單
    if not os.path.exists("stock_list.csv"): 
        print("❌ 找不到點名簿")
        return
    df_list = pd.read_csv("stock_list.csv")
    
    # 整理代號 (這次我們挑戰全市場，不限制 100 檔了！)
    def format_id(row):
        s_id, m_type = str(row['stock_id']), str(row['type']).lower()
        if len(s_id) > 4: return None # 暫時只抓 4 碼的純股票
        return f"{s_id}.TW" if 'twse' in m_type else f"{s_id}.TWO"

    df_list['full_id'] = df_list.apply(format_id, axis=1)
    df_list = df_list.dropna(subset=['full_id'])
    all_tickers = df_list['full_id'].tolist()
    
    # 2. 獲取籌碼數據 (利用 Bulk Request 節省額度)
    token = os.getenv('FINMIND_TOKEN')
    dl = DataLoader()
    if token: dl.login(token)

    # 找出最近三個交易日 (簡單邏輯：抓最近 5 天，通常含 3 個交易日)
    today = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
    
    print(f"⏳ 正在抓取全市場籌碼資料 ({start_date} ~ {today})...")
    # 抓法人
    inst_data = dl.taiwan_stock_institutional_investors(start_date=start_date)
    # 抓融資
    margin_data = dl.taiwan_stock_margin_purchase_short_sale(start_date=start_date)
    
    print("✅ 籌碼資料獲取成功，開始計算技術指標...")

    # 3. 抓取股價 (全市場批次抓取)
    data = yf.download(all_tickers, period="60d", group_by='ticker', threads=True)
    
    results = []
    for ticker in all_tickers:
        try:
            if ticker not in data or data[ticker].empty: continue
            df = data[ticker].copy().dropna()
            if len(df) < 35: continue

            # 計算 7 大技術指標 (MA, MACD, RSI, KD, BB, 下影線, 缺口)
            df['MA20'] = ta.sma(df['Close'], length=20)
            df['MA5'] = ta.sma(df['Close'], length=5)
            # (中間省略技術指標計算，維持昨天邏輯...)
            
            stock_id = ticker.split('.')[0]
            
            for i in range(-3, 0): # 抓最後 3 天
                date_str = df.index[i].strftime('%Y-%m-%d')
                
                # --- 關鍵：從大表中找出這檔股票當天的籌碼 ---
                # 篩選法人
                day_inst = inst_data[(inst_data['date'] == date_str) & (inst_data['stock_id'] == stock_id)]
                inst_sum = day_inst['buy'].sum() - day_inst['sell'].sum()
                inst_status = 1 if inst_sum > 0 else -1
                
                # 篩選融資
                day_margin = margin_data[(margin_data['date'] == date_str) & (margin_data['stock_id'] == stock_id)]
                # 融資減少通常視為看多 (1), 融資增加看空 (-1)
                margin_diff = day_margin['MarginPurchaseBuy'].sum() - day_margin['MarginPurchaseSell'].sum()
                margin_status = 1 if margin_diff < 0 else -1

                results.append({
                    '日期': date_str,
                    '代號': ticker,
                    '均線': 1 if (df.iloc[i]['Close'] > df.iloc[i]['MA20']) else -1,
                    '法人': inst_status,
                    '融資': margin_status,
                    '收盤價': round(float(df.iloc[i]['Close']), 2)
                })
        except: continue

    # 4. 存檔
    if results:
        pd.DataFrame(results).to_csv("daily_scan.csv", index=False, encoding="utf-8-sig")
        print(f"🎉 全市場掃描完成！產出 {len(results)} 筆資料。")

if __name__ == "__main__":
    run_scanner()
