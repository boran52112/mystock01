import os
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from FinMind.data import DataLoader

def run_scanner():
    print("🚀 啟動全市場掃描流程...")
    
    # 1. 讀取點名簿
    if not os.path.exists("stock_list.csv"):
        print("❌ 找不到點名簿，請先確認 stock_list.csv 是否存在。")
        return
    
    df_list = pd.read_csv("stock_list.csv")
    
    # 💡 這裡加一個檢查：印出所有的欄位名稱，萬一又錯了我們可以立刻抓到
    print(f"📊 目前清單中的欄位有: {df_list.columns.tolist()}")

    # 2. 幫代號加工（修正後的邏輯）
    # FinMind 的上市標記通常是 'twse'，上櫃是 'otc'
    def format_id(row):
        # 我們用 .get('type') 安全地拿取資料，並判斷它是上市還是上櫃
        m_type = str(row['type']).lower()
        s_id = str(row['stock_id'])
        
        if 'twse' in m_type:
            return f"{s_id}.TW"
        elif 'otc' in m_type:
            return f"{s_id}.TWO"
        else:
            return None # 如果不是這兩類就先不管它

    df_list['full_id'] = df_list.apply(format_id, axis=1)
    
    # 剔除掉那些我們無法辨認的代號
    df_list = df_list.dropna(subset=['full_id'])
    all_tickers = df_list['full_id'].tolist()
    
    # --- 關鍵節流動作 ---
    # 因為全市場 1800 檔下載太久，我們測試時先抓「前 50 檔」就好
    test_tickers = all_tickers[:50] 
    print(f"✅ 準備測試掃描前 50 檔標的...")

    # 3. 批次抓取股價
    print("⏳ 正在下載股價資料...")
    data = yf.download(test_tickers, period="60d", interval="1d", group_by='ticker', threads=True)
    
    print("✅ 股價抓取完成！")
    
    # 測試印出第一檔股票的資料
    first_stock = test_tickers[0]
    if first_stock in data:
        print(f"📈 範例 {first_stock} 最新股價：")
        print(data[first_stock].tail(1))

if __name__ == "__main__":
    run_scanner()
