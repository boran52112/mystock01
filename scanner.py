import os
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from FinMind.data import DataLoader

def run_scanner():
    print("🚀 啟動全市場掃描流程...")
    
    # 1. 讀取我們剛剛產出的「點名簿」
    if not os.path.exists("stock_list.csv"):
        print("❌ 找不到點名簿，請先執行先前的名單獲取程式。")
        return
    
    df_list = pd.read_csv("stock_list.csv")
    
    # 2. 幫代號加工（加上 .TW 或 .TWO）
    # 這段代碼是在做「字串拼接」，把數字變成國際通用的代號
    def format_id(row):
        if row['market_type'] == 'list':
            return f"{row['stock_id']}.TW"
        else:
            return f"{row['stock_id']}.TWO"
    
    df_list['full_id'] = df_list.apply(format_id, axis=1)
    all_tickers = df_list['full_id'].tolist()
    
    print(f"✅ 準備掃描 {len(all_tickers)} 檔標的...")

    # 3. 批次抓取股價 (核心重點！)
    # 我們一次抓最近 60 天的資料，這包含了我們要補齊的 3 天
    # group_by='ticker' 讓資料按股票排列
    print("⏳ 正在從 yfinance 批次抓取 60 天股價資料 (這可能需要 1-2 分鐘)...")
    data = yf.download(all_tickers, period="60d", interval="1d", group_by='ticker', threads=True)
    
    print("✅ 股價抓取完成！開始進行技術指標運算...")
    
    # 下一部分：我們將在這裡插入 9 大指標的計算邏輯
    # (我們分步來，先確認抓取功能正常)
    
    # 先隨便抓一檔看看有沒有資料 (例如台積電)
    sample_id = "2330.TW"
    if sample_id in data:
        print(f"成功抓取 {sample_id} 範例資料：")
        print(data[sample_id].tail(3)) # 顯示最後三天的股價

if __name__ == "__main__":
    run_scanner()
