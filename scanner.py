import os
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import time
from datetime import datetime, timedelta
from FinMind.data import DataLoader

# --- 1. 初始化與環境設定 ---
FINMIND_TOKEN = os.getenv("FINMIND_TOKEN", "")
dl = DataLoader()
if FINMIND_TOKEN:
    dl.login_token(FINMIND_TOKEN)

# 硬性規定的 13 個欄位名稱
COLUMNS = [
    'Date', 'StockID', 'StockName', 'Close', 
    'MA_Signal', 'MACD_Signal', 'RSI_Signal', 'KD_Signal', 'BB_Signal',
    'Shadow_Signal', 'Gap_Signal', 'Inst_Signal', 'Margin_Signal'
]

# --- 2. 核心功能：動態獲取成交量前 200 名 ---

def get_top_200_stocks():
    """自動尋找最近一個有資料的交易日，抓取成交量前 200 名"""
    print("正在搜尋最近交易日之成交量排行...")
    for i in range(0, 10):  # 往回找 10 天
        target_date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
        try:
            # 獲取當日全市場統計
            df_vol = dl.taiwan_stock_daily_statistics(date=target_date)
            if not df_vol.empty:
                print(f"成功找到交易日資料: {target_date}")
                # 排序並取前 200 名
                top_200_df = df_vol.sort_values('total_volume', ascending=False).head(200)
                # 為了拿到中文名稱，需與 stock_info 合併
                df_info = dl.taiwan_stock_info()
                final_list = pd.merge(top_200_df[['stock_id']], df_info[['stock_id', 'stock_name']], on='stock_id', how='left')
                return final_list
        except Exception as e:
            continue
    return pd.DataFrame() # 若都沒找到則回傳空

# --- 3. 核心功能：計算 9 大指標 (確保回傳 13 欄位) ---

def compute_signals(df_yf, sid, sname, inst_data, margin_data):
    # 先建立一個全為 0 的樣板
    res = {col: 0 for col in COLUMNS}
    res['Date'] = datetime.now().strftime('%Y-%m-%d')
    res['StockID'] = sid
    res['StockName'] = sname
    
    if df_yf.empty or len(df_yf) < 30:
        return res
    
    try:
        close_series = df_yf['Close'].astype(float)
        high_series = df_yf['High'].astype(float)
        low_series = df_yf['Low'].astype(float)
        open_series = df_yf['Open'].astype(float)
        latest_close = close_series.iloc[-1]
        res['Close'] = round(latest_close, 2)

        # 1. 均線 MA
        ma5 = ta.sma(close_series, length=5)
        ma20 = ta.sma(close_series, length=20)
        res['MA_Signal'] = 1 if (ma5.iloc[-1] > ma20.iloc[-1] and latest_close > ma20.iloc[-1]) else -1

        # 2. MACD
        macd = ta.macd(close_series)
        res['MACD_Signal'] = 1 if macd['MACDs_12_26_9'].iloc[-1] > 0 else -1

        # 3. RSI
        rsi = ta.rsi(close_series, length=14)
        res['RSI_Signal'] = 1 if rsi.iloc[-1] > 50 else -1

        # 4. KD
        kd = ta.stoch(high_series, low_series, close_series)
        res['KD_Signal'] = 1 if kd['STOCHk_14_3_3'].iloc[-1] > kd['STOCHd_14_3_3'].iloc[-1] else -1

        # 5. BBands
        bb = ta.bbands(close_series, length=20)
        res['BB_Signal'] = 1 if latest_close > bb['BBM_20_2.0'].iloc[-1] else -1

        # 6. 下影線
        body = abs(latest_close - open_series.iloc[-1])
        lower_shadow = min(open_series.iloc[-1], latest_close) - low_series.iloc[-1]
        res['Shadow_Signal'] = 1 if lower_shadow > (body * 2) and body > 0 else 0

        # 7. 跳空
        res['Gap_Signal'] = 1 if low_series.iloc[-1] > high_series.iloc[-2] else (-1 if high_series.iloc[-1] < low_series.iloc[-2] else 0)

        # 8. 法人 (FinMind)
        if not inst_data.empty:
            net_buy = inst_data['buy'].sum() - inst_data['sell'].sum()
            res['Inst_Signal'] = 1 if net_buy > 0 else -1

        # 9. 融資 (FinMind)
        if len(margin_data) >= 2:
            res['Margin_Signal'] = 1 if margin_data['MarginPurchaseStock'].iloc[-1] < margin_data['MarginPurchaseStock'].iloc[-2] else -1

    except Exception as e:
        print(f"指標計算錯誤 {sid}: {e}")
        
    return res

# --- 4. 主程式執行 ---

def main():
    print(f"--- 啟動掃描器 v3.2 (修復版) ---")
    
    # A. 獲取前 200 檔股票 (成交量排行)
    target_stocks = get_top_200_stocks()
    
    if target_stocks.empty:
        print("致命錯誤：無法獲取股票清單，請檢查 API Token 或網路狀態。")
        return

    print(f"成功獲取 {len(target_stocks)} 檔股票清單，開始進行指標掃描...")
    
    all_results = []
    today_str = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=50)).strftime('%Y-%m-%d')

    for index, row in target_stocks.iterrows():
        sid = row['stock_id']
        sname = row['stock_name']
        
        try:
            print(f"[{index+1}/200] 正在分析: {sid} {sname}")
            
            # 抓取 yfinance 資料 (技術指標用)
            df_yf = yf.download(f"{sid}.TW", start=start_date, progress=False)
            
            # 抓取 FinMind 資料 (籌碼用)
            inst_data = pd.DataFrame()
            margin_data = pd.DataFrame()
            if FINMIND_TOKEN:
                try:
                    inst_data = dl.taiwan_stock_institutional_investors(stock_id=sid, start_date=today_str)
                    margin_data = dl.taiwan_stock_margin_purchase_short_sale(stock_id=sid, start_date=start_date)
                    time.sleep(0.3) # 避免過快被封鎖
                except:
                    pass

            # 計算所有訊號
            result = compute_signals(df_yf, sid, sname, inst_data, margin_data)
            all_results.append(result)
            
        except Exception as e:
            print(f"處理 {sid} 時發生非預期錯誤: {e}")

    # B. 存檔邏輯
    new_df = pd.DataFrame(all_results)
    file_path = 'daily_scan.csv'
    
    # 這裡直接產出檔案，不再進行複雜的舊檔合併，確保格式 100% 正確
    new_df.to_csv(file_path, index=False, encoding='utf-8-sig')
    print(f"--- 執行成功！共掃描 {len(new_df)} 檔股票，CSV 檔案已生成 ---")

if __name__ == "__main__":
    main()
