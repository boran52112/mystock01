import os
import pandas as pd
import pandas_ta as ta
import time
from datetime import datetime, timedelta
from FinMind.data import DataLoader

# --- 1. 設定區 ---
FINMIND_TOKEN = os.getenv("FINMIND_TOKEN", "")
dl = DataLoader()
if FINMIND_TOKEN:
    dl.login_token(FINMIND_TOKEN)

COLUMNS = [
    'Date', 'StockID', 'StockName', 'Close', 
    'MA_Signal', 'MACD_Signal', 'RSI_Signal', 'KD_Signal', 'BB_Signal',
    'Shadow_Signal', 'Gap_Signal', 'Inst_Signal', 'Margin_Signal'
]

# 備援清單 (確保至少有基本股可掃)
BACKUP_IDS = ["2330","2317","2454","2308","2382","2303","2881","2882","3008","2603"]

# --- 2. 核心功能：獲取排行與資料清洗 ---

def get_target_list():
    """獲取成交量前 200 名排行，並去重"""
    print("正在向 FinMind 索取成交量排行榜...")
    try:
        df_info = dl.taiwan_stock_info()
        for i in range(0, 7):
            t_date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
            df_vol = dl.taiwan_stock_daily_statistics(date=t_date)
            if not df_vol.empty:
                print(f"✅ 成功獲取 {t_date} 排行。")
                top_200 = df_vol.sort_values('total_volume', ascending=False).head(200)
                return pd.merge(top_200[['stock_id']], df_info[['stock_id', 'stock_name']], on='stock_id', how='left')
    except:
        pass
    print("⚠️ 排行榜獲取失敗，使用基本備援清單。")
    return pd.DataFrame([{"stock_id": i, "stock_name": i} for i in BACKUP_IDS])

def normalize_df(df):
    """將 FinMind 格式翻譯成指標庫需要的格式"""
    rename_map = {
        'date': 'Date', 'open': 'Open', 'max': 'High', 
        'min': 'Low', 'close': 'Close', 'Assistant': 'Volume'
    }
    df = df.rename(columns=rename_map)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    return df

def compute_signals(df, sid, sname, inst_data, margin_data):
    """計算 9 大指標"""
    res = {col: 0 for col in COLUMNS}
    if df.empty or len(df) < 20: return None
    
    try:
        actual_date = df.index[-1].strftime('%Y-%m-%d')
        close_p = float(df['Close'].iloc[-1])
        res.update({'Date': actual_date, 'StockID': sid, 'StockName': sname, 'Close': round(close_p, 2)})
        
        # 技術指標
        ma5, ma20 = ta.sma(df['Close'], 5), ta.sma(df['Close'], 20)
        res['MA_Signal'] = 1 if (ma5.iloc[-1] > ma20.iloc[-1] and close_p > ma20.iloc[-1]) else -1
        macd = ta.macd(df['Close'])
        res['MACD_Signal'] = 1 if macd['MACDs_12_26_9'].iloc[-1] > 0 else -1
        rsi = ta.rsi(df['Close'], 14)
        res['RSI_Signal'] = 1 if rsi.iloc[-1] > 50 else -1
        kd = ta.stoch(df['High'], df['Low'], df['Close'])
        res['KD_Signal'] = 1 if kd['STOCHk_14_3_3'].iloc[-1] > kd['STOCHd_14_3_3'].iloc[-1] else -1
        bb = ta.bbands(df['Close'], 20)
        res['BB_Signal'] = 1 if close_p > bb['BBM_20_2.0'].iloc[-1] else -1
        
        # 下影線與跳空
        body = abs(close_p - df['Open'].iloc[-1])
        shadow = min(df['Open'].iloc[-1], close_p) - df['Low'].iloc[-1]
        res['Shadow_Signal'] = 1 if shadow > (body * 2) and body > 0 else 0
        res['Gap_Signal'] = 1 if df['Low'].iloc[-1] > df['High'].iloc[-2] else (-1 if df['High'].iloc[-1] < df['Low'].iloc[-2] else 0)
        
        # 籌碼 (抓取最近 3 天合計)
        if not inst_data.empty:
            res['Inst_Signal'] = 1 if (inst_data['buy'].sum() - inst_data['sell'].sum()) > 0 else -1
        if len(margin_data) >= 2:
            res['Margin_Signal'] = 1 if margin_data['MarginPurchaseStock'].iloc[-1] < margin_data['MarginPurchaseStock'].iloc[-2] else -1
            
        return res
    except:
        return None

# --- 3. 主程式 ---

def main():
    print(f"--- 啟動掃描器 v3.7 (FinMind 一條龍版) ---")
    target_stocks = get_target_list()
    target_stocks = target_stocks.drop_duplicates(subset=['stock_id']) # 徹底去重
    
    all_results = []
    # 籌碼抓取區間 (最近 3 天)
    chip_start = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
    # 股價抓取區間 (最近 50 天)
    price_start = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')

    for index, row in target_stocks.iterrows():
        sid, sname = str(row['stock_id']), str(row['stock_name'])
        try:
            # 步驟三：Token 控管，前 50 檔優先
            print(f"[{index+1}/{len(target_stocks)}] 掃描: {sid} {sname}")
            
            # A. 抓取股價
            df_raw = dl.taiwan_stock_daily(stock_id=sid, start_date=price_start)
            df = normalize_df(df_raw)
            
            # B. 抓取籌碼 (法人 + 融資)
            inst_data = dl.taiwan_stock_institutional_investors(stock_id=sid, start_date=chip_start)
            margin_data = dl.taiwan_stock_margin_purchase_short_sale(stock_id=sid, start_date=chip_start)
            
            # C. 計算
            res = compute_signals(df, sid, sname, inst_data, margin_data)
            if res: all_results.append(res)
            
            # 避免請求過快
            time.sleep(0.1)
            
        except Exception as e:
            # 檢查是否為 Token 耗盡 (429 錯誤或相似訊息)
            if "429" in str(e) or "limit" in str(e).lower():
                print(f"⚠️ Token 額度已達上限，停止掃描並準備存檔。")
                break
            print(f"跳過 {sid}: 資料暫時無法獲取")
            continue

    # --- 4. 存檔與合併 (細節 B: 累積歷史) ---
    if all_results:
        new_df = pd.DataFrame(all_results)
        file_path = 'daily_scan.csv'
        
        if os.path.exists(file_path):
            try:
                old_df = pd.read_csv(file_path)
                # 合併並去除重複 (以 Date+StockID 為準，保留最新掃描結果)
                final_df = pd.concat([old_df, new_df]).drop_duplicates(subset=['Date', 'StockID'], keep='last')
            except:
                final_df = new_df
        else:
            final_df = new_df
        
        # 只保留最近 3 個交易日的資料
        keep_dates = sorted(final_df['Date'].unique(), reverse=True)[:3]
        final_df = final_df[final_df['Date'].isin(keep_dates)]
        
        final_df.to_csv(file_path, index=False, encoding='utf-8-sig')
        print(f"✅ 完成！'daily_scan.csv' 已更新，共保留 {len(final_df)} 筆歷史紀錄。")
    else:
        print("❌ 未抓取到任何資料。")

if __name__ == "__main__":
    main()
