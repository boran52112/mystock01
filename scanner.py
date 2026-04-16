import os
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import time
from datetime import datetime, timedelta
from FinMind.data import DataLoader

# --- 1. 設定區 ---
FINMIND_TOKEN = os.getenv("FINMIND_TOKEN", "") # 從環境變數讀取
dl = DataLoader()
if FINMIND_TOKEN:
    dl.login_token(FINMIND_TOKEN)

# 硬性規定的 13 個欄位 (Schema)
COLUMNS = [
    'Date', 'StockID', 'StockName', 'Close', 
    'MA_Signal', 'MACD_Signal', 'RSI_Signal', 'KD_Signal', 'BB_Signal',
    'Shadow_Signal', 'Gap_Signal', 'Inst_Signal', 'Margin_Signal'
]

# 精選股票清單 (以 0050, 0056, 00878 成份股及高成交量股為核心，共約 200-250 檔)
# 這裡先預設常用代號，實務上可動態調整
BASE_STOCK_LIST = [
    "2330", "2317", "2454", "2308", "2382", "2303", "2881", "2882", "3008", "2603",
    "2609", "2615", "2357", "3231", "2376", "6669", "2408", "2409", "3481", "3037"
    # ... 此處為節省篇幅縮寫，程式執行時會先抓取台股前 200 大成交量股
]

# --- 2. 輔助函式 ---

def get_stock_list():
    """從 FinMind 獲取今日成交量前 200 名的股票"""
    try:
        # 抓取台股基本資訊（包含名稱）
        df_info = dl.taiwan_stock_info()
        # 獲取昨日成交量（作為篩選參考）
        yesterday = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
        df_vol = dl.taiwan_stock_daily_statistics(date=yesterday)
        
        if df_vol.empty:
            # 如果沒抓到統計資料，回傳預設核心股
            return df_info[df_info['stock_id'].isin(BASE_STOCK_LIST)]
            
        top_200 = df_vol.sort_values('total_volume', ascending=False).head(200)['stock_id'].tolist()
        return df_info[df_info['stock_id'].isin(top_200)]
    except:
        return pd.DataFrame(columns=['stock_id', 'stock_name'])

def compute_signals(df, stock_id, stock_name, inst_data, margin_data):
    """計算 9 大指標，確保回傳 13 個欄位"""
    # 初始化回傳字典
    res = {col: 0 for col in COLUMNS}
    res['Date'] = datetime.now().strftime('%Y-%m-%d')
    res['StockID'] = stock_id
    res['StockName'] = stock_name
    
    if len(df) < 30: return res # 資料不足
    
    res['Close'] = round(df['Close'].iloc[-1], 2)
    
    # 1. MA (5MA > 20MA & Close > 20MA)
    ma5 = ta.sma(df['Close'], length=5)
    ma20 = ta.sma(df['Close'], length=20)
    res['MA_Signal'] = 1 if (ma5.iloc[-1] > ma20.iloc[-1] and df['Close'].iloc[-1] > ma20.iloc[-1]) else -1

    # 2. MACD (OSC > 0)
    macd = ta.macd(df['Close'])
    res['MACD_Signal'] = 1 if macd['MACDs_12_26_9'].iloc[-1] > 0 else -1

    # 3. RSI (RSI > 50)
    rsi = ta.rsi(df['Close'], length=14)
    res['RSI_Signal'] = 1 if rsi.iloc[-1] > 50 else -1

    # 4. KD (K > D)
    kd = ta.stoch(df['High'], df['Low'], df['Close'])
    res['KD_Signal'] = 1 if kd['STOCHk_14_3_3'].iloc[-1] > kd['STOCHd_14_3_3'].iloc[-1] else -1

    # 5. BB (Price > Middle)
    bb = ta.bbands(df['Close'], length=20)
    res['BB_Signal'] = 1 if df['Close'].iloc[-1] > bb['BBM_20_2.0'].iloc[-1] else -1

    # 6. 下影線 (下影線 > 實體 2 倍)
    body = abs(df['Close'].iloc[-1] - df['Open'].iloc[-1])
    lower_shadow = min(df['Open'].iloc[-1], df['Close'].iloc[-1]) - df['Low'].iloc[-1]
    res['Shadow_Signal'] = 1 if lower_shadow > (body * 2) and body > 0 else 0

    # 7. 跳空 (今日最低 > 昨日最高)
    res['Gap_Signal'] = 1 if df['Low'].iloc[-1] > df['High'].iloc[-2] else (-1 if df['High'].iloc[-1] < df['Low'].iloc[-2] else 0)

    # 8. 法人籌碼 (三大法人合計買超)
    if not inst_data.empty:
        total_buy = inst_data['buy'].sum() - inst_data['sell'].sum()
        res['Inst_Signal'] = 1 if total_buy > 0 else -1

    # 9. 融資籌碼 (融資餘額減少為多)
    if len(margin_data) >= 2:
        res['Margin_Signal'] = 1 if margin_data['MarginPurchaseStock'].iloc[-1] < margin_data['MarginPurchaseStock'].iloc[-2] else -1

    return res

# --- 3. 主程式 ---

def main():
    print("啟動掃描器 v3.0...")
    target_stocks = get_stock_list()
    if target_stocks.empty:
        print("無法獲取股票清單，終止。")
        return

    all_results = []
    today_str = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=40)).strftime('%Y-%m-%d')

    for index, row in target_stocks.iterrows():
        sid = row['stock_id']
        sname = row['stock_name']
        print(f"正在分析: {sid} {sname}...")
        
        try:
            # A. 技術指標資料 (yfinance)
            df = yf.download(f"{sid}.TW", start=start_date, progress=False)
            
            # B. 籌碼資料 (FinMind) - 設置 try-except 避免 Token 耗盡導致崩潰
            inst_data = pd.DataFrame()
            margin_data = pd.DataFrame()
            if FINMIND_TOKEN:
                try:
                    inst_data = dl.taiwan_stock_institutional_investors(stock_id=sid, start_date=today_str)
                    margin_data = dl.taiwan_stock_margin_purchase_short_sale(stock_id=sid, start_date=start_date)
                    time.sleep(0.5) # 稍微降載
                except:
                    print(f"FinMind Token 可能耗盡，跳過籌碼分析: {sid}")

            # C. 計算
            signal_res = compute_signals(df, sid, sname, inst_data, margin_data)
            all_results.append(signal_res)
            
        except Exception as e:
            print(f"處理 {sid} 時發生錯誤: {e}")
        
        # 每處理 20 檔休息一下，防 API 封鎖
        if index % 20 == 0: time.sleep(2)

    # --- 4. 存檔與合併 ---
    new_df = pd.DataFrame(all_results)
    
    # 讀取舊檔
    file_path = 'daily_scan.csv'
    if os.path.exists(file_path):
        old_df = pd.read_csv(file_path)
        # 合併並去重 (以 Date + StockID 為準)
        final_df = pd.concat([old_df, new_df]).drop_duplicates(subset=['Date', 'StockID'], keep='last')
    else:
        final_df = new_df

    # 僅保留最近 3 天的歷史資料
    dates = sorted(final_df['Date'].unique(), reverse=True)[:3]
    final_df = final_df[final_df['Date'].isin(dates)]
    
    # 存檔
    final_df.to_csv(file_path, index=False, encoding='utf-8-sig')
    print(f"掃描完成，資料已存入 {file_path}，共 {len(new_df)} 筆。")

if __name__ == "__main__":
    main()
