import os
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import time
from datetime import datetime, timedelta
from FinMind.data import DataLoader

# --- 1. 初始化 ---
FINMIND_TOKEN = os.getenv("FINMIND_TOKEN", "")
dl = DataLoader()
if FINMIND_TOKEN:
    dl.login_token(FINMIND_TOKEN)

COLUMNS = [
    'Date', 'StockID', 'StockName', 'Close', 
    'MA_Signal', 'MACD_Signal', 'RSI_Signal', 'KD_Signal', 'BB_Signal',
    'Shadow_Signal', 'Gap_Signal', 'Inst_Signal', 'Margin_Signal'
]

# --- 2. 備援名單 (200 檔核心股) ---
# 包含 0050, 0056, 00878 成份股，確保 API 故障時也能覆蓋主要飆股
BACKUP_IDS = [
    "2330","2317","2454","2308","2382","2303","2881","2882","3008","2603","2609","2615","2357","3231","2376","6669","2408","2409","3481","3037",
    "2324","2353","2356","2377","2379","2383","2401","2449","2451","3034","3035","3044","3443","3532","3711","4919","4938","4958","4961","6176",
    "6213","6239","6415","8046","8210","1101","1102","1216","1301","1303","1326","1402","1503","1504","1513","1519","1605","1722","1802","2002",
    "2006","2105","2201","2204","2206","2347","2501","2542","2606","2610","2618","2707","2801","2809","2812","2834","2880","2883","2884","2885",
    "2886","2887","2888","2890","2891","2892","2912","5871","5876","5880","6505","8454","9904","9910","9921","9945","1476","9933","3017","2360",
    "2395","3661","6515","2345","3013","2368","1560","2615","2603","2609","3045","3036","2474","2354","2352","1504","1514","2371","6285","2458",
    "3533","6121","6223","5347","5483","6488","8069","3105","3264","6274","3293","4966","8069","6462","3653","3680","5274","6231","8299","3324",
    "3227","6147","4966","3131","3529","3545","3592","6510","6414","3406","4958","6446","1795","4147","6547","4743","1760","6472","1789","4105",
    "1513","1519","1514","1503","1605","1608","1609","1611","1618","2618","2610","2633","2634","2637","2707","2727","5284","2207","2201","2204"
]

# --- 3. 輔助函式 ---

def get_clean_float(val):
    """處理 yfinance 常見的字串雜訊，強制轉回純數字"""
    try:
        if isinstance(val, (int, float)): return float(val)
        # 移除 Ticker 相關雜訊字串
        s = str(val).split()[-1]
        return float(s)
    except:
        return 0.0

def get_top_200_stocks():
    """優先抓取成交量排行，失敗則使用備援 ID 並補有名稱"""
    print("正在嘗試獲取成交量前 200 名排行...")
    df_info = pd.DataFrame()
    try:
        df_info = dl.taiwan_stock_info()
    except:
        print("FinMind 基本資訊獲取失敗。")

    for i in range(0, 7):
        target_date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
        try:
            df_vol = dl.taiwan_stock_daily_statistics(date=target_date)
            if not df_vol.empty:
                print(f"✅ 成功獲取 {target_date} 排行資料。")
                top_200 = df_vol.sort_values('total_volume', ascending=False).head(200)
                if not df_info.empty:
                    return pd.merge(top_200[['stock_id']], df_info[['stock_id', 'stock_name']], on='stock_id', how='left')
                else:
                    top_200['stock_name'] = top_200['stock_id']
                    return top_200[['stock_id', 'stock_name']]
        except:
            continue
    
    print("⚠️ 切換至 200 檔備援名單模式...")
    backup_df = pd.DataFrame({"stock_id": BACKUP_IDS})
    if not df_info.empty:
        return pd.merge(backup_df, df_info[['stock_id', 'stock_name']], on='stock_id', how='left')
    else:
        backup_df['stock_name'] = backup_df['stock_id']
        return backup_df

def compute_signals(df_yf, sid, sname, inst_data, margin_data):
    res = {col: 0 for col in COLUMNS}
    res.update({'Date': datetime.now().strftime('%Y-%m-%d'), 'StockID': sid, 'StockName': sname})
    
    if df_yf.empty or len(df_yf) < 30: return res
    
    try:
        # 強制數據清洗：確保收盤價是純數字
        close_price = get_clean_float(df_yf['Close'].iloc[-1])
        res['Close'] = round(close_price, 2)
        
        c = df_yf['Close'].astype(float)
        # 1. 均線 MA
        ma5, ma20 = ta.sma(c, 5), ta.sma(c, 20)
        res['MA_Signal'] = 1 if (ma5.iloc[-1] > ma20.iloc[-1] and close_price > ma20.iloc[-1]) else -1
        # 2. MACD
        macd = ta.macd(c)
        res['MACD_Signal'] = 1 if macd['MACDs_12_26_9'].iloc[-1] > 0 else -1
        # 3. RSI
        rsi = ta.rsi(c, 14)
        res['RSI_Signal'] = 1 if rsi.iloc[-1] > 50 else -1
        # 4. KD
        kd = ta.stoch(df_yf['High'].astype(float), df_yf['Low'].astype(float), c)
        res['KD_Signal'] = 1 if kd['STOCHk_14_3_3'].iloc[-1] > kd['STOCHd_14_3_3'].iloc[-1] else -1
        # 5. BB
        bb = ta.bbands(c, 20)
        res['BB_Signal'] = 1 if close_price > bb['BBM_20_2.0'].iloc[-1] else -1
        # 6. 下影線
        body = abs(close_price - get_clean_float(df_yf['Open'].iloc[-1]))
        low_shadow = min(get_clean_float(df_yf['Open'].iloc[-1]), close_price) - get_clean_float(df_yf['Low'].iloc[-1])
        res['Shadow_Signal'] = 1 if low_shadow > (body * 2) and body > 0 else 0
        # 7. 跳空
        res['Gap_Signal'] = 1 if get_clean_float(df_yf['Low'].iloc[-1]) > get_clean_float(df_yf['High'].iloc[-2]) else (-1 if get_clean_float(df_yf['High'].iloc[-1]) < get_clean_float(df_yf['Low'].iloc[-2]) else 0)
        
        # 籌碼 (FinMind)
        if not inst_data.empty:
            res['Inst_Signal'] = 1 if (inst_data['buy'].sum() - inst_data['sell'].sum()) > 0 else -1
        if len(margin_data) >= 2:
            res['Margin_Signal'] = 1 if margin_data['MarginPurchaseStock'].iloc[-1] < margin_data['MarginPurchaseStock'].iloc[-2] else -1
    except Exception as e:
        print(f"計算出錯 {sid}: {e}")
    return res

# --- 4. 主程式 ---

def main():
    print("--- 啟動掃描器 v3.4 ---")
    target_stocks = get_top_200_stocks()
    
    all_results = []
    today_str = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')

    for index, row in target_stocks.iterrows():
        sid, sname = str(row['stock_id']), str(row['stock_name'])
        try:
            print(f"[{index+1}/{len(target_stocks)}] 分析中: {sid} {sname}")
            # 使用 yfinance 抓取，確保不含元數據
            df_yf = yf.download(f"{sid}.TW", start=start_date, progress=False, actions=False)
            
            inst_data, margin_data = pd.DataFrame(), pd.DataFrame()
            if FINMIND_TOKEN:
                try:
                    inst_data = dl.taiwan_stock_institutional_investors(stock_id=sid, start_date=today_str)
                    margin_data = dl.taiwan_stock_margin_purchase_short_sale(stock_id=sid, start_date=start_date)
                except:
                    pass
            
            all_results.append(compute_signals(df_yf, sid, sname, inst_data, margin_data))
        except:
            continue

    # 存檔
    final_df = pd.DataFrame(all_results)
    final_df.to_csv('daily_scan.csv', index=False, encoding='utf-8-sig')
    print(f"✅ 完成！'daily_scan.csv' 已更新，共 {len(final_df)} 筆資料。")

if __name__ == "__main__":
    main()
