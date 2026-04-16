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
    try:
        dl.login_token(FINMIND_TOKEN)
    except:
        pass

COLUMNS = [
    'Date', 'StockID', 'StockName', 'Close', 
    'MA_Signal', 'MACD_Signal', 'RSI_Signal', 'KD_Signal', 'BB_Signal',
    'Shadow_Signal', 'Gap_Signal', 'Inst_Signal', 'Margin_Signal'
]

# --- 2. 備援名單 (200 檔核心股，確保 API 故障時也能掃描熱門股) ---
BACKUP_IDS = [
    "2330","2317","2454","2308","2382","2303","2881","2882","3008","2603","2609","2615","2357","3231","2376","6669","2408","2409","3481","3037",
    "2324","2353","2356","2377","2379","2383","2401","2449","2451","3034","3035","3044","3443","3532","3711","4919","4938","4958","4961","6176",
    "6213","6239","6415","8046","8210","1101","1102","1216","1301","1303","1326","1402","1503","1504","1513","1519","1605","1722","1802","2002",
    "2006","2105","2201","2204","2206","2347","2501","2542","2606","2610","2618","2707","2801","2809","2812","2834","2880","2883","2884","2885",
    "2886","2887","2888","2890","2891","2892","2912","5871","5876","5880","6505","8454","9904","9910","9921","9945","1476","9933","3017","2360",
    "2395","3661","6515","2345","3013","2368","1560","3045","3036","2474","2354","2352","2371","6285","2458","3533","6121","6223","5347","5483",
    "6488","8069","3105","3264","6274","3293","4966","6462","3653","3680","5274","6231","8299","3324","3227","6147","3131","3529","3545","3592",
    "6510","6414","3406","6446","1795","4147","6547","4743","1760","6472","1789","4105","1608","1609","1611","1618","2633","2634","2637","2727",
    "5284","2207","2363","2344","2337","2367","2313","2401","2402","2404","2405","2406","2412","2415","2417","2419","2420","2421","2428","2430"
]
# 自動去除重複代號
BACKUP_IDS = list(dict.fromkeys(BACKUP_IDS))

# --- 3. 核心功能 ---

def get_top_200_stocks():
    """優先抓取排行，失敗則備援，確保一定有清單"""
    print("正在獲取成交量排行清單...")
    df_info = pd.DataFrame()
    try: df_info = dl.taiwan_stock_info()
    except: pass

    # 往回找最近 7 天的統計資料
    for i in range(0, 7):
        t_date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
        try:
            df_vol = dl.taiwan_stock_daily_statistics(date=t_date)
            if not df_vol.empty:
                print(f"✅ 成功獲取 {t_date} 排行榜。")
                top_200 = df_vol.sort_values('total_volume', ascending=False).head(200)
                if not df_info.empty:
                    return pd.merge(top_200[['stock_id']], df_info[['stock_id', 'stock_name']], on='stock_id', how='left')
                return top_200[['stock_id']].assign(stock_name=top_200['stock_id'])
        except: continue
    
    print(f"⚠️ API 排行榜暫時無法獲取，載入備援名單模式 (共 {len(BACKUP_IDS)} 檔)...")
    b_df = pd.DataFrame({"stock_id": BACKUP_IDS})
    if not df_info.empty:
        return pd.merge(b_df, df_info[['stock_id', 'stock_name']], on='stock_id', how='left')
    return b_df.assign(stock_name=b_df['stock_id'])

def compute_signals(df, sid, sname, inst_data, margin_data):
    """計算指標，自動偵測並標註數據的真實日期"""
    # 基本防錯：如果 yfinance 給的是空值，直接跳過
    if df is None or df.empty or len(df) < 20:
        return None 
    
    try:
        # 自動抓取資料庫中最後一筆資料的日期 (解決未來日期陷阱)
        actual_date = df.index[-1].strftime('%Y-%m-%d')
        
        res = {col: 0 for col in COLUMNS}
        res.update({'Date': actual_date, 'StockID': sid, 'StockName': sname})
        
        c = df['Close'].astype(float)
        h = df['High'].astype(float)
        l = df['Low'].astype(float)
        o = df['Open'].astype(float)
        
        close_p = c.iloc[-1]
        res['Close'] = round(close_p, 2)
        
        # 指標計算 (使用 pandas_ta)
        ma5, ma20 = ta.sma(c, 5), ta.sma(c, 20)
        res['MA_Signal'] = 1 if (ma5.iloc[-1] > ma20.iloc[-1] and close_p > ma20.iloc[-1]) else -1
        
        macd = ta.macd(c)
        if macd is not None: res['MACD_Signal'] = 1 if macd['MACDs_12_26_9'].iloc[-1] > 0 else -1
        
        rsi = ta.rsi(c, 14)
        if rsi is not None: res['RSI_Signal'] = 1 if rsi.iloc[-1] > 50 else -1
        
        kd = ta.stoch(h, l, c)
        if kd is not None: res['KD_Signal'] = 1 if kd['STOCHk_14_3_3'].iloc[-1] > kd['STOCHd_14_3_3'].iloc[-1] else -1
        
        bb = ta.bbands(c, 20)
        if bb is not None: res['BB_Signal'] = 1 if close_p > bb['BBM_20_2.0'].iloc[-1] else -1
        
        body = abs(close_p - o.iloc[-1])
        shadow = min(o.iloc[-1], close_p) - l.iloc[-1]
        res['Shadow_Signal'] = 1 if shadow > (body * 2) and body > 0 else 0
        
        res['Gap_Signal'] = 1 if l.iloc[-1] > h.iloc[-2] else (-1 if h.iloc[-1] < l.iloc[-2] else 0)
        
        # 籌碼指標 (FinMind)
        if not inst_data.empty:
            res['Inst_Signal'] = 1 if (inst_data['buy'].sum() - inst_data['sell'].sum()) > 0 else -1
        if len(margin_data) >= 2:
            res['Margin_Signal'] = 1 if margin_data['MarginPurchaseStock'].iloc[-1] < margin_data['MarginPurchaseStock'].iloc[-2] else -1
            
        return res
    except Exception as e:
        print(f"   ⚠️ {sid} 資料處理跳過: {e}")
        return None

# --- 4. 主程式 ---

def main():
    print(f"--- 啟動掃描器 v3.6 (動態日期版) ---")
    target_stocks = get_top_200_stocks()
    
    all_results = []
    # 設定 yfinance 抓取最新的 1 個月資料，不受系統時間限制
    # 設定 FinMind 抓取最近 40 天的籌碼資料
    chip_start_date = (datetime.now() - timedelta(days=40)).strftime('%Y-%m-%d')
    today_str = datetime.now().strftime('%Y-%m-%d')

    for index, row in target_stocks.iterrows():
        sid, sname = str(row['stock_id']), str(row['stock_name'])
        try:
            print(f"[{index+1}/{len(target_stocks)}] 正在掃描: {sid} {sname}")
            # period='1mo' 會自動抓取最新可用的一個月資料
            df_yf = yf.download(f"{sid}.TW", period='1mo', progress=False, actions=False)
            
            inst_data, margin_data = pd.DataFrame(), pd.DataFrame()
            if FINMIND_TOKEN:
                try:
                    inst_data = dl.taiwan_stock_institutional_investors(stock_id=sid, start_date=today_str)
                    margin_data = dl.taiwan_stock_margin_purchase_short_sale(stock_id=sid, start_date=chip_start_date)
                except: pass
            
            res = compute_signals(df_yf, sid, sname, inst_data, margin_data)
            if res: all_results.append(res)
            
        except: continue

    # 存檔邏輯
    if all_results:
        new_df = pd.DataFrame(all_results)
        new_df.to_csv('daily_scan.csv', index=False, encoding='utf-8-sig')
        print(f"✅ 完成！'daily_scan.csv' 已生成，共 {len(new_df)} 檔資料。")
    else:
        print("❌ 錯誤：未獲取任何有效數據。")

if __name__ == "__main__":
    main()
