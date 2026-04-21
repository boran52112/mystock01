import yfinance as yf
import pandas as pd
import pandas_ta as ta
import gspread
from google.oauth2.service_account import Credentials
import os
import json
import pytz
import requests
import time
from datetime import datetime

# ==========================================
# 1. 取得台灣時間與證交所清單
# ==========================================
def get_top_200_stocks():
    print("正在從證交所獲取全市場成交量排行...")
    # 證交所所有上市公司即時行情 URL
    url = "https://www.twse.com.tw/exchangeReport/STOCK_DAY_ALL?response=json"
    
    try:
        res = requests.get(url, timeout=10)
        data = res.json()
        if data['stat'] != 'OK':
            return {}
        
        # 轉換為 DataFrame 方便排序
        # 欄位說明: 0:代號, 1:名稱, 2:成交股數, ...
        df_all = pd.DataFrame(data['data'])
        # 移除非數字的成交量內容並轉為整數
        df_all[2] = df_all[2].str.replace(',', '').astype(float)
        
        # 根據成交股數排序 (前 200 名)
        df_top = df_all.sort_values(by=2, ascending=False).head(200)
        
        # 建立格式為 {"2330.TW": "台積電"} 的字典
        stock_dict = {}
        for _, row in df_top.iterrows():
            symbol = f"{row[0]}.TW"
            name = row[1]
            stock_dict[symbol] = name
        
        print(f"成功取得前 200 大熱門股清單！")
        return stock_dict
    except Exception as e:
        print(f"抓取證交所資料失敗: {e}")
        return {}

# ==========================================
# 2. Google 試算表連線設定
# ==========================================
def setup_google_sheets():
    creds_json = os.environ.get('GCP_SERVICE_ACCOUNT_KEY')
    if not creds_json:
        raise ValueError("找不到 GCP_SERVICE_ACCOUNT_KEY")
    
    info = json.loads(creds_json)
    info['private_key'] = info['private_key'].replace('\\n', '\n')
    
    scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    creds = Credentials.from_service_account_info(info, scopes=scope)
    client = gspread.authorize(creds)
    
    sheet_id = "1UH-fwxENhGUDmQjTQJq72g1DzsxML_CPIJGg39cWAgM"
    sh = client.open_by_key(sheet_id)
    return sh.get_worksheet(0)

# ==========================================
# 3. 核心運算邏輯
# ==========================================
def run_scanner():
    wks = setup_google_sheets()
    
    # 設定台灣時間
    tz = pytz.timezone('Asia/Taipei')
    today_date = datetime.now(tz).strftime('%Y-%m-%d')
    
    # 獲取動態選股清單 (Top 200)
    stock_list = get_top_200_stocks()
    if not stock_list:
        print("無法獲取選股清單，終止執行。")
        return

    # 定義標題 (17 欄)
    headers = [
        "日期", "股號", "股名", "開盤價", "最高價", "最低價", "收盤價", "成交量",
        "MA5", "MA20", "RSI14", "K", "D", "MACD", "BB_Upper", "BB_Lower", "漲跌幅%"
    ]
    
    # 強制初始化標題 (確保欄位正確)
    wks.update('A1', [headers])

    final_rows = []
    count = 0
    
    print(f"開始分析 200 支股票指標... (請耐心等候)")
    
    for symbol, name in stock_list.items():
        try:
            # 抓取 3 個月資料 (yfinance)
            df = yf.download(symbol, period="3mo", interval="1d", progress=False)
            
            # 檢查資料是否足以計算指標 (至少要 25 天)
            if len(df) < 25:
                continue
            
            # --- 指標運算 ---
            df['MA5'] = ta.sma(df['Close'], length=5)
            df['MA20'] = ta.sma(df['Close'], length=20)
            df['RSI14'] = ta.rsi(df['Close'], length=14)
            kd = ta.stoch(df['High'], df['Low'], df['Close'], k=9, d=3)
            df['K'] = kd['STOCHk_9_3_3']
            df['D'] = kd['STOCHd_9_3_3']
            macd = ta.macd(df['Close'])
            df['MACD'] = macd['MACD_12_26_9']
            bb = ta.bbands(df['Close'], length=20)
            df['BB_U'] = bb['BBU_20_2.0']
            df['BB_L'] = bb['BBL_20_2.0']
            df['Chg'] = df['Close'].pct_change() * 100
            
            latest = df.iloc[-1]
            
            # 組合資料行
            row = [
                today_date, symbol, name,
                round(float(latest['Open']), 2),
                round(float(latest['High']), 2),
                round(float(latest['Low']), 2),
                round(float(latest['Close']), 2),
                int(latest['Volume']),
                round(float(latest['MA5']), 2),
                round(float(latest['MA20']), 2),
                round(float(latest['RSI14']), 2),
                round(float(latest['K']), 2),
                round(float(latest['D']), 2),
                round(float(latest['MACD']), 2),
                round(float(latest['BB_U']), 2),
                round(float(latest['BB_L']), 2),
                f"{round(float(latest['Chg']), 2)}%"
            ]
            final_rows.append(row)
            count += 1
            
            # 每處理 20 支顯示一次進度
            if count % 20 == 0:
                print(f"進度: {count}/200...")
                
        except Exception as e:
            # 某些股票可能因除權息或停牌報錯，直接跳過
            continue

    # 批次寫入：一次把 200 支股票塞進試算表
    if final_rows:
        # 清空第 2 列之後的所有內容，只保留最新熱門 200 支
        # 這樣你的試算表永遠只有最新、最熱門的資料，AI 才不會被舊資料干擾
        range_to_clear = 'A2:Q500'
        wks.batch_clear([range_to_clear])
        wks.append_rows(final_rows)
        print(f"✨ 任務完成！成功更新 {len(final_rows)} 筆熱門股指標數據！")
    else:
        print("❌ 警告：沒有任何數據被成功計算。")

if __name__ == "__main__":
    run_scanner()
