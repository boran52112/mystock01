import os
import json
import time
import requests
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime

# --- 1. 設定與授權 ---
GCP_KEY_JSON = os.getenv("GCP_SERVICE_ACCOUNT_KEY")
SHEET_NAME = "台股資料庫_2026"

def get_gspread_client():
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    key_dict = json.loads(GCP_KEY_JSON)
    creds = Credentials.from_service_account_info(key_dict, scopes=scopes)
    return gspread.authorize(creds)

# --- 2. 核心功能：從 Open API 抓取「正確且最新」的資料 ---

def fetch_latest_twse_openapi():
    """使用證交所 Open API 抓取全市場最新行情 (免輸入日期，避開未來錯誤)"""
    print("正在透過 Open API 獲取官方最新收盤資料...")
    url = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
    
    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            print(f"❌ API 連線失敗，狀態碼：{response.status_code}")
            return pd.DataFrame()

        raw_data = response.json()
        df = pd.DataFrame(raw_data)
        
        # 今天的系統日期 (2026-04-16)
        today_str = datetime.now().strftime('%Y-%m-%d')

        # 欄位轉換 (Open API 欄位名與 MI_INDEX 不同)
        # 欄位名對照: Code(代號), Name(名稱), TradeVolume(成交量), OpeningPrice(開盤), 
        # HighestPrice(最高), LowestPrice(最低), ClosingPrice(收盤)
        clean_df = pd.DataFrame({
            '日期': today_str,
            '代號': df['Code'],
            '名稱': df['Name'],
            '開盤': pd.to_numeric(df['OpeningPrice'], errors='coerce'),
            '最高': pd.to_numeric(df['HighestPrice'], errors='coerce'),
            '最低': pd.to_numeric(df['LowestPrice'], errors='coerce'),
            '收盤': pd.to_numeric(df['ClosingPrice'], errors='coerce'),
            '成交量': pd.to_numeric(df['TradeVolume'], errors='coerce')
        })

        # 排除無效數據 (沒收盤價的)
        clean_df = clean_df.dropna(subset=['收盤'])
        
        # 按成交量排序，取前 200 名熱門股 (飆股發動通常伴隨成交量)
        top_200 = clean_df.sort_values(by='成交量', ascending=False).head(200)
        
        return top_200

    except Exception as e:
        print(f"❌ 抓取過程發生錯誤: {e}")
        return pd.DataFrame()

# --- 3. 主程式 ---

def main():
    print(f"--- 啟動掃描器 v4.1 (正確資料版) ---")
    
    # A. 抓取資料
    data_to_save = fetch_latest_twse_openapi()
    
    if data_to_save.empty:
        print("❌ 無法獲取有效資料。")
        return

    # B. 寫入 Google 試算表
    try:
        client = get_gspread_client()
        sh = client.open(SHEET_NAME)
        worksheet = sh.get_worksheet(0)

        # 轉為二維清單
        rows_to_append = data_to_save.values.tolist()
        
        # 寫入最後一行
        worksheet.append_rows(rows_to_append)
        
        print(f"✅ 成功！已將 {len(rows_to_append)} 筆正確行情存入試算表。")
        print(f"資料標註日期：{data_to_save['日期'].iloc[0]}")

    except Exception as e:
        print(f"❌ 試算表寫入失敗: {e}")

if __name__ == "__main__":
    main()
