import os
import json
import time
import requests
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta

# --- 1. 設定與授權 ---

# 從 GitHub Secrets 讀取 Google 通行證
GCP_KEY_JSON = os.getenv("GCP_SERVICE_ACCOUNT_KEY")
# 您的試算表名稱
SHEET_NAME = "台股資料庫_2026"

def get_gspread_client():
    """初始化 Google Sheets 連線"""
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    key_dict = json.loads(GCP_KEY_JSON)
    creds = Credentials.from_service_account_info(key_dict, scopes=scopes)
    return gspread.authorize(creds)

# --- 2. 核心功能：從證交所搬貨 ---

def fetch_twse_data():
    """從證交所抓取最新的全市場收盤行情"""
    print("正在尋找證交所最新交易日資料...")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    # 從今天往回找 10 天，確保避開週末或國定假日
    for i in range(10):
        target_date = (datetime.now() - timedelta(days=i)).strftime('%Y%m%d')
        url = f"https://www.twse.com.tw/exchangeReport/MI_INDEX?response=json&date={target_date}&type=ALLBUT0999"
        
        try:
            response = requests.get(url, headers=headers)
            data = response.json()
            
            if data.get('stat') == 'OK':
                print(f"✅ 成功找到證交所資料，日期：{target_date}")
                # 證交所資料表中的 table9 通常是所有股票的行情
                # 欄位說明: 0:代號, 1:名稱, 2:成交股數, 5:開盤, 6:最高, 7:最低, 8:收盤
                raw_data = data['data9']
                df = pd.DataFrame(raw_data)
                
                # 挑選我們需要的欄位
                # 證交所原始格式是字串，需要清洗掉逗號並轉為數字
                clean_df = pd.DataFrame({
                    '日期': target_date,
                    '代號': df[0],
                    '名稱': df[1],
                    '成交量': df[2].str.replace(',', '').astype(float),
                    '開盤': pd.to_numeric(df[5].str.replace(',', ''), errors='coerce'),
                    '最高': pd.to_numeric(df[6].str.replace(',', ''), errors='coerce'),
                    '最低': pd.to_numeric(df[7].str.replace(',', ''), errors='coerce'),
                    '收盤': pd.to_numeric(df[8].str.replace(',', ''), errors='coerce')
                })
                
                # 排除沒有成交或停牌的股票 (收盤價為空的)
                clean_df = clean_df.dropna(subset=['收盤'])
                
                # 按成交量排序，取前 200 名
                top_200 = clean_df.sort_values(by='成交量', ascending=False).head(200)
                
                # 重新排列欄位順序以符合試算表: 日期, 代號, 名稱, 開盤, 最高, 最低, 收盤, 成交量
                return top_200[['日期', '代號', '名稱', '開盤', '最高', '最低', '收盤', '成交量']]
                
        except Exception as e:
            print(f"嘗試抓取 {target_date} 失敗: {e}")
        
        time.sleep(3) # 禮貌性的延遲，避免被封鎖
        
    return pd.DataFrame()

# --- 3. 主程式 ---

def main():
    print(f"--- 啟動掃描器 v4.0 (證交所 ➔ 試算表) ---")
    
    # A. 抓資料
    data_to_save = fetch_twse_data()
    
    if data_to_save.empty:
        print("❌ 錯誤：無法從證交所獲取任何有效資料，請檢查網路或日期設定。")
        return

    # B. 寫入 Google 試算表
    try:
        client = get_gspread_client()
        sh = client.open(SHEET_NAME)
        worksheet = sh.get_worksheet(0) # 開啟第一個工作表

        # 將 DataFrame 轉為二維清單格式以便寫入
        rows_to_append = data_to_save.values.tolist()
        
        # 執行續寫動作 (Append)
        worksheet.append_rows(rows_to_append)
        
        print(f"✅ 任務成功！已將 {len(rows_to_append)} 筆最新數據存入 Google 試算表。")
        print(f"目前最新日期：{data_to_save['日期'].iloc[0]}")

    except Exception as e:
        print(f"❌ 寫入試算表失敗：{e}")
        print("請檢查：1. 是否已共用試算表給服務帳號 Email？ 2. Secret GCP_SERVICE_ACCOUNT_KEY 內容是否正確？")

if __name__ == "__main__":
    main()
