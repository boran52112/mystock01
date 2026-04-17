import os
import json
import requests
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta
import pytz
import urllib3

# 告訴 Python 隱藏「安全檢查已關閉」的警告訊息，讓畫面看起來乾淨
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# 1. 配置與初始化 (Google 試算表設定)
# ==========================================
SHEET_ID = "1UH-fwxENhGUDmQjTQJq72g1DzsxML_CPIJGg39cWAgM"
SCOPE = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

def get_gspread_client():
    """取得 Google Sheets 控制權 (支援本地端與 GitHub 雲端)"""
    creds_json = os.environ.get("GCP_SERVICE_ACCOUNT_KEY")
    if creds_json:
        # GitHub Actions 雲端執行時使用
        info = json.loads(creds_json)
        creds = Credentials.from_service_account_info(info, scopes=SCOPE)
    else:
        # 本地電腦執行時使用 (請確保資料夾內有 credentials.json)
        creds = Credentials.from_service_account_file("credentials.json", scopes=SCOPE)
    return gspread.authorize(creds)

# ==========================================
# 2. 獲取絕對正確的台北時間 (確保時間正確)
# ==========================================
def get_authoritative_taiwan_time():
    print("\n--- [第一部分：時間偵測] ---")
    try:
        # 抓取台北標準時間 (加入 verify=False 避開安全證書報錯)
        response = requests.get("http://worldtimeapi.org/api/timezone/Asia/Taipei", timeout=10, verify=False)
        data = response.json()
        now_taiwan = datetime.fromisoformat(data['datetime'])
        print(f"權威機構時間校準成功: {now_taiwan.strftime('%Y-%m-%d %H:%M:%S')}")
        return now_taiwan
    except Exception as e:
        print(f"網路校準失敗，使用系統時區備案 (請確保電腦時間正確): {e}")
        return datetime.now(pytz.timezone('Asia/Taipei'))

def get_target_date(now):
    """根據時間決定要抓哪一天的資料 (方案一：收盤派)"""
    # 如果現在是下午 3 點 (15:00) 之後，目標就是今日
    if now.hour >= 15:
        target = now.strftime('%Y-%m-%d')
        print(f"判定結果：今日收盤已完成，抓取今日 ({target}) 數據。")
    else:
        # 否則，目標日期就是昨天 (或上個交易日)
        target = (now - timedelta(days=1)).strftime('%Y-%m-%d')
        print(f"判定結果：今日尚未收盤，抓取昨日 ({target}) 數據。")
    return target

# ==========================================
# 3. 資料比對與執行
# ==========================================
def run_scanner():
    # A. 獲取正確時間
    now_time = get_authoritative_taiwan_time()
    target_date = get_target_date(now_time)

    # B. 連線 Google 試算表
    print("\n--- [第二部分：資料庫比對] ---")
    try:
        client = get_gspread_client()
        sh = client.open_by_key(SHEET_ID)
        worksheet = sh.get_worksheet(0)
        
        # 檢查 A 欄的第一個欄位 (日期)，看最後一筆是幾號
        all_dates = worksheet.col_values(1)
        last_date_in_sheet = all_dates[-1] if all_dates else ""
        print(f"試算表最後記錄日期為: {last_date_in_sheet}")

        # 比對：如果最後日期跟目標日期一樣，就停止
        if last_date_in_sheet == target_date:
            print(f"結果：{target_date} 資料已存在。不重複更新，程式結束。")
            return
        
        # C. 開始抓取證交所資料
        print(f"\n--- [第三部分：抓取與寫入] ---")
        print(f"正在連線證交所抓取 {target_date} 的行情...")
        url = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
        
        # 加入 verify=False 避開安全證書報錯
        resp = requests.get(url, timeout=30, verify=False)
        data = resp.json()
        
        if not data:
            print("錯誤：證交所目前未提供資料 (可能是假日)。")
            return
            
        df = pd.DataFrame(data)
        
        # 數據清理：將成交量轉為數字，並取前 200 名
        df['TradeVolume'] = pd.to_numeric(df['TradeVolume'], errors='coerce').fillna(0)
        top_200 = df.sort_values(by='TradeVolume', ascending=False).head(200).copy()
        
        # 強制標註我們校準後的目標日期
        top_200['Date'] = target_date 
        
        # 整理欄位順序
        result = top_200[['Date', 'Code', 'Name', 'OpeningPrice', 'HighestPrice', 'LowestPrice', 'ClosingPrice', 'TradeVolume']]
        
        # 如果表單是空的，先寫入標頭 (Title)
        if len(all_dates) == 0:
            worksheet.append_row(result.columns.tolist())
            
        # 寫入 200 筆資料
        worksheet.append_rows(result.values.tolist())
        print(f"成功！已將 {target_date} 的 200 筆資料存入 Google 試算表。")

    except Exception as e:
        print(f"發生錯誤：{e}")

if __name__ == "__main__":
    run_scanner()
