import os
import json
import requests
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta
import pytz

# ==========================================
# 1. 配置與初始化
# ==========================================
SHEET_ID = "1UH-fwxENhGUDmQjTQJq72g1DzsxML_CPIJGg39cWAgM"
SCOPE = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

def get_gspread_client():
    """取得 Google Sheets 控制權 (支援本地與 GitHub)"""
    creds_json = os.environ.get("GCP_SERVICE_ACCOUNT_KEY")
    if creds_json:
        info = json.loads(creds_json)
        creds = Credentials.from_service_account_info(info, scopes=SCOPE)
    else:
        # 請確保 credentials.json 在同一個資料夾
        creds = Credentials.from_service_account_file("credentials.json", scopes=SCOPE)
    return gspread.authorize(creds)

# ==========================================
# 2. 獲取絕對正確的台北時間 (第一優先)
# ==========================================
def get_authoritative_taiwan_time():
    print("--- 步驟 1: 執行權威時間校準 ---")
    try:
        # 抓取台北標準時間 API
        response = requests.get("http://worldtimeapi.org/api/timezone/Asia/Taipei", timeout=10)
        data = response.json()
        now_taiwan = datetime.fromisoformat(data['datetime'])
        print(f"網路校準成功: {now_taiwan.strftime('%Y-%m-%d %H:%M:%S')}")
        return now_taiwan
    except Exception as e:
        print(f"網路校準失敗，使用系統時區備案: {e}")
        return datetime.now(pytz.timezone('Asia/Taipei'))

def get_target_date(now):
    """根據當前時間決定：我們該擁有哪一天的完整收盤資料"""
    # 方案一邏輯：下午 3 點 (15:00) 之後才算今日收盤已完成
    if now.hour >= 15:
        target = now.strftime('%Y-%m-%d')
        print(f"判定結果: 已過今日收盤時間，目標日期為 {target}")
    else:
        # 15:00 以前，目標是昨天
        target = (now - timedelta(days=1)).strftime('%Y-%m-%d')
        print(f"判定結果: 今日尚未收盤，目標日期為昨日 {target}")
    return target

# ==========================================
# 3. 資料比對與寫入
# ==========================================
def run_scanner():
    # A. 獲取時間
    now = get_authoritative_taiwan_time()
    target_date = get_target_date(now)

    # B. 連線試算表並檢查
    client = get_gspread_client()
    sh = client.open_by_key(SHEET_ID)
    worksheet = sh.get_worksheet(0)
    
    # 抓取最後一行的日期 (假設日期在第一欄 A 欄)
    all_dates = worksheet.col_values(1)
    last_date = all_dates[-1] if all_dates else ""
    
    print(f"試算表最後記錄日期: {last_date}")

    if last_date == target_date:
        print(f">>> 偵測完成：{target_date} 的資料已存在，符合「方案一」不重複抓取。")
        return

    # C. 抓取證交所數據
    print(f">>> 偵測到資料缺口，正在從證交所抓取 {target_date} 資料...")
    url = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
    try:
        resp = requests.get(url, timeout=30)
        df = pd.DataFrame(resp.json())
        
        # 清理數據
        df['TradeVolume'] = pd.to_numeric(df['TradeVolume'], errors='coerce').fillna(0)
        top_200 = df.sort_values(by='TradeVolume', ascending=False).head(200).copy()
        top_200['Date'] = target_date # 使用我們校準後的日期
        
        # 整理格式
        result = top_200[['Date', 'Code', 'Name', 'OpeningPrice', 'HighestPrice', 'LowestPrice', 'ClosingPrice', 'TradeVolume']]
        
        # 寫入 (Append)
        if len(all_dates) == 0:
            worksheet.append_row(result.columns.tolist())
        
        worksheet.append_rows(result.values.tolist())
        print(f"成功更新！已寫入 {target_date} 的 200 檔股票資料。")
        
    except Exception as e:
        print(f"抓取或寫入過程出錯: {e}")

if __name__ == "__main__":
    run_scanner()
