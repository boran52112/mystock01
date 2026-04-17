import os
import json
import requests
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta
import pytz
import urllib3

# 隱藏安全警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- Configuration ---
SHEET_ID = "1UH-fwxENhGUDmQjTQJq72g1DzsxML_CPIJGg39cWAgM"
SCOPE = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

def get_gspread_client():
    creds_json = os.environ.get("GCP_SERVICE_ACCOUNT_KEY")
    if creds_json:
        info = json.loads(creds_json)
        creds = Credentials.from_service_account_info(info, scopes=SCOPE)
    else:
        creds = Credentials.from_service_account_file("credentials.json", scopes=SCOPE)
    return gspread.authorize(creds)

def get_authoritative_time():
    """雙重校準時間邏輯"""
    print("--- Step 1: Authoritative Time Check ---")
    
    # 嘗試方案 A: WorldTimeAPI
    try:
        print("Trying WorldTimeAPI...")
        response = requests.get("http://worldtimeapi.org/api/timezone/Asia/Taipei", timeout=8, verify=False)
        if response.status_code == 200:
            now = datetime.fromisoformat(response.json()['datetime'])
            print(f"Success (WorldTimeAPI): {now.strftime('%Y-%m-%d %H:%M:%S')}")
            return now
    except Exception as e:
        print(f"WorldTimeAPI failed: {e}")

    # 嘗試方案 B: 從 Google 的標頭抓取時間 (極度穩定)
    try:
        print("Trying Google Time Server...")
        res = requests.get("https://www.google.com", timeout=8)
        # Google 的 Header 裡會有 Date: Sat, 08 Feb 2025 12:00:00 GMT
        g_date = res.headers['Date']
        # 轉換為台灣時間 (UTC+8)
        now = datetime.strptime(g_date, '%a, %d %b %Y %H:%M:%S %Z').replace(tzinfo=pytz.UTC).astimezone(pytz.timezone('Asia/Taipei'))
        print(f"Success (Google Server): {now.strftime('%Y-%m-%d %H:%M:%S')}")
        return now
    except Exception as e:
        print(f"Google Time failed: {e}")

    # 方案 C: 系統時間備案
    print("Warning: Using System Clock as last resort.")
    return datetime.now(pytz.timezone('Asia/Taipei'))

def run_scanner():
    # 1. 獲取校準時間
    now_taiwan = get_authoritative_time()
    
    # 決定目標日期 (方案一：收盤派)
    if now_taiwan.hour >= 15:
        target_date = now_taiwan.strftime('%Y-%m-%d')
        print(f"Decision: Market closed. Target is Today ({target_date})")
    else:
        target_date = (now_taiwan - timedelta(days=1)).strftime('%Y-%m-%d')
        print(f"Decision: Market open or early. Target is Yesterday ({target_date})")

    print("\n--- Step 2: Database Sync ---")
    try:
        client = get_gspread_client()
        sh = client.open_by_key(SHEET_ID)
        worksheet = sh.get_worksheet(0)
        
        all_dates = worksheet.col_values(1)
        last_date = all_dates[-1] if all_dates else ""
        print(f"Last recorded date: {last_date}")

        if last_date == target_date:
            print(f"Result: Data for {target_date} already exists. Mission aborted.")
            return
        
        print("\n--- Step 3: Fetching TWSE Data ---")
        url = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
        resp = requests.get(url, timeout=30, verify=False)
        data = resp.json()
        
        if not data:
            print("Error: No data received from TWSE.")
            return

        df = pd.DataFrame(data)
        df['TradeVolume'] = pd.to_numeric(df['TradeVolume'], errors='coerce').fillna(0)
        top_200 = df.sort_values(by='TradeVolume', ascending=False).head(200).copy()
        top_200['Date'] = target_date 
        
        result = top_200[['Date', 'Code', 'Name', 'OpeningPrice', 'HighestPrice', 'LowestPrice', 'ClosingPrice', 'TradeVolume']]
        
        if len(all_dates) == 0:
            worksheet.append_row(result.columns.tolist())
            
        worksheet.append_rows(result.values.tolist())
        print(f"--- SUCCESS! {target_date} data saved to Google Sheets ---")

    except Exception as e:
        print(f"Process failed: {e}")

if __name__ == "__main__":
    print("========== Taiwan Stock AI Scanner v4.2 ==========")
    run_scanner()
