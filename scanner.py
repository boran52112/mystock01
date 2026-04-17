import os
import json
import requests
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime

# ==========================================
# 1. 配置與初始化
# ==========================================
# Google 試算表 ID
SHEET_ID = "1UH-fwxENhGUDmQjTQJq72g1DzsxML_CPIJGg39cWAgM"

# 設定 Google Sheets API 權限範圍
SCOPE = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

def get_gspread_client():
    """驗證並取得 Google Sheets 控制權"""
    # 優先從 GitHub Secrets 讀取 (JSON 字串)
    creds_json = os.environ.get("GCP_SERVICE_ACCOUNT_KEY")
    
    if creds_json:
        # GitHub Actions 環境
        info = json.loads(creds_json)
        creds = Credentials.from_service_account_info(info, scopes=SCOPE)
    else:
        # 本地測試環境：請確保資料夾內有 credentials.json
        creds = Credentials.from_service_account_file("credentials.json", scopes=SCOPE)
    
    return gspread.authorize(creds)

# ==========================================
# 2. 抓取證交所數據 (STOCK_DAY_ALL)
# ==========================================
def fetch_twse_data():
    print("正在從證交所 Open API 抓取全市場行情...")
    url = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data)
        
        # 轉換欄位名稱 (根據 API 返回的原始欄位進行 mapping)
        # 欄位說明：Code(代碼), Name(名稱), TradeVolume(成交量), TradeValue(成交金額), 
        # OpeningPrice(開盤), HighestPrice(最高), LowestPrice(最低), ClosingPrice(收盤)
        
        # 核心數據清理
        df['TradeVolume'] = pd.to_numeric(df['TradeVolume'], errors='coerce').fillna(0)
        df['OpeningPrice'] = pd.to_numeric(df['OpeningPrice'], errors='coerce')
        df['HighestPrice'] = pd.to_numeric(df['HighestPrice'], errors='coerce')
        df['LowestPrice'] = pd.to_numeric(df['LowestPrice'], errors='coerce')
        df['ClosingPrice'] = pd.to_numeric(df['ClosingPrice'], errors='coerce')
        
        # 篩選前 200 名成交量
        top_200 = df.sort_values(by='TradeVolume', ascending=False).head(200).copy()
        
        # 加入日期欄位 (當前系統模擬時間 2026-04-16)
        # 如果是實際執行，可用 datetime.now().strftime("%Y-%m-%d")
        top_200['Date'] = "2026-04-16"
        
        # 整理輸出欄位
        result = top_200[['Date', 'Code', 'Name', 'OpeningPrice', 'HighestPrice', 'LowestPrice', 'ClosingPrice', 'TradeVolume']]
        return result
    
    except Exception as e:
        print(f"抓取失敗: {e}")
        return None

# ==========================================
# 3. 寫入 Google 試算表
# ==========================================
def update_google_sheet(df):
    if df is None or df.empty:
        print("無資料可寫入")
        return

    try:
        client = get_gspread_client()
        sh = client.open_by_key(SHEET_ID)
        
        # 取得第一個工作表 (或指定名稱)
        worksheet = sh.get_worksheet(0) 
        
        # 檢查是否為空表 (如果是空的，寫入標頭)
        existing_data = worksheet.get_all_values()
        if len(existing_data) == 0:
            worksheet.append_row(df.columns.tolist())
            print("已建立表頭")

        # 將 DataFrame 轉為列表格式 (準備批量追加)
        values = df.values.tolist()
        
        # 執行 Append (續寫)
        worksheet.append_rows(values)
        print(f"成功寫入 {len(values)} 筆資料至 Google 試算表！")

    except Exception as e:
        print(f"寫入試算表錯誤: {e}")

# ==========================================
# 執行主程式
# ==========================================
if __name__ == "__main__":
    market_data = fetch_twse_data()
    update_google_sheet(market_data)
