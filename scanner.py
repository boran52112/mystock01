import os
import pandas as pd
from FinMind.data import DataLoader

def run_scanner():
    print("機器人啟動中...")
    
    # 1. 初始化 FinMind
    # 它會自動去抓環境變數裡的 Token，我們不需要再寫 login 了
    dl = DataLoader()
    
    # 2. 核心任務：獲取台股清單 (上市與上櫃)
    print("正在向 FinMind 獲取台股全市場清單...")
    try:
        df_list = dl.taiwan_stock_info()
        
        # 3. 簡單過濾：我們只要「股票」，不要權證或認購權證 (通常編號是 4 或 6 碼)
        # 我們過濾掉 stock_id 太長的，保留最純粹的股票
        df_list = df_list[df_list['stock_id'].str.len() <= 6]
        
        print(f"✅ 成功獲取清單！全市場共有 {len(df_list)} 檔標的。")
        
        # 4. 顯示前 10 檔讓老闆(你)看看結果
        print("名單前 10 檔範例：")
        print(df_list[['stock_id', 'stock_name', 'industry_category']].head(10))
        
        # 5. 把這份名單存成 CSV (這是我們未來掃描的點名簿)
        df_list.to_csv("stock_list.csv", index=False, encoding="utf-8-sig")
        print("✅ 已將名單存為 stock_list.csv")
        
    except Exception as e:
        print(f"❌ 獲取清單時發生錯誤: {e}")

if __name__ == "__main__":
    run_scanner()
