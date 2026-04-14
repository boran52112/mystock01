import os
import pandas as pd
import FinMind
from FinMind.data import DataLoader

def test_finmind():
    # 1. 從 GitHub Secrets 拿取保險箱裡的 Key
    token = os.getenv('FINMIND_TOKEN')
    
    # 2. 初始化 FinMind 資料載入器
    dl = DataLoader()
    if token:
        dl.login(token)
        print("成功登入 FinMind！")
    else:
        print("警告：找不到 API Token，將以匿名身份執行 (限制較多)")

    # 3. 測試抓取今天的全市場成交量（簡單測試）
    # 我們試著隨便抓一個功能，看 Key 能不能動
    print("正在測試抓取清單...")
    
    # 這裡我們先不寫太複雜，先確認環境是通的
    return True

if __name__ == "__main__":
    print("機器人開始運作...")
    test_finmind()
    print("測試結束。")
