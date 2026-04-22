# ... (前面的 Import 與初始化維持不變) ...

# ==========================================
# 2. 資料獲取邏輯 (加入去重機制)
# ==========================================
def get_detective_data(symbol):
    symbol = symbol.upper().strip()
    yf_symbol = f"{symbol}.TW" if ".TW" not in symbol and ".TWO" not in symbol else symbol

    # --- 方案 A：搜尋 5 日歷史資料庫 ---
    try:
        scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        creds = Credentials.from_service_account_info(gcp_info, scopes=scope)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key("1UH-fwxENhGUDmQjTQJq72g1DzsxML_CPIJGg39cWAgM")
        wks = sh.get_worksheet(0)
        
        all_df = pd.DataFrame(wks.get_all_records())
        
        # 【重要去重邏輯】：如果同一天有重複存檔，只保留最後一筆
        all_df = all_df.drop_duplicates(subset=['日期', '股號'], keep='last')
        
        target_df = all_df[all_df['股號'].astype(str).str.contains(symbol)].tail(5)
        
        if not target_df.empty and len(target_df) >= 1: # 即使只有1天也分析
            return target_df, "📊 數據來源：200大歷史資料庫 (已自動去重)"
    except:
        pass

    # --- 方案 B：現場即時偵查 ---
    try:
        df = yf.download(yf_symbol, period="4mo", interval="1d", progress=False)
        if df.empty: return None, None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df.ta.sma(length=5, append=True)
        df.ta.sma(length=20, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.stoch(k=9, d=3, append=True)
        df.ta.macd(append=True)
        df.ta.bbands(length=20, append=True)
        df['漲跌幅%'] = df['Close'].pct_change() * 100
        
        res = df.tail(5).copy()
        res['日期'] = res.index.strftime('%Y-%m-%d')
        res['股號'] = yf_symbol
        res['股名'] = "現場偵查標的"
        
        name_map = {
            'Close': '收盤價', 'SMA_5': 'MA5', 'SMA_20': 'MA20', 'RSI_14': 'RSI14',
            'STOCHk_9_3_3': 'K', 'STOCHd_9_3_3': 'D', 'MACD_12_26_9': 'MACD'
        }
        res = res.rename(columns={k: v for k, v in name_map.items() if k in res.columns})
        return res.round(2), "🔍 數據來源：現場即時偵查"
    except:
        return None, None

# ==========================================
# 3. UI 呈現 (強化 AI 語言鎖定)
# ==========================================
# ... (中間 UI 部分維持不變) ...

            # --- 強化版 2026 偵探指令 (Prompt) ---
            prompt = f"""
            你是一位派駐在 2026 年的台股資深分析偵探。
            請完全使用【繁體中文】回覆，禁止使用英文標題（如 Persona, Verdict 等）。

            【辦案對象】：{stock_id}
            【5日技術指標證據】：
            {evidence_table}

            請嚴格依照以下格式輸出報告：

            📌 偵探報告編號：2026-{stock_id}-SCAN
            🕵️ 偵探身份：2026年派駐資深分析偵探

            ### 第一區塊：【九項指標與趨勢判讀】
            (請在此詳細分析指標走向與市場心理，分析投資人是恐慌還是興奮...)

            ### 第二區塊：【指標矛盾整合與風險抓漏】
            (請檢查是否有技術面打架或誘多陷阱...)

            ### 第三區塊：【全方位操作戰略：雙重劇本】
            1. 保守型劇本：(支撐點、停損位、進場邏輯)
            2. 激進型劇本：(突破點、目標價、短線追擊)

            ### 第四區塊：【偵探總結與信心分數】
            * 總結：(用一句話定調短線趨勢)
            * 信心分數：(0-100)
            """
# ... (後續 AI 呼叫代碼不變) ...
