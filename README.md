# LLM-Assisted Cleaning of Bilingual Hong Kong Building Records

規則式基線 + 選擇性 DeepSeek 清洗管線  
（對應 COMP6132 專案提案）

## 功能概述

本腳本實作提案中的混合式清洗流程：

1. **流式 XML 解析**：不將整個檔案載入記憶體，適合處理約 12,421 筆記錄。
2. **規則式正規化**：
   - 多欄位名稱與地址拼接
   - 常見街道縮寫展開（Rd. → Road、Blk → Block 等）
   - 年份分類（numeric / blank / dash / pre_1945 / other）
   - 置信度計算與問題標記
3. **選擇性 LLM 路由**：僅對低置信度或有問題的記錄呼叫 DeepSeek，控制 token 成本。
4. **評估指標輸出**：對應提案第 IV 節所列指標。
5. **圖表產生**：自動輸出摘要圖表（PNG）。

## 環境需求

```bash
pip install openai matplotlib
# 選用（簡繁轉換）
pip install opencc-python-reimplemented
# 或
pip install zhconv
```

## 使用方式

### 1. 純規則式（不消耗 token）

```bash
python importData.py \
  --url https://www.rvd.gov.hk/datagovhk/bnb-u.xml \
  -o cleaned_rule.csv \
  --limit 1000
```

### 2. 選擇性 LLM 模式（僅低置信度記錄消耗 token）

```bash
export DEEPSEEK_API_KEY="sk-your-api-key-here"

python importData.py \
  --url https://www.rvd.gov.hk/datagovhk/bnb-u.xml \
  -o cleaned_hybrid.csv \
  --use-llm \
  --limit 200
```

### 3. 使用本地檔案 + 簡繁轉換

```bash
# 先下載
curl -L -o bnb-u.xml https://www.rvd.gov.hk/datagovhk/bnb-u.xml

# 香港繁體轉簡體
python importData.py \
  --file bnb-u.xml \
  -o cleaned.csv \
  --convert hk2s \
  --use-llm
```

## 主要參數

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--url` | XML 檔案網址 | - |
| `--file` | 本地 XML 檔案路徑 | - |
| `-o` / `--output` | 輸出檔案路徑（必填） | - |
| `--format` | 輸出格式（csv / jsonl） | csv |
| `--limit` | 最多處理筆數（0 = 全部） | 0 |
| `--convert` | 簡繁轉換（none / t2s / hk2s / s2t） | none |
| `--use-llm` | 啟用選擇性 DeepSeek 呼叫 | 關閉 |
| `--model` | DeepSeek 模型名稱 | deepseek-chat |
| `--archived-count` | 原始檔案預期總筆數 | 12421 |

## 輸出說明

### CSV / JSONL 欄位

- 清洗後的中英文名稱與地址
- 年份分類與數值
- 置信度分數與 `needs_llm` 標記
- `input_tokens` / `output_tokens`（僅 LLM 模式有值）
- `llm_applied`、`llm_rationale`、`unresolved`（LLM 模式）

### 圖表

執行完成後會產生 `{output_stem}_summary.png`，包含：

- 年份分類分布
- 主要缺失率
- 置信度分數分布
- 路由與 LLM 應用比例

### 終端機評估報告

對應提案第 IV 節指標，包括：

- Parsing completeness
- Name / Address bilingual consistency
- Year anomalies
- Records per minute
- Routed record ratio
- Total input / output tokens
- Runtime

## 注意事項

- 未加 `--use-llm` 時完全不消耗 token。
- 建議先以 `--limit 200` 測試 LLM 模式，確認 API Key 與成本。
- 原始資料來源：[Rating and Valuation Department – Names of Buildings](https://www.rvd.gov.hk/datagovhk/bnb-u.xml)
- 本管線保留原始欄位與清洗結果，支援後續品質比較與溯源。

## 專案對應

本實作對應提案目標：

- (i) 建立可重現的規則式清洗基線
- (ii) 僅對低置信度記錄加入 LLM 輔助
- (iii) 提供三個資料規模的評估框架
- (iv) 輸出錯誤、耗時、token 用量與成本相關資訊
