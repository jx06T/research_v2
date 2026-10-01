# 掃描稿紙資料管線

這條流程將圖片或 PDF 的規則稿紙切成單格，使用 Gemini 3.5 Flash 或 Flash-Lite 產生候選標籤，覆核後匯出現有 PyTorch 訓練程式可讀的配對資料。原始掃描檔不會修改。請在專案根目錄執行以下命令。

## 建議使用：單一批次入口

```powershell
python scripts/scan_workflow.py run `
  --input data/scanned_document/1140101/114_1-1.pdf `
  --output data/processed_scans/1140101 `
  --writer-id writer_1140101 `
  --package data/scanned_packages/1140101.npz
```

這個入口依序呼叫下方各腳本：切格、以 Gemini 3.5 Flash 每張 10 字卡辨識、開啟本地覆核頁、產生訓練 CSV 與可攜式 NPZ。按覆核頁的「完成覆核」才會匯出。預設兩次請求開始時間至少間隔 4 秒，遇到 HTTP 429 會暫停後重試；可用 `--min-interval` 調整。若已有 `cells.csv`，會略過切格並接續尚未辨識的格位。也可以單獨使用 `prepare`、`label`、`review`、`package`、`train` 子命令。

`run` 是新資料的完整流程。若只要接續目前的樣本覆核，不必重跑切格與 Gemini：

```powershell
python scripts/scan_workflow.py review --output data/processed_scans/pilot
```

覆核完可分別產生人工確認版與包含 Gemini 候選的探索版：

```powershell
python scripts/scan_workflow.py package --output data/processed_scans/pilot `
  --package data/scanned_packages/1140101_reviewed.npz

python scripts/scan_workflow.py package --output data/processed_scans/pilot `
  --package data/scanned_packages/1140101_gemini_proposals.npz --include-proposed
```

`prepare` 只切格，`label` 只辨識未處理格位，`review` 只開覆核頁，`package` 只匯出封包，`train` 讀封包並呼叫現有訓練程式。各子命令可用 `--help` 查看參數。`run` 遇到既有 `cells.csv` 會沿用，因此不會覆蓋已覆核標籤。

先前若有 Flash-Lite 候選字，要改由 3.5 Flash 重辨，可用：

```powershell
python scripts/scan_workflow.py label --output data/processed_scans/1140101 `
  --model gemini-3.5-flash --relabel-lite --min-interval 4
```

它會先備份原始 `cells.csv`，只重辨 Flash-Lite 的未覆核候選字；人工已接受的標籤不變。

## 1. 安裝依賴

```powershell
python -m pip install -r requirements.txt
```

PDF 使用 `pypdfium2` 渲染。`.env` 可寫 `GEMINI_API_KEY = "..."`；腳本也接受同名環境變數，並優先使用環境變數。`.env` 和 `data/processed_scans/` 已列入 `.gitignore`。

## 2. 切格與背景清理

```powershell
python scripts/prepare_scans.py data/scanned_document/1140101/114_1-1.pdf `
  --output data/processed_scans/1140101 --cols 26 --rows 22 --writer-id writer_1140101
```

若輸入是資料夾，會處理其中的 PDF 與常見圖片格式。每頁產生 `page_preview.png`、`grid_overlay.png`、`raw/`、`clean/` 及 `calibration.json`，總表在輸出目錄的 `cells.csv`。**先看格線覆蓋圖**；有錯位時，另選新的輸出目錄，根據 `page_preview.png` 的像素座標提供四個外框角點：

```powershell
python scripts/prepare_scans.py data/scanned_document/1140101/114_1-1.pdf `
  --output data/processed_scans/manual_trial --cols 26 --rows 22 `
  --corners UL_X UL_Y UR_X UR_Y LR_X LR_Y LL_X LL_Y
```

四角順序固定為左上、右上、右下、左下；數值要填實際像素座標。輸出目錄若已有 `cells.csv`，腳本會拒絕覆寫標籤，除非明確加 `--overwrite`。此版自動校正以淡綠格線為主；其他顏色或版型可使用四角座標與對應列欄數。

`cells.csv` 每格一列，保留來源檔、頁碼、行列、閱讀順序、字圖路徑、格框、候選標籤及處理狀態。欄序由右往左，每欄由上往下；空格不會讓後續 ID 位移。背景清理會移除淡綠印刷痕跡，保留暗色手寫線條及未處理原圖供覆核。

## 3. Gemini 辨識

先只製作字卡而不呼叫 API，可檢查每格的原圖與清理圖：

```powershell
python scripts/label_scans_gemini.py data/processed_scans/1140101/cells.csv --cards-only --limit 10
```

確認後開始辨識。預設模型為 `gemini-3.5-flash`、每張字卡 10 格，並在每張字卡完成後更新 `cells.csv`，中斷後可以續跑：

```powershell
python scripts/label_scans_gemini.py data/processed_scans/1140101/cells.csv
```

較省成本的模型：

```powershell
python scripts/label_scans_gemini.py data/processed_scans/1140101/cells.csv --model gemini-3.5-flash-lite
```

模型對每格回傳一個字或空字串。合法單字標為 `proposed`；空字串標為 `rejected`。回覆缺號、多字等不會直接寫入訓練標籤。若確定可接受模型的單字答案直接進入訓練，可加 `--auto-accept`；預設仍要求覆核。

## 4. 覆核與匯出

匯出覆核表，在 `decision` 欄填 `accept` 或 `reject`，接受時可於 `final_label` 改正文字。未填的列保持原狀。

```powershell
python scripts/review_scan_labels.py data/processed_scans/1140101/cells.csv --export data/processed_scans/1140101/review.csv
python scripts/review_scan_labels.py data/processed_scans/1140101/cells.csv --apply data/processed_scans/1140101/review.csv
python scripts/export_scanned_training.py data/processed_scans/1140101/cells.csv `
  --output data/processed_scans/1140101/train_manifest.csv
```

`train_manifest.csv` 只包含狀態為 `accepted`、標籤為單一漢字且圖檔存在的格位。每列包含 `id,clean_path,label,writer_id,source_file,page,row,col,label_source`。可保留重複字的不同實際筆跡。標點、空格與無法辨識的字保留在 `cells.csv` 供稽核，不加入訓練。

若要先用未人工確認的 Gemini 候選字做探索性訓練，可另輸出一份資料，加入 `--include-proposed`；這些列的 `label_source` 仍標示為 `gemini_proposal`，方便與人工確認資料區分。

### 可視化覆核

```powershell
python scripts/scan_workflow.py review --output data/processed_scans/1140101
```

本機瀏覽器會同時顯示原始字圖、清理字圖與該字在整頁上的紅框位置。可直接修正標籤、接受、捨棄或跳過；每次決定都立即儲存到 `cells.csv`，並追加到 `review_events.jsonl` 留下修訂紀錄。預設只顯示待覆核項目，也能切到全部、已接受或空格。伺服器只綁定 `127.0.0.1`。

鍵盤操作：`Enter` 接受目前候選並前進、`R` 拒絕並前進、`E` 聚焦正確字欄供修改（編輯後按 `Enter` 接受）、`←`／`→` 切換格位。換格時不會自動聚焦輸入框。人工拒絕會記為已覆核，從「待覆核」清單移除；Gemini 自動判為空字的格位仍可人工檢查。可用「已覆核」、「人工拒絕」與「模型未辨識」篩選。

## GitHub 與雲端訓練

CSV 只有標籤與本機絕對圖片路徑，**不能單獨在雲端訓練**。`scan_workflow.py package` 會把每張手寫字圖的 128×128 灰階像素、標籤、格位 ID 與書寫者 ID 壓縮到 `data/scanned_packages/*.npz`，另附 JSON 版本、SHA-256 校驗資訊與逐字可讀的 `_labels.jsonl`。NPZ 是資料陣列檔，不是逐張 PNG；每個標籤仍能對應其筆畫影像。這三個檔案未被 `.gitignore` 排除，可在檢查內容與大小後加入 GitHub。原始 PDF、工作中的字卡與本地快取不需要跟著同步。

雲端環境拉取 GitHub 專案、安裝依賴後可直接執行：

```bash
python scripts/scan_workflow.py train \
  --package data/scanned_packages/1140101.npz \
  --src-font data/fonts/NotoSansTC-Regular.ttf \
  --writer-id writer_1140101
```

`train` 會檢查封包摘要，將像素還原到忽略追蹤的快取目錄，產生當地可用的 `train_manifest.csv`，再呼叫現有 `src/train.py --dataset scanned`。這樣 GitHub 不必保存大量單字圖檔，也不會把本機絕對路徑帶進雲端。

## 5. 接到現有訓練

```powershell
python src/train.py --dataset scanned --config configs/scanned_unet_size128.yaml `
  --manifest data/processed_scans/1140101/train_manifest.csv `
  --src_font data/fonts/NotoSansTC-Regular.ttf --writer_id writer_1140101 `
  --output_dir runs/scanned_1140101 --checkpoint_dir runs_checkpoints/scanned_1140101
```

`ScannedGlyphDataset` 用標籤字從來源字體渲染來源影像，並把清理後的手寫字圖當成目標影像，輸出與現有訓練迴圈相同的 `(source, target, label)` 格式。兩張影像使用相同的隨機幾何增強；若 manifest 混合多位書寫者，必須用 `--writer_id` 選定一位。字圖都轉成現有模型使用的「白筆畫、黑背景」張量。

這份樣本頁可用來驗證管線。正式訓練前應先檢查接受字數、不同漢字數及書寫者一致性；單頁資料通常不足以評估對未見字的泛化。
