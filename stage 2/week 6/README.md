# 📘 短文本分類預測服務

本專案使用 Doc2Vec + PyTorch 進行 PTT 文章標題分類，並提供 FastAPI Web 介面讓使用者輸入標題並預測其類別。

---

## 📦 環境安裝

### 1️⃣ 安裝相依套件

```bash
pip install -r requirements.txt
```

> 若無 GPU，PyTorch 請選擇安裝 CPU 版本即可。

---

## 🚀 執行方式

### 2️⃣ 切換至專案根目錄（`week 6` 資料夾）

```bash
cd path/to/week\ 6
```

### 3️⃣ 啟動 FastAPI 本地伺服器

```bash
uvicorn web.main:app --reload
```

啟動後可於瀏覽器開啟：
```
http://127.0.0.1:8000
```

---

## 🔍 功能介紹

### ✅ Doc2Vec 模型訓練

- `scripts/train_doc2vec.py`：訓練 Doc2Vec 模型

### ✅ 分類模型訓練

- `scripts/run_classifier.py`：使用已訓練的 Doc2Vec vectors 搭配 PyTorch 類神經網路訓練分類模型

### ✅ Web 預測介面

- 使用者輸入文章標題，即時顯示預測結果與建議類別
- 使用者可點選正確分類回傳 feedback 資料

---

## 📁 專案結構

```bash
week 6/
├── README.md
├── requirements.txt
├── title_label_training
│   ├── cache
│   ├── classifier
│   ├── data
│   ├── data_utils
│   ├── doc2vec_models
│   ├── evaluation
│   ├── scripts
│   └── tokenizer
├── user-labeled-titles.csv
└── web
    ├── __init__.py
    ├── config.py
    ├── main.py
    ├── models
    ├── routers
    ├── schemas
    ├── scripts
    ├── templates
    └── utils
```

---

## 📮 API Routes

| 方法 | 路由 | 說明 |
|------|------|------|
| GET | `/` | 首頁（輸入預測） |
| GET | `/api/model/prediction?title=...` | 回傳分類與建議類別 |
| POST | `/api/model/feedback` | 儲存使用者選擇的分類 |

---

## 🙌 補充說明

- 若遇到 `ModuleNotFoundError`，請確認你是在 `week 6` 根目錄下執行 uvicorn
- 若使用 macOS 空格資料夾，記得用反斜線或引號：

```bash
cd "WeHelp - Deep Learning/stage 2/week 6"
```

---
