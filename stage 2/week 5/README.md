# 📚 Doc2Vec + Classifier Pipeline

本專案是一個模組化文本分類系統，結合 Doc2Vec 向量訓練與 PyTorch 類別分類器。支援模型訓練、grid search、自動快取推論向量，以及分類準確率評估。

---

## 📁 專案結構

```
.
├── data_utils/
│   └── data_loader.py
├── models/
│   ├── doc2vec_trainer.py
│   ├── classifier_model.py
│   └── classifier_trainer.py
├── utils/
│   └── vector_cache.py
├── evaluation/
│   └── evaluator.py
├── crawler/
│   └── web_crawler.py
├── cleaner/
│   └── data_cleaner.py
├── tokenizer/
│   └── tokenizer.py
├── scripts/
│   ├── train_doc2vec.py
│   ├── run_classifier.py
│   └── stratified_sample.py
├── data/                     # 儲存原始 tokenized_data
├── cache/                    # 自動產生向量快取
├── doc2vec_models/           # 儲存訓練好的 Doc2Vec 模型
```

---

## 📦 安裝需求

```
pip install gensim torch scikit-learn pandas tqdm
```

---

## 🧠 資料格式說明

### `tokenized_data.csv` 格式：
| label | token1 | token2 | token3 | ... |
|-------|--------|--------|--------|-----|

每行一筆資料，第一欄為分類標籤，後面是經過分詞的詞彙。

---

## 🚀 使用方式

### ✅ 訓練 Doc2Vec 模型

```bash
python3 -m scripts.train_doc2vec \
  --csv tokenized_data.csv \
  --save_model \
  --vector_size 100 \
  --epochs 100 \
  --min_count 2 \
  --window 5
```

### ✅ 執行 Grid Search 並儲存結果

```bash
python3 -m scripts.train_doc2vec \
  --csv tokenized_data.csv \
  --grid_search \
  --grid_size small \
  --save_csv grid_results.csv
```

### ✅ 使用分類器對模型評估準確率（含快取）

```bash
python3 -m scripts.run_classifier \
  --csv tokenized_data.csv \
  --model models/d2v_v100_e100_m2_w5.model \
  --epochs 200 \
  --hidden_dims "128,64"
```

若要強制重跑向量推論：

```bash
python3 -m scripts.run_classifier \
  --csv tokenized_data.csv \
  --model models/d2v_v100_e100_m2_w5.model \
  --force_infer
```

---

## 🧪 評估指標

- Self Similarity@1 / @2：Doc2Vec 模型自我相似度評估
- Top-1 / Top-2 / Top-3 Accuracy：分類器預測準確率

---

## 📌 TODO

- 支援不同分類器（如 LightGBM）
- 加入 TensorBoard 可視化訓練
- 多語系資料集相容性強化

---

## 👨‍💻 作者

Lucas Lin ｜2024 @ WeHelp 深度學習實習計畫
