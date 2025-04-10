# Task Submission: Doc2Vec Embedding Model

Submit your final crawler, cleaner, tokenizer, training code for embedding model and classification model, sample documents, and answer the following questions based on what you have actually done.

---

## A. Data Statistics

- **Total Number of Source Titles**: `1,064,015`
- **Total Number of Tokenized Titles**: `1,064,015`

---

## B. If A and B are different, what have you done for that?

- No

---

## C. Parameters of Doc2Vec Embedding Model

- **Total Number of Training Documents**: `1,064,015`
- **Output Vector Size**: `50`
- **Min Count**: `2`
- **Window**: `3`
- **Epochs**: `100`
- **Workers**: `4`
- **Sample Size**: `10,000`
- **First Self Similarity**: `84.87%`
- **Second Self Similarity**: `87.63%`

---

## D. Self-Similarity Evaluation

- **a. Arrangement of Linear Layers**:  `50x256x128x64x9`
- **b. Activation Function for Hidden Layers**:  `ReLU`
- **c. Activation Function for Output Layers**:  `Softmax`
- **d. Loss Function**:  `Categorical Cross Entropy`
- **e. Algorithms for Back-Propagation**:  `Adam (Adaptive Moment Estimation)`
- **f. Total Number of Training Documents**:  `851,212`
- **g. Total Number of Testing Documents**:  `212,803`
- **h. Epochs**:  `1000`  **Learning Rate**:  `0.001`
- **i. First Match**:  `83.89%`  **Second Match**:  `93.20%`

---

## E. Share your experience of optimization, including at least 2 change/result pairs.

### 1. Increased training epochs

- **Change:** Increased training epochs from 50 to 100.
- **Result:** Significant improvement in self similarity across all settings, especially when vector_size is small.

### 2. Increased embedding vector size

- **Change:** Increased vector size of Doc2Vec embedding model from 50 to 100 or 150.
- **Result:** No significant improvement in self similarity. Small vector size (50) already performed well.

### 3. Increased min_count

- **Change:** Increased min_count from 1 to 2.
- **Result:** Slight improvement in self similarity. Impact is more obvious when epochs is higher.

### 4. Increased window size

- **Change:** Increased window size from 3 to 5.
- **Result:** Slight performance drop in self similarity. Smaller window (3) is generally better.

### 5. Increased classifier epochs

`python3 -m scripts.run_classifier --csv data/tokenized_data.csv --model doc2vec_models/d2v_v50_e100_m2_w3.model --epochs 500 --hidden_dims "256,128,64"`

`python3 -m scripts.run_classifier --csv data/tokenized_data.csv --model doc2vec_models/d2v_v50_e100_m2_w3.model --epochs 1000 --hidden_dims "256,128,64"`

- **Change:** Changed epochs size of classification model from 500 to 1000.
- **Result:** First Match**:  `82.42% -> 83.89%`  **Second Match**:  `92.24% -> 93.20%`

### 6. Increased hidden layers

`python3 -m scripts.run_classifier --csv data/tokenized_data.csv --model doc2vec_models/d2v_v50_e100_m2_w3.model --epochs 500 --hidden_dims "256,128,64"`

`python3 -m scripts.run_classifier --csv data/tokenized_data.csv --model doc2vec_models/d2v_v50_e100_m2_w3.model --epochs 500 --hidden_dims "512,256,128,64"`

- **Change:** Changed hidden layers of classification model from 256x128x64 to 512x256x128x64.
- **Result:** First Match**:  `82.42% -> 83.13%`  **Second Match**:  `92.24% -> 92.84`, Starting from epoch 390, the loss began to fluctuate slightly back and forth.

---
