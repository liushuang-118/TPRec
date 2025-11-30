import os
import pickle
from collections import defaultdict, Counter
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from data_utils import AmazonDataset

# ================================
# 配置
# ================================
DATASET = "Amazon_Clothing"
DATA_DIR = f"./data/{DATASET}"
PATH_FILE = f"./tmp/{DATASET}/train_agent/policy_paths_epoch_50.pkl"

# ================================
# 1. 读取 reasoning paths
# ================================
with open(PATH_FILE, "rb") as f:
    data = pickle.load(f)

paths = data["paths"]
probs = data["probs"]
print(f"[INFO] Loaded {len(paths)} paths")

# ================================
# 2. 构建 Su（解释词） & Top-10 推荐产品
# ================================
user_Su = defaultdict(set)
user_prod_scores = defaultdict(lambda: defaultdict(float))

for path, p_list in zip(paths, probs):
    path_prob = float(np.prod(p_list))

    uid = None
    pid = None
    words = set()

    for rel, typ, idx in path:
        if typ == "user" and uid is None:
            uid = idx
        elif typ == "product":
            pid = idx
        elif typ == "word":
            words.add(idx)

    if uid is None or pid is None:
        continue

    # Su 收集全部 word 实体
    user_Su[uid].update(words)

    # 每个产品累积路径概率
    user_prod_scores[uid][pid] += path_prob

# 取 top-10 产品
user_top10 = {}
for uid, score_dict in user_prod_scores.items():
    sorted_p = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    user_top10[uid] = [pid for pid, s in sorted_p[:10]]

print(f"[INFO] Top-10 product list generated for {len(user_top10)} users")

# ================================
# 3. 加载真实评论（Gu 来源）
# ================================
dataset = AmazonDataset(DATA_DIR, set_name="train")
reviews = dataset.review.data
print(f"[INFO] Loaded {len(reviews)} reviews")

# ================================
# 4. 构建 Gu：用户对 top-10 产品的真实评论词
# ================================
user_Gu = defaultdict(set)

for review in reviews:
    uid, pid, word_indices = review[:3]  # 取前 3 个元素
    if uid in user_top10 and pid in user_top10[uid]:
        user_Gu[uid].update(word_indices)

print(f"[INFO] Built Gu for {len(user_Gu)} users")

# ================================
# 5. 过滤 高频 & 低 TF-IDF 的词
# ================================
all_words = [w for ws in user_Gu.values() for w in ws]
freq = Counter(all_words)

# 做 TF-IDF
texts = []
uids = []
for uid, ws in user_Gu.items():
    texts.append(" ".join(dataset.word.vocab[w] for w in ws))
    uids.append(uid)

tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
X = tfidf.fit_transform(texts)
names = tfidf.get_feature_names_out()
avg_tfidf = np.asarray(X.mean(axis=0)).ravel()
word2tfidf = dict(zip(names, avg_tfidf))

# 找到要过滤的词
remove_words = set()
for w, f in freq.items():
    if f > 5000:
        word = dataset.word.vocab[w]
        if word2tfidf.get(word, 1) < 0.1:
            remove_words.add(w)

# 更新 Gu
for uid in user_Gu:
    user_Gu[uid] = {w for w in user_Gu[uid] if w not in remove_words}

print(f"[INFO] Gu filtering done.")

# ================================
# 6. 计算 Precision / Recall / F1
# ================================
pre_list, rec_list, f1_list = [], [], []

for uid in tqdm(user_Su.keys(), desc="Evaluating"):
    Su = user_Su[uid]
    Gu = user_Gu.get(uid, set())

    if not Su or not Gu:
        continue

    inter = Su & Gu

    precision = len(inter) / (len(Su) + 1)
    recall    = len(inter) / (len(Gu) + 1)
    f1        = 2 * precision * recall / (precision + recall + 1)

    pre_list.append(precision)
    rec_list.append(recall)
    f1_list.append(f1)

print("\n===== Explainability Evaluation =====")
print(f"Precision: {np.mean(pre_list):.4f}")
print(f"Recall:    {np.mean(rec_list):.4f}")
print(f"F1:        {np.mean(f1_list):.4f}")


