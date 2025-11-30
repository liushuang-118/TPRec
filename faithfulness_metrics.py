import pickle
from collections import defaultdict, Counter
import numpy as np
from scipy.spatial.distance import jensenshannon
import random

# ========= 参数 =========
DATASET = "Amazon_Clothing"
BASE_DIR = f"tmp/{DATASET}/train_agent"
TRAIN_FILE = f"{BASE_DIR}/train_sampled_paths.pkl"
TEST_FILE  = f"{BASE_DIR}/policy_paths_epoch_50.pkl"

TOPK_PRODUCTS = 20          # 增加 top-K 产品数量
NUM_PATHS_PER_PRODUCT = 50  # 每个产品随机采样路径数量

# ========= 有效关系 =========
# VALID_RELATIONS = [
#     "purchase", "produced_by", "belongs_to", "also_bought",
#     "also_viewed", "bought_together", "mention", "described_as"
# ]
VALID_RELATIONS = (
    ["also_bought", "also_viewed", "belongs_to", "bought_together", "produced_by"]
    + [f"mention_{i}" for i in range(14)]
    + [f"described_as_{i}" for i in range(14)]
    + [f"purchase_{i}" for i in range(14)]
)

# ================================
# 1. 构建训练 F(u)
# ================================
with open(TRAIN_FILE, "rb") as f:
    train_data = pickle.load(f)

F_u = defaultdict(Counter)
for path in train_data['paths']:
    uid = None
    for rel, typ, idx in path:
        if typ == "user" and uid is None:
            uid = idx
        if rel in VALID_RELATIONS:
            F_u[uid][rel] += 1

# 归一化
F_u_norm = {}
for uid, counter in F_u.items():
    total = sum(counter.values())
    if total > 0:
        F_u_norm[uid] = {rel: count / total for rel, count in counter.items()}

train_users = set(F_u_norm.keys())
print(f"[INFO] Built F(u) for {len(F_u_norm)} users")

# ================================
# 2. 处理测试路径 (仅训练用户)
# ================================
with open(TEST_FILE, "rb") as f:
    test_data = pickle.load(f)

paths = test_data['paths']
probs = test_data['probs']

user_prod_scores = defaultdict(lambda: defaultdict(float))
user_prod_paths  = defaultdict(lambda: defaultdict(list))

for path, p_list in zip(paths, probs):
    path_prob = float(np.prod(p_list))
    uid, pid = None, None
    for rel, typ, idx in path:
        if typ == "user" and uid is None:
            uid = idx
        elif typ == "product":
            pid = idx
    if uid is None or pid is None or uid not in train_users:
        continue

    user_prod_scores[uid][pid] += path_prob
    user_prod_paths[uid][pid].append({"path": path, "prob": path_prob})

# 随机采样 top-K 产品及每个产品多条路径
user_topk_paths = defaultdict(list)
for uid in train_users:
    score_dict = user_prod_scores.get(uid, {})
    if not score_dict:
        continue
    sorted_prods = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    topk_pids = [pid for pid, _ in sorted_prods[:TOPK_PRODUCTS]]

    for pid in topk_pids:
        paths_list = user_prod_paths[uid][pid]
        sampled_paths = random.sample(paths_list, min(NUM_PATHS_PER_PRODUCT, len(paths_list)))
        user_topk_paths[uid].extend(sampled_paths)

print(f"[INFO] Got top-{TOPK_PRODUCTS} products & sampled paths for {len(user_topk_paths)} users")

# ================================
# 3. 构建测试规则分布 Qf(u) / Qw(u)
# ================================
Qf_u = defaultdict(Counter)
Qw_u = defaultdict(Counter)

for uid, path_list in user_topk_paths.items():
    for item in path_list:
        path = item['path']
        prob = item['prob']
        for rel, typ, idx in path:
            if rel in VALID_RELATIONS:
                Qf_u[uid][rel] += 1          # 计数
                Qw_u[uid][rel] += prob       # 权重累积

# 归一化 Qf(u)
Qf_u_norm = {}
for uid, counter in Qf_u.items():
    total = sum(counter.values())
    if total > 0:
        Qf_u_norm[uid] = {rel: count / total for rel, count in counter.items()}

# 归一化 Qw(u)
Qw_u_norm = {}
for uid, counter in Qw_u.items():
    total = sum(counter.values())
    if total > 0:
        Qw_u_norm[uid] = {rel: val / total for rel, val in counter.items()}

print("[INFO] Built Qf(u) and Qw(u) distributions")

# ================================
# 4. 计算 JSf / JSw
# ================================
jsf_list, jsw_list = [], []

for uid in train_users:
    if uid not in Qf_u_norm or uid not in F_u_norm:
        continue

    rels = set(F_u_norm[uid].keys()) | set(Qf_u_norm[uid].keys())
    P = np.array([F_u_norm[uid].get(r, 0) for r in rels])
    Qf = np.array([Qf_u_norm[uid].get(r, 0) for r in rels])
    Qw = np.array([Qw_u_norm[uid].get(r, 0) for r in rels])

    jsf_list.append(jensenshannon(P, Qf)**2)
    jsw_list.append(jensenshannon(P, Qw)**2)

JSf = np.mean(jsf_list)
JSw = np.mean(jsw_list)

print(f"JSf = {JSf:.6f}")
print(f"JSw = {JSw:.6f}")



# import pickle

# PATH_FILE = r"tmp\Amazon_Beauty\train_agent\policy_paths_epoch_50.pkl"

# with open(PATH_FILE, "rb") as f:
#     data = pickle.load(f)

# paths = data["paths"]

# relation_set = set()

# for path in paths:
#     for rel, node_type, node_id in path:
#         if rel != "self_loop":  # 去掉 self_loop
#             relation_set.add(rel)

# # 查看所有关系类型
# relation_list = sorted(list(relation_set))
# print("Total relations =", len(relation_list))
# for r in relation_list:
#     print(r)

