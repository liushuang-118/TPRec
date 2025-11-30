import os
import pickle
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
from data_utils import AmazonDataset
from knowledge_graph import KnowledgeGraph

# ===== 配置 =====
DATASET_NAME = 'Amazon_Clothing'  # 可改为其他数据集
DATA_DIR = f'./data/{DATASET_NAME}'
PATH_FILE = f'./tmp/{DATASET_NAME}/train_agent/policy_paths_epoch_50.pkl'
TIME_TRAIN_FILE = f'{DATA_DIR}/time_train.csv'
TIME_TEST_FILE = f'{DATA_DIR}/time_test.csv'

TOP_P = 10        # 每个用户推荐 Top-P 个产品
TOP_K_PATHS = 10  # 每个用户前 K 条路径
BETA_LIR = 0.5    # temporal decay factor for LIR
BETA_SEP = 0.5    # 衰减因子用于 SEP

# ===== 加载 reasoning paths =====
with open(PATH_FILE, 'rb') as f:
    data = pickle.load(f)
print(f"[INFO] 已加载 {len(data['paths'])} 条 reasoning paths")

# ===== 使用 KG 全局度数作为实体流行度 =====
dataset = AmazonDataset(DATA_DIR)
KG = KnowledgeGraph(dataset)
KG.compute_degrees()

entity_popularity_normalized = {}
for etype, deg_dict in KG.degrees.items():
    values = list(deg_dict.values())
    min_val, max_val = min(values), max(values)
    if max_val == min_val:
        entity_popularity_normalized[etype] = {eid: 0.0 for eid in deg_dict}
    else:
        entity_popularity_normalized[etype] = {
            eid: (deg - min_val) / (max_val - min_val) for eid, deg in deg_dict.items()
        }
print("[INFO] 全局实体流行度加载完成（来自 KG 度数）")

# ===== 构建用户的 top-P 产品和 top-K 路径 =====
user_products_tmp = defaultdict(list)
user_topk_paths_tmp = defaultdict(list)

for path, probs in zip(data['paths'], data['probs']):
    user_id = None
    last_product_id = None
    path_prob = np.prod(probs)
    for rel, etype, eid in path:
        if etype == 'user' and user_id is None:
            user_id = eid
        elif etype == 'product':
            last_product_id = eid
    if user_id is None or last_product_id is None:
        continue
    user_products_tmp[user_id].append((last_product_id, path_prob))
    user_topk_paths_tmp[user_id].append((path, path_prob))

user_products = {}
user_topk_paths = {}

for uid, items in user_products_tmp.items():
    sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
    user_products[uid] = [pid for pid, _ in sorted_items[:TOP_P]]

for uid, paths in user_topk_paths_tmp.items():
    sorted_paths = sorted(paths, key=lambda x: x[1], reverse=True)
    user_topk_paths[uid] = [p for p, _ in sorted_paths[:TOP_K_PATHS]]

print(f"[INFO] 提取完成: {len(user_products)} 个用户")

# ===== 计算 SEP =====
user_sep = {}
for uid, paths in user_topk_paths.items():
    sep_scores = []
    for path in paths:
        sep_values = [
            entity_popularity_normalized.get(etype, {}).get(eid, 0.0)
            for rel, etype, eid in path if etype != 'word'
        ]
        if not sep_values:
            continue
        sep_score = sep_values[0]
        for v in sep_values[1:]:
            sep_score = (1 - BETA_SEP) * sep_score + BETA_SEP * v
        sep_scores.append(sep_score)
    if sep_scores:
        user_sep[uid] = np.mean(sep_scores)
avg_sep = np.mean(list(user_sep.values()))
print(f"[INFO] 平均 SEP: {avg_sep:.4f}")

# ===== 计算 ETD =====
global_last_rels = set()
for path in data['paths']:
    last_rel = None
    for rel, etype, eid in path:
        if etype != 'word':
            last_rel = rel
    if last_rel:
        global_last_rels.add(last_rel)
total_global_last_rels = len(global_last_rels)

user_etd = {}
for uid, paths in user_topk_paths.items():
    last_rels = set()
    for path in paths:
        last_rel = None
        for rel, etype, eid in path:
            if etype != 'word':
                last_rel = rel
        if last_rel:
            last_rels.add(last_rel)
    user_etd[uid] = len(last_rels) / min(TOP_K_PATHS, total_global_last_rels)
avg_etd = np.mean(list(user_etd.values()))
print(f"[INFO] 平均 ETD: {avg_etd:.4f}")

# ===== 加载时间数据并转换为天数 =====
train_df = pd.read_csv(TIME_TRAIN_FILE)
test_df = pd.read_csv(TIME_TEST_FILE)
all_time_df = pd.concat([train_df, test_df], ignore_index=True)
all_time_df['PURCHASE_Time'] = pd.to_datetime(all_time_df['PURCHASE_Time'], errors='coerce')
all_time_df = all_time_df.dropna(subset=['PURCHASE_Time'])
min_date = all_time_df['PURCHASE_Time'].min()
all_time_df['PURCHASE_Time_days'] = (all_time_df['PURCHASE_Time'] - min_date).dt.days

user_item_time = defaultdict(dict)
for row in all_time_df.itertuples(index=False):
    user_item_time[row.UID][row.PID] = row.PURCHASE_Time_days

# ===== 计算 LIR（考虑 product 和 related_product） =====
user_lir_raw = {}
for uid, paths in user_topk_paths.items():
    lir_scores = []
    for path in paths:
        product_times = [
            user_item_time.get(uid, {}).get(eid)
            for rel, etype, eid in path if etype in ['product', 'related_product']
        ]
        product_times = [t for t in product_times if t is not None]
        if not product_times:
            continue
        product_times.sort()
        lir = product_times[0]
        for t in product_times[1:]:
            lir = (1 - BETA_LIR) * lir + BETA_LIR * t
        lir_scores.append(lir)
    if lir_scores:
        user_lir_raw[uid] = lir_scores

# ===== 每个用户的 LIR min-max 归一化 =====
user_lir = {}
for uid, lir_list in user_lir_raw.items():
    min_lir = min(lir_list)
    max_lir = max(lir_list)
    if max_lir == min_lir:
        user_lir[uid] = 0.0
    else:
        user_lir[uid] = np.mean([(x - min_lir) / (max_lir - min_lir) for x in lir_list])
avg_lir = np.mean(list(user_lir.values()))
print(f"[INFO] 平均 LIR: {avg_lir:.4f}")
