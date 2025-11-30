CLUSNUM = 14
USER = 'user' 
PRODUCT = 'product' 
WORD = 'word' 
RPRODUCT = 'related_product' 
BRAND = 'brand' 
CATEGORY = 'category' 
PRODUCED_BY = 'produced_by' 
BELONG_TO = 'belongs_to' 
ALSO_BOUGHT = 'also_bought' 
ALSO_VIEWED = 'also_viewed' 
BOUGHT_TOGETHER = 'bought_together' 
SELF_LOOP = 'self_loop' 
PURCHASE = {0: 'purchase_0', 1: 'purchase_1', 2: 'purchase_2', 3: 'purchase_3', 4: 'purchase_4', 5: 'purchase_5', 6: 'purchase_6', 7: 'purchase_7', 8: 'purchase_8', 9: 'purchase_9', 10: 'purchase_10', 11: 'purchase_11', 12: 'purchase_12', 13: 'purchase_13'}
MENTION = {0: 'mention_0', 1: 'mention_1', 2: 'mention_2', 3: 'mention_3', 4: 'mention_4', 5: 'mention_5', 6: 'mention_6', 7: 'mention_7', 8: 'mention_8', 9: 'mention_9', 10: 'mention_10', 11: 'mention_11', 12: 'mention_12', 13: 'mention_13'}
DESCRIBED_AS = {0: 'described_as_0', 1: 'described_as_1', 2: 'described_as_2', 3: 'described_as_3', 4: 'described_as_4', 5: 'described_as_5', 6: 'described_as_6', 7: 'described_as_7', 8: 'described_as_8', 9: 'described_as_9', 10: 'described_as_10', 11: 'described_as_11', 12: 'described_as_12', 13: 'described_as_13'}
KG_RELATION = { 
USER: { 
'purchase_0': PRODUCT, 
'mention_0': WORD, 
'purchase_1': PRODUCT, 
'mention_1': WORD, 
'purchase_2': PRODUCT, 
'mention_2': WORD, 
'purchase_3': PRODUCT, 
'mention_3': WORD, 
'purchase_4': PRODUCT, 
'mention_4': WORD, 
'purchase_5': PRODUCT, 
'mention_5': WORD, 
'purchase_6': PRODUCT, 
'mention_6': WORD, 
'purchase_7': PRODUCT, 
'mention_7': WORD, 
'purchase_8': PRODUCT, 
'mention_8': WORD, 
'purchase_9': PRODUCT, 
'mention_9': WORD, 
'purchase_10': PRODUCT, 
'mention_10': WORD, 
'purchase_11': PRODUCT, 
'mention_11': WORD, 
'purchase_12': PRODUCT, 
'mention_12': WORD, 
'purchase_13': PRODUCT, 
'mention_13': WORD, 
}, 
WORD: { 
'mention_0': USER, 
'described_as_0': PRODUCT, 
'mention_1': USER, 
'described_as_1': PRODUCT, 
'mention_2': USER, 
'described_as_2': PRODUCT, 
'mention_3': USER, 
'described_as_3': PRODUCT, 
'mention_4': USER, 
'described_as_4': PRODUCT, 
'mention_5': USER, 
'described_as_5': PRODUCT, 
'mention_6': USER, 
'described_as_6': PRODUCT, 
'mention_7': USER, 
'described_as_7': PRODUCT, 
'mention_8': USER, 
'described_as_8': PRODUCT, 
'mention_9': USER, 
'described_as_9': PRODUCT, 
'mention_10': USER, 
'described_as_10': PRODUCT, 
'mention_11': USER, 
'described_as_11': PRODUCT, 
'mention_12': USER, 
'described_as_12': PRODUCT, 
'mention_13': USER, 
'described_as_13': PRODUCT, 
}, 
PRODUCT: { 
'purchase_0': USER, 
'described_as_0': WORD, 
'purchase_1': USER, 
'described_as_1': WORD, 
'purchase_2': USER, 
'described_as_2': WORD, 
'purchase_3': USER, 
'described_as_3': WORD, 
'purchase_4': USER, 
'described_as_4': WORD, 
'purchase_5': USER, 
'described_as_5': WORD, 
'purchase_6': USER, 
'described_as_6': WORD, 
'purchase_7': USER, 
'described_as_7': WORD, 
'purchase_8': USER, 
'described_as_8': WORD, 
'purchase_9': USER, 
'described_as_9': WORD, 
'purchase_10': USER, 
'described_as_10': WORD, 
'purchase_11': USER, 
'described_as_11': WORD, 
'purchase_12': USER, 
'described_as_12': WORD, 
'purchase_13': USER, 
'described_as_13': WORD, 
PRODUCED_BY: BRAND, 
BELONG_TO: CATEGORY, 
ALSO_BOUGHT: RPRODUCT, 
ALSO_VIEWED: RPRODUCT, 
BOUGHT_TOGETHER: RPRODUCT, 
}, 
BRAND: { 
PRODUCED_BY: PRODUCT, 
}, 
CATEGORY: { 
BELONG_TO: PRODUCT, 
}, 
RPRODUCT: { 
ALSO_BOUGHT: PRODUCT, 
ALSO_VIEWED: PRODUCT, 
BOUGHT_TOGETHER: PRODUCT, 
}} 
