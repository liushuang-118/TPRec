from __future__ import absolute_import, division, print_function

from easydict import EasyDict as edict
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from timeUtils import *
from data_utils import AmazonDataset


class KnowledgeEmbedding(nn.Module):
    def __init__(self, dataset, args):
        super(KnowledgeEmbedding, self).__init__()
        self.embed_size = args.embed_size
        self.num_neg_samples = args.num_neg_samples
        self.device = args.device
        self.l2_lambda = args.l2_lambda
        self.init_embeds = load_init_embed(args.dataset)

        # Initialize entity embeddings.
        self.entities = edict(
            user=edict(vocab_size=dataset.user.vocab_size),
            product=edict(vocab_size=dataset.product.vocab_size),
            word=edict(vocab_size=dataset.word.vocab_size),
            related_product=edict(vocab_size=dataset.related_product.vocab_size),
            brand=edict(vocab_size=dataset.brand.vocab_size),
            category=edict(vocab_size=dataset.category.vocab_size),
        )
        for e in self.entities:
            embed = self._entity_embedding(self.entities[e].vocab_size, e)
            setattr(self, e, embed)

        # Initialize relation embeddings and relation biases.
        self.relations = edict(
            produced_by=edict(
                et='brand',
                et_distrib=self._make_distrib(dataset.produced_by.et_distrib)),
            belongs_to=edict(
                et='category',
                et_distrib=self._make_distrib(dataset.belongs_to.et_distrib)),
            also_bought=edict(
                et='related_product',
                et_distrib=self._make_distrib(dataset.also_bought.et_distrib)),
            also_viewed=edict(
                et='related_product',
                et_distrib=self._make_distrib(dataset.also_viewed.et_distrib)),
            bought_together=edict(
                et='related_product',
                et_distrib=self._make_distrib(dataset.bought_together.et_distrib)),
        )
        for ti in range(CLUSNUM):
            self.relations[PURCHASE[ti]] = edict(
                et='product',
                et_distrib=self._make_distrib(dataset.review.product_uniform_distrib))
            self.relations[MENTION[ti]] = edict(
                et='word',
                et_distrib=self._make_distrib(dataset.review.word_distrib))
            self.relations[DESCRIBED_AS[ti]] = edict(
                et='word',
                et_distrib=self._make_distrib(dataset.review.word_distrib))

        for r in self.relations:
            embed = self._relation_embedding(r)
            setattr(self, r, embed)
            bias = self._relation_bias(len(self.relations[r].et_distrib), r)
            setattr(self, r + '_bias', bias)

    def _entity_embedding(self, vocab_size, entities_name):
        """
        Create entity embedding of size [vocab_size+1, embed_size].
        Ensure the weight matrix has shape (vocab_size+1, embed_size) and the
        last row is the padding/dummy vector (all zeros). Set padding_idx = vocab_size.
        """
        num_embeddings = vocab_size + 1
        padding_idx = vocab_size  # last row

        # create embedding module with proper padding_idx
        embed = nn.Embedding(num_embeddings, self.embed_size, padding_idx=padding_idx, sparse=False)

        # load initial weights from init_embeds
        weight_np = np.array(self.init_embeds[entities_name])
        # If init weight has only vocab_size rows, append one zero row for padding.
        if weight_np.shape[0] == vocab_size:
            zero_row = np.zeros((1, weight_np.shape[1]), dtype=weight_np.dtype)
            weight_np = np.vstack([weight_np, zero_row])
        elif weight_np.shape[0] != num_embeddings:
            # Unexpected shape: try to fix by either truncating or padding zeros
            if weight_np.shape[0] > num_embeddings:
                weight_np = weight_np[:num_embeddings, :]
            else:
                # pad additional zero rows
                pad_rows = num_embeddings - weight_np.shape[0]
                pad = np.zeros((pad_rows, weight_np.shape[1]), dtype=weight_np.dtype)
                weight_np = np.vstack([weight_np, pad])

        # convert to tensor and ensure last row is zero
        weight = torch.FloatTensor(weight_np)
        with torch.no_grad():
            weight[padding_idx].zero_()

        # assign Parameter
        embed.weight = nn.Parameter(weight)
        return embed

    def _relation_embedding(self, relation_name):
        """Create relation vector of size [1, embed_size]."""
        np_weight = None
        for remd in self.init_embeds:
            r1 = remd[:7]
            r2 = relation_name[:7]
            if r1 == r2:
                np_weight = self.init_embeds[remd][0]
                break
        if np_weight is None:
            # fallback: zeros
            np_weight = np.zeros((self.embed_size,), dtype=float)
        weight = torch.unsqueeze(torch.FloatTensor(np_weight), 0)
        embed = nn.Parameter(weight)
        return embed

    def _relation_bias(self, vocab_size, relation_name):
        """
        Create relation bias embedding of size [vocab_size+1, 1]
        padding_idx = vocab_size (last row).
        Ensure weight shape matches (vocab_size+1, 1).
        """
        num_embeddings = vocab_size + 1
        padding_idx = vocab_size
        bias = nn.Embedding(num_embeddings, 1, padding_idx=padding_idx, sparse=False)

        np_weight = None
        for remd in self.init_embeds:
            r1 = remd[:7]
            r2 = relation_name[:7]
            if r1 == r2:
                np_weight = self.init_embeds[remd][1]
                break

        if np_weight is None:
            # fallback zeros
            np_weight = np.zeros((num_embeddings, 1), dtype=float)

        weight_arr = np.array(np_weight)

        # If weight only length vocab_size, pad a zero row
        if weight_arr.shape[0] == vocab_size and weight_arr.ndim == 2:
            pad = np.zeros((1, weight_arr.shape[1]), dtype=weight_arr.dtype)
            weight_arr = np.vstack([weight_arr, pad])
        elif weight_arr.shape[0] != num_embeddings:
            # fix by trunc/pad
            if weight_arr.shape[0] > num_embeddings:
                weight_arr = weight_arr[:num_embeddings, :]
            else:
                pad_rows = num_embeddings - weight_arr.shape[0]
                pad = np.zeros((pad_rows, weight_arr.shape[1]), dtype=weight_arr.dtype)
                weight_arr = np.vstack([weight_arr, pad])

        weight = torch.FloatTensor(weight_arr)
        with torch.no_grad():
            weight[padding_idx].zero_()

        bias.weight = nn.Parameter(weight)
        return bias

    def _make_distrib(self, distrib):
        """Normalize input numpy vector to distribution."""
        distrib = np.power(np.array(distrib, dtype=float), 0.75)
        distrib = distrib / distrib.sum()
        distrib = torch.FloatTensor(distrib).to(self.device)
        return distrib

    def forward(self, batch_idxs):
        loss = self.compute_loss(batch_idxs)
        return loss

    def compute_loss(self, batch_idxs):
        """
        Compute knowledge graph negative sampling loss.
        batch_idxs: batch_size * 9 array, where each row is
                (u_id, p_id, w_id, cluster_num, b_id, c_id, rp1, rp2, rp3).
        """
        user_idxs = batch_idxs[:, 0]
        product_idxs = batch_idxs[:, 1]
        word_idxs = batch_idxs[:, 2]
        cluster_num = np.array(batch_idxs[:, 3].cpu())
        brand_idxs = batch_idxs[:, 4]
        category_idxs = batch_idxs[:, 5]
        rproduct1_idxs = batch_idxs[:, 6]
        rproduct2_idxs = batch_idxs[:, 7]
        rproduct3_idxs = batch_idxs[:, 8]

        regularizations = []

        num_id, num_index, num_n = np.unique(cluster_num, return_index=True, return_counts=True)
        up_lo = []
        uw_lo = []
        pw_lo = []
        flag = 0
        if len(num_id) == 1:
            up_loss, up_embeds = self.neg_loss('user', 'purchase_' + str(num_id[0]), 'product', user_idxs, product_idxs)
            regularizations.extend(up_embeds)
            loss = up_loss

            uw_loss, uw_embeds = self.neg_loss('user', 'mention_'  + str(num_id[0]),  'word', user_idxs, word_idxs)
            regularizations.extend(uw_embeds)
            loss += uw_loss

            pw_loss, pw_embeds = self.neg_loss('product',  'described_as_'  + str(num_id[0]), 'word', product_idxs, word_idxs)
            loss += pw_loss
            regularizations.extend(pw_embeds)
        else:
            for index in range(len(num_id)):
                cur_cls = num_id[index]
                startID = num_index[index]
                endID = num_index[index]+num_n[index]
                up_loss, up_embeds = self.neg_loss('user', 'purchase_' + str(cur_cls), 'product', user_idxs[startID:endID], product_idxs[startID:endID])
                up_lo.append(up_loss.unsqueeze(0))
                uw_loss, uw_embeds = self.neg_loss('user', 'mention_'  + str(cur_cls), 'word', user_idxs[startID:endID], word_idxs[startID:endID])
                uw_lo.append(uw_loss.unsqueeze(0))
                pw_loss, pw_embeds = self.neg_loss('product', 'described_as_'  + str(cur_cls), 'word', product_idxs[startID:endID], word_idxs[startID:endID])
                pw_lo.append(pw_loss.unsqueeze(0))

                if flag == 0:
                    flag = 1
                    up_re = up_embeds
                    uw_re = uw_embeds
                    pw_re = pw_embeds
                else:
                    up_re[0] = torch.cat((up_re[0], up_embeds[0]), 0)
                    up_re[1] = torch.cat((up_re[1], up_embeds[1]), 0)
                    up_re[2] = torch.cat((up_re[2], up_embeds[2]), 0)
                    uw_re[0] = torch.cat((uw_re[0], uw_embeds[0]), 0)
                    uw_re[1] = torch.cat((uw_re[1], uw_embeds[1]), 0)
                    uw_re[2] = torch.cat((uw_re[2], uw_embeds[2]), 0)
                    pw_re[0] = torch.cat((pw_re[0], pw_embeds[0]), 0)
                    pw_re[1] = torch.cat((pw_re[1], pw_embeds[1]), 0)
                    pw_re[2] = torch.cat((pw_re[2], pw_embeds[2]), 0)

            loss = torch.cat(up_lo,dim=0).mean()
            loss += torch.cat(uw_lo,dim=0).mean() + torch.cat(pw_lo,dim=0).mean()
            regularizations.extend(up_re)
            regularizations.extend(uw_re)
            regularizations.extend(pw_re)

        pb_loss, pb_embeds = self.neg_loss('product', 'produced_by', 'brand', product_idxs, brand_idxs)
        if pb_loss is not None:
            regularizations.extend(pb_embeds)
            loss += pb_loss

        pc_loss, pc_embeds = self.neg_loss('product', 'belongs_to', 'category', product_idxs, category_idxs)
        if pc_loss is not None:
            regularizations.extend(pc_embeds)
            loss += pc_loss

        pr1_loss, pr1_embeds = self.neg_loss('product', 'also_bought', 'related_product', product_idxs, rproduct1_idxs)
        if pr1_loss is not None:
            regularizations.extend(pr1_embeds)
            loss += pr1_loss

        pr2_loss, pr2_embeds = self.neg_loss('product', 'also_viewed', 'related_product', product_idxs, rproduct2_idxs)
        if pr2_loss is not None:
            regularizations.extend(pr2_embeds)
            loss += pr2_loss

        pr3_loss, pr3_embeds = self.neg_loss('product', 'bought_together', 'related_product', product_idxs, rproduct3_idxs)
        if pr3_loss is not None:
            regularizations.extend(pr3_embeds)
            loss += pr3_loss

        if self.l2_lambda > 0:
            l2_loss = 0.0
            for term in regularizations:
                l2_loss += torch.norm(term)
            loss += self.l2_lambda * l2_loss

        return loss

    def neg_loss(self, entity_head, relation, entity_tail, entity_head_idxs, entity_tail_idxs):
        mask = entity_tail_idxs >= 0
        fixed_entity_head_idxs = entity_head_idxs[mask]
        fixed_entity_tail_idxs = entity_tail_idxs[mask]
        if fixed_entity_head_idxs.size(0) <= 0:
            return None, []

        entity_head_embedding = getattr(self, entity_head)
        entity_tail_embedding = getattr(self, entity_tail)
        relation_vec = getattr(self, relation)
        relation_bias_embedding = getattr(self, relation + '_bias')
        entity_tail_distrib = self.relations[relation].et_distrib

        return kg_neg_loss(entity_head_embedding, entity_tail_embedding,
                           fixed_entity_head_idxs, fixed_entity_tail_idxs,
                           relation_vec, relation_bias_embedding, self.num_neg_samples, entity_tail_distrib)


def kg_neg_loss(entity_head_embed, entity_tail_embed, entity_head_idxs, entity_tail_idxs,
                relation_vec, relation_bias_embed, num_samples, distrib):
    batch_size = entity_head_idxs.size(0)

    # debug prints (可选)
    # print(f"DEBUG: entity_head_idxs max = {int(entity_head_idxs.max())}, min = {int(entity_head_idxs.min())}")
    # print(f"DEBUG: embedding size = {entity_head_embed.weight.shape[0]}")
    # print(f"DEBUG: entity_tail_idxs max = {int(entity_tail_idxs.max())}, min = {int(entity_tail_idxs.min())}")
    # print(f"DEBUG: tail embedding size = {entity_tail_embed.weight.shape[0]}")

    entity_head_vec = entity_head_embed(entity_head_idxs)  # [batch_size, embed_size]
    example_vec = entity_head_vec + relation_vec  # [batch_size, embed_size]
    example_vec = example_vec.unsqueeze(2)  # [batch_size, embed_size, 1]

    entity_tail_vec = entity_tail_embed(entity_tail_idxs)  # [batch_size, embed_size]
    pos_vec = entity_tail_vec.unsqueeze(1)  # [batch_size, 1, embed_size]
    relation_bias = relation_bias_embed(entity_tail_idxs).squeeze(1)  # [batch_size]
    pos_logits = torch.bmm(pos_vec, example_vec).squeeze() + relation_bias  # [batch_size]
    pos_loss = -pos_logits.sigmoid().log()  # [batch_size]
    neg_sample_idx = torch.multinomial(distrib, num_samples, replacement=True).view(-1)
    neg_vec = entity_tail_embed(neg_sample_idx)  # [num_samples, embed_size]
    neg_logits = torch.mm(example_vec.squeeze(2), neg_vec.transpose(1, 0).contiguous())
    neg_logits += relation_bias.unsqueeze(1)  # [batch_size, num_samples]
    neg_loss = -neg_logits.neg().sigmoid().log().sum(1)  # [batch_size]

    loss = (pos_loss + neg_loss).mean()
    return loss, [entity_head_vec, entity_tail_vec, neg_vec]
