import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType

import utils
from spmm import SpecialSpmm, CHUNK_SIZE_FOR_SPMM


class RGCF(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(RGCF, self).__init__(config, dataset)

        # load base para
        self.embedding_size = config['embedding_size']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']

        # generate interaction_matrix
        self.inter_matrix_type = config['inter_matrix_type']
        value_field = self.RATING if self.inter_matrix_type == 'rating' else None
        self.interaction_matrix = dataset.inter_matrix(form='coo', value_field=value_field).astype(np.float32)

        # define layers
        self.user_linear = torch.nn.Linear(in_features=self.n_items, out_features=self.embedding_size, bias=False)
        self.item_linear = torch.nn.Linear(in_features=self.n_users, out_features=self.embedding_size, bias=False)

        # define loss
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # generate intermediate data
        self.adj_matrix = self.get_adj_mat(self.interaction_matrix)
        self.norm_adj_matrix = self.get_norm_mat(self.adj_matrix).to(self.device)

        # for learn adj
        self.spmm = config['spmm']
        self.special_spmm = SpecialSpmm() if self.spmm == 'spmm' else torch.sparse.mm

        self.prune_threshold = config['prune_threshold']
        self.MIM_weight = config['MIM_weight']
        self.tau = config['tau']
        self.aug_ratio = config['aug_ratio']
        self.pool_multi = 10

        self.for_learning_adj()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # parameters initialization
        self.apply(self._init_weights)
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def for_learning_adj(self):
        self.adj_indices = self.norm_adj_matrix.indices()
        self.adj_shape = self.norm_adj_matrix.shape
        self.adj = self.norm_adj_matrix

        inter_data = torch.FloatTensor(self.interaction_matrix.data).to(self.device)
        inter_user = torch.LongTensor(self.interaction_matrix.row).to(self.device)
        inter_item = torch.LongTensor(self.interaction_matrix.col).to(self.device)
        inter_mask = torch.stack([inter_user, inter_item], dim=0)

        self.inter_spTensor = torch.sparse.FloatTensor(inter_mask, inter_data, self.interaction_matrix.shape).coalesce()
        self.inter_spTensor_t = self.inter_spTensor.t().coalesce()

        self.inter_indices = self.inter_spTensor.indices()
        self.inter_shape = self.inter_spTensor.shape

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            normal_(module.weight.data, 0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
        elif isinstance(module, nn.Embedding):
            normal_(module.weight.data, 0, 0.01)

    # Returns: torch.FloatTensor: The embedding tensor of all user, shape: [n_users, embedding_size]
    def get_all_user_embedding(self):
        all_user_embedding = torch.sparse.mm(self.inter_spTensor, self.user_linear.weight.t())
        return all_user_embedding

    def get_all_item_embedding(self):
        all_item_embedding = torch.sparse.mm(self.inter_spTensor_t, self.item_linear.weight.t())
        return all_item_embedding

    # Generate adj
    def get_adj_mat(self, inter_M, data=None):
        if data is None:
            data = [1] * inter_M.data
        inter_M_t = inter_M.transpose()
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), data))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), data)))
        A._update(data_dict)  # dok_matrix
        return A

    def get_norm_mat(self, A):
        r""" A_{hat} = D^{-0.5} \times A \times D^{-0.5} """
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        SparseL = utils.sp2tensor(L)
        return SparseL

    # Learn adj
    def sp_cos_sim(self, a, b, eps=1e-8, CHUNK_SIZE=CHUNK_SIZE_FOR_SPMM):
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))

        L = self.inter_indices.shape[1]
        sims = torch.zeros(L, dtype=a.dtype).to(self.device)
        for idx in range(0, L, CHUNK_SIZE):
            batch_indices = self.inter_indices[:, idx:idx + CHUNK_SIZE]

            a_batch = torch.index_select(a_norm, 0, batch_indices[0, :])
            b_batch = torch.index_select(b_norm, 0, batch_indices[1, :])

            dot_prods = torch.mul(a_batch, b_batch).sum(1)
            sims[idx:idx + CHUNK_SIZE] = dot_prods

        return torch.sparse_coo_tensor(self.inter_indices, sims, size=self.interaction_matrix.shape,
                                       dtype=sims.dtype).coalesce()

    def get_sim_mat(self):
        user_feature = self.get_all_user_embedding().to(self.device)
        item_feature = self.get_all_item_embedding().to(self.device)
        sim_inter = self.sp_cos_sim(user_feature, item_feature)
        return sim_inter

    def inter2adj(self, inter):
        inter_t = inter.t().coalesce()
        data = inter.values()
        data_t = inter_t.values()
        adj_data = torch.cat([data, data_t], dim=0)
        adj = torch.sparse.FloatTensor(self.adj_indices, adj_data, self.adj_shape).to(self.device).coalesce()
        return adj

    def get_sim_adj(self, pruning):
        sim_mat = self.get_sim_mat()
        sim_adj = self.inter2adj(sim_mat)

        # pruning
        sim_value = torch.div(torch.add(sim_adj.values(), 1), 2)
        pruned_sim_value = torch.where(sim_value < pruning, torch.zeros_like(sim_value),
                                       sim_value) if pruning > 0 else sim_value
        pruned_sim_adj = torch.sparse.FloatTensor(sim_adj.indices(), pruned_sim_value, self.adj_shape).coalesce()
        self.pruned_sim_adj = pruned_sim_adj

        # normalize
        pruned_sim_indices = pruned_sim_adj.indices()
        diags = torch.sparse.sum(pruned_sim_adj, dim=1).to_dense() + 1e-7
        diags = torch.pow(diags, -1)
        diag_lookup = diags[pruned_sim_indices[0, :]]

        pruned_sim_adj_value = pruned_sim_adj.values()
        normal_sim_value = torch.mul(pruned_sim_adj_value, diag_lookup)
        normal_sim_adj = torch.sparse.FloatTensor(pruned_sim_indices, normal_sim_value,
                                                  self.adj_shape).to(self.device).coalesce()

        return normal_sim_adj

    def ssl_triple_loss(self, z1: torch.Tensor, z2: torch.Tensor, all_emb: torch.Tensor):
        norm_emb1 = F.normalize(z1)
        norm_emb2 = F.normalize(z2)
        norm_all_emb = F.normalize(all_emb)
        pos_score = torch.mul(norm_emb1, norm_emb2).sum(dim=1)
        ttl_score = torch.matmul(norm_emb1, norm_all_emb.transpose(0, 1))
        pos_score = torch.exp(pos_score / self.tau)
        ttl_score = torch.exp(ttl_score / self.tau).sum(dim=1)

        ssl_loss = -torch.log(pos_score / ttl_score).sum()
        return ssl_loss

    def cal_cos_sim(self, u_idx, i_idx, eps=1e-8, CHUNK_SIZE=CHUNK_SIZE_FOR_SPMM):
        user_feature = self.get_all_user_embedding().to(self.device)
        item_feature = self.get_all_item_embedding().to(self.device)

        L = u_idx.shape[0]
        sims = torch.zeros(L, dtype=user_feature.dtype).to(self.device)
        for idx in range(0, L, CHUNK_SIZE):
            a_batch = torch.index_select(user_feature, 0, u_idx[idx:idx + CHUNK_SIZE])
            b_batch = torch.index_select(item_feature, 0, i_idx[idx:idx + CHUNK_SIZE])
            dot_prods = torch.mul(a_batch, b_batch).sum(1)
            sims[idx:idx + CHUNK_SIZE] = dot_prods
        return sims

    def get_aug_adj(self, adj):
        # random sampling
        aug_user = torch.from_numpy(np.random.choice(self.n_users,
                                                     int(adj._nnz() * self.aug_ratio * 0.5 * self.pool_multi))).to(self.device)
        aug_item = torch.from_numpy(np.random.choice(self.n_items,
                                                     int(adj._nnz() * self.aug_ratio * 0.5 * self.pool_multi))).to(self.device)

        # consider reliability
        cos_sim = self.cal_cos_sim(aug_user, aug_item)
        val, idx = torch.topk(cos_sim, int(adj._nnz() * self.aug_ratio * 0.5))
        aug_user = aug_user[idx]
        aug_item = aug_item[idx]

        aug_indices = torch.stack([aug_user, aug_item + self.n_users], dim=0)
        aug_value = torch.ones_like(aug_user) * torch.median(adj.values())
        sub_aug = torch.sparse.FloatTensor(aug_indices, aug_value, adj.shape).coalesce()
        aug = sub_aug + sub_aug.t()
        aug_adj = (adj + aug).coalesce()

        aug_adj_indices = aug_adj.indices()
        diags = torch.sparse.sum(aug_adj, dim=1).to_dense() + 1e-7
        diags = torch.pow(diags, -0.5)
        diag_lookup = diags[aug_adj_indices[0, :]]

        value_DA = diag_lookup.mul(aug_adj.values())
        normal_aug_value = value_DA.mul(diag_lookup)
        normal_aug_adj = torch.sparse.FloatTensor(aug_adj_indices, normal_aug_value,
                                                  self.norm_adj_matrix.shape).to(self.device).coalesce()
        return normal_aug_adj

    # Train
    def forward(self, pruning=0.0, epoch_idx=0):
        user_embeddings = self.get_all_user_embedding()
        item_embeddings = self.get_all_item_embedding()
        all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        embeddings_list = [all_embeddings]

        self.adj = self.norm_adj_matrix if pruning < 0.0 else self.get_sim_adj(pruning)
        for _ in range(self.n_layers):
            all_embeddings = self.special_spmm(self.adj, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings

    def ssl_forward(self, epoch_idx=0):
        user_embeddings = self.get_all_user_embedding()
        item_embeddings = self.get_all_item_embedding()
        all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        embeddings_list = [all_embeddings]

        self.aug_adj = self.get_aug_adj(self.adj.detach())
        for _ in range(self.n_layers):
            all_embeddings = self.special_spmm(self.aug_adj, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction, epoch_idx, tensorboard):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        # obtain embedding
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward(pruning=self.prune_threshold, epoch_idx=epoch_idx)
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)
        loss = mf_loss

        # calculate L2 reg
        if self.reg_weight > 0.:
            user_embeddings = self.get_all_user_embedding()
            item_embeddings = self.get_all_item_embedding()
            u_ego_embeddings = user_embeddings[user]
            pos_ego_embeddings = item_embeddings[pos_item]
            neg_ego_embeddings = item_embeddings[neg_item]
            reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings).squeeze()
            loss += self.reg_weight * reg_loss

        # calculate agreement
        if self.MIM_weight > 0.:
            aug_user_all_embeddings, _ = self.ssl_forward()
            aug_u_embeddings = aug_user_all_embeddings[user]
            mutual_info = self.ssl_triple_loss(u_embeddings, aug_u_embeddings, aug_user_all_embeddings)
            loss += self.MIM_weight * mutual_info

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward(pruning=self.prune_threshold)

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward(pruning=self.prune_threshold)

        self.restore_user_e, self.restore_item_e = self.forward(pruning=self.prune_threshold)

        u_embeddings = self.restore_user_e[user]
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))
        return scores.view(-1)
