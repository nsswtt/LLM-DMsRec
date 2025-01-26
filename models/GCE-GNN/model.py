import datetime
import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from aggregator import LocalAggregator, GlobalAggregator
from torch.nn import Module, Parameter
import torch.nn.functional as F
import pickle


class CombineGraph(Module):
    def __init__(self, opt, num_node, adj_all, num):
        super(CombineGraph, self).__init__()
        self.opt = opt

        self.batch_size = opt.batch_size
        self.num_node = num_node
        self.dim = opt.hiddenSize
        self.dropout_local = opt.dropout_local
        self.dropout_global = opt.dropout_global
        self.hop = opt.n_iter
        self.sample_num = opt.n_sample
        self.adj_all = trans_to_cuda(torch.Tensor(adj_all)).long()
        self.num = trans_to_cuda(torch.Tensor(num)).float()

        # Aggregator
        self.local_agg = LocalAggregator(self.dim, self.opt.alpha, dropout=0.0)
        self.global_agg = []
        for i in range(self.hop):
            if opt.activate == 'relu':
                agg = GlobalAggregator(self.dim, opt.dropout_gcn, act=torch.relu)
            else:
                agg = GlobalAggregator(self.dim, opt.dropout_gcn, act=torch.tanh)
            self.add_module('agg_gcn_{}'.format(i), agg)
            self.global_agg.append(agg)

        # Item representation & Position representation
        self.embedding = nn.Embedding(num_node, self.dim)
        self.pos_embedding = nn.Embedding(256, self.dim)

        # Parameters
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.glu1 = nn.Linear(self.dim, self.dim)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=False)
        # self.linear_transform = nn.Linear(self.dim, self.dim, bias=False)

        self.leakyrelu = nn.LeakyReLU(opt.alpha)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        
        self.w = nn.Parameter(torch.Tensor(self.dim, 768))
        self.b = nn.Parameter(torch.Tensor(self.dim))
        self.linear_transform = nn.Linear(self.dim * 3, self.dim, bias=False)

        # self.attention_layer = torch.nn.Sequential(
        #     torch.nn.Linear(768, 128),  # 将每个意图的表示压缩到128维
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(0.1),
        #     torch.nn.Linear(128, 1)     # 输出一个单一的权重值
        # )
        # self.importance_fn = nn.Linear(768 * 2, 1)
        # self.beta = nn.Parameter(torch.Tensor([0.0]))
        # self.alpha = nn.Parameter(torch.Tensor([0.0, 0.0]))
        # self.alpha_0 = nn.Parameter(torch.randn(1, self.dim))  # 初始化为随机值
        # self.alpha_0.data = torch.sigmoid(self.alpha_0.data)  # 确保 alpha_0 在 0 到 1 之间
       
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def sample(self, target, n_sample):
        # neighbor = self.adj_all[target.view(-1)]
        # index = np.arange(neighbor.shape[1])
        # np.random.shuffle(index)
        # index = index[:n_sample]
        # return self.adj_all[target.view(-1)][:, index], self.num[target.view(-1)][:, index]
        return self.adj_all[target.view(-1)], self.num[target.view(-1)]
    
    def contrastive_loss(self, hidden1, hidden2):
        # hidden1 = hidden1.float()
        # hidden2 = hidden2.float()
        hidden1 = F.normalize(hidden1, p=2, dim=1)
        hidden2 = F.normalize(hidden2, p=2, dim=1)

        # 计算相似度
        sim_matrix = torch.matmul(hidden1, hidden2.T) / 0.5
        sim_pos = torch.diag(sim_matrix)
        # 对比损失
        contrastive_loss = -torch.log(
            torch.exp(sim_pos) / torch.exp(sim_matrix).sum(dim=1)
        )
        return contrastive_loss.mean()

    def SSL(self, hidden1, hidden2):
        def row_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            return corrupted_embedding

        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            corrupted_embedding = corrupted_embedding[:, torch.randperm(corrupted_embedding.size()[1])]
            return corrupted_embedding

        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), 1)

        hidden1 = F.normalize(hidden1, p=2, dim=1)
        hidden2 = F.normalize(hidden2, p=2, dim=1)
        pos = score(hidden1, hidden2)
        neg1 = score(hidden2, row_column_shuffle(hidden1))
        one = torch.cuda.FloatTensor(neg1.shape[0]).fill_(1)
        # one = zeros = torch.ones(neg1.shape[0])
        con_loss = torch.sum(-torch.log(1e-8 + torch.sigmoid(pos)) - torch.log(1e-8 + (one - torch.sigmoid(neg1))))
        return con_loss
    
    def KLAlignmentModel(self, hidden1, hidden2):
        # hidden1_n = F.normalize(hidden1, p=2, dim=1)
        # hidden2_n = F.normalize(hidden2, p=2, dim=1)
        # la = F.mse_loss(hidden1_n, hidden2_n)
        p_hidden1 = F.softmax(hidden1, dim=-1) + 1e-8
        p_hidden2 = F.softmax(hidden2, dim=-1) + 1e-8
        kl_loss = F.kl_div(p_hidden2.log(), p_hidden1, reduction='batchmean')
        # kl_loss = F.kl_div(torch.log(p_hidden2), p_hidden1, reduction='batchmean')
        return kl_loss
    
    def info_nce_loss(self, hidden1, hidden2, temperature=0.07):
        batch_size = hidden1.size(0)
        similarity_matrix = F.cosine_similarity(hidden1.unsqueeze(1), hidden2.unsqueeze(0), dim=-1) / temperature
        labels = torch.arange(batch_size).to(hidden1.device)
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss
    
    def direct_AU(self, hidden1, hidden2):
        hidden1, hidden2 = F.normalize(hidden1, dim=-1), F.normalize(hidden2, dim=-1)
        align = (hidden1 - hidden2).norm(p=2, dim=1).pow(2).mean()

        uniform = (torch.pdist(hidden1, p=2).pow(2).mul(-2).exp().mean().log() + torch.pdist(hidden2, p=2).pow(2).mul(-2).exp().mean().log()) / 2
        loss = align + uniform * 0.5

        return loss
    
    def sparsemax(self, input_tensor):
        """
        Compute the Sparsemax activation on the input tensor.
        
        Args:
            input_tensor (torch.Tensor): The input tensor to apply Sparsemax on.
        
        Returns:
            torch.Tensor: The output tensor with Sparsemax applied.
        """
        input_sorted, _ = torch.sort(input_tensor, dim=1, descending=True)
        cumulative_sum = torch.cumsum(input_sorted, dim=1)
        m = input_tensor.size(-1)  # The number of elements in the last dimension
        rho = torch.arange(1, m + 1, device=input_tensor.device).float()
        threshold = (cumulative_sum - 1) / rho
        support = (input_sorted > threshold.unsqueeze(-1)).sum(dim=-1)
        thresholded_input = torch.gather(input_sorted, dim=-1, index=support.unsqueeze(-1).long() - 1)
        return F.relu(input_tensor - thresholded_input.unsqueeze(-1))

    def compute_scores(self, hidden, mask, explicit_responses, latent_responses):
        # 求最大值和均值
        # mul_responses_max, _ = torch.max(mul_responses, dim=1)
        # mul_responses_mean = torch.sum(mul_responses, dim=1) / intent_num.unsqueeze(1)
        # fusion_response = mul_responses_mean
        # fusion_response = F.dropout(fusion_response, 0.3, training=self.training)
        # explicit_response = torch.matmul(fusion_response, self.w.T) + self.b



        # 注意力机制
        # attention_scores = self.attention_layer(mul_responses)  # shape: (batch_size, intent_num, 1)
        # attention_weights = F.softmax(attention_scores, dim=1)  # shape: (batch_size, intent_num, 1)
        # weighted_responses = mul_responses * attention_weights  # shape: (batch_size, intent_num, embedding_dim)
        # final_intent_vector = weighted_responses.sum(dim=1)  # shape: (batch_size, embedding_dim)
        # explicit_response = torch.matmul(final_intent_vector, self.w.T) + self.b

        # 重要性选择策略
        # a, _ = torch.max(mul_responses, dim=-1)
        # seq_mask = a.gt(0)
        # output_w_mean = torch.cat([mul_responses[:, 0, :].unsqueeze(1).repeat(1, mul_responses.size(1), 1), mul_responses], dim=-1)
        # importance = self.importance_fn(output_w_mean).to(torch.double)
        # importance = torch.where(seq_mask.unsqueeze(-1), importance, -9e15)
        # # alpha_for_entmax = self.entmax_alpha
        # # gamma_prob = entmax_bisect(importance, alpha_for_entmax, dim=1)
        # gamma_prob = torch.softmax(importance, dim=1)
        # output = mul_responses * gamma_prob.to(torch.float)
        # output = F.normalize(output, dim=2) # [B, L+1, D]
        # max_logits = torch.max(output, dim=1)[0]  # [B, N]  # logits[1] = [-1.4918,  0.1595, -2.9134,  ..., -1.2865,  1.9893,  2.8495],
        # output[a == 0] = 0
        # mean_logits = torch.sum(output, dim=1) / torch.sum(a != 0, dim=1, keepdim=True)  # [B, N]
        # mean_logits = mean_logits
        # alpha = F.softmax(self.alpha, dim=0)
        # fusion_response = mean_logits * 0.7 + max_logits * 0.3
        # fusion_response = F.dropout(fusion_response, 0.2, training=self.training)
        # explicit_response = torch.matmul(fusion_response, self.w.T) + self.b
        
        explicit_responses = F.dropout(explicit_responses, 0.6, training=self.training)
        latent_responses = F.dropout(latent_responses, 0.6, training=self.training)
        explicit_responses = torch.matmul(explicit_responses, self.w.T) + self.b
        latent_responses = torch.matmul(latent_responses, self.w.T) + self.b

        mask = mask.float().unsqueeze(-1)

        batch_size = hidden.shape[0]
        len = hidden.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * hidden, 1)

        con_loss = self.KLAlignmentModel(select, explicit_responses) + self.KLAlignmentModel(select, latent_responses)
        select = self.linear_transform(torch.cat([select, explicit_responses, latent_responses], 1))
        # select = select * explicit_response

        b = self.embedding.weight[1:]  # n_nodes x latent_size
        scores = torch.matmul(select, b.transpose(1, 0))
        return scores, con_loss * 0.7

    def forward(self, inputs, adj, mask_item, item):
        batch_size = inputs.shape[0]
        seqs_len = inputs.shape[1]
        h = self.embedding(inputs)

        # local
        h_local = self.local_agg(h, adj, mask_item)

        # global
        item_neighbors = [inputs]
        weight_neighbors = []
        support_size = seqs_len

        for i in range(1, self.hop + 1):
            item_sample_i, weight_sample_i = self.sample(item_neighbors[-1], self.sample_num)
            support_size *= self.sample_num
            item_neighbors.append(item_sample_i.view(batch_size, support_size))
            weight_neighbors.append(weight_sample_i.view(batch_size, support_size))

        entity_vectors = [self.embedding(i) for i in item_neighbors]
        weight_vectors = weight_neighbors

        session_info = []
        item_emb = self.embedding(item) * mask_item.float().unsqueeze(-1)
        
        # mean 
        sum_item_emb = torch.sum(item_emb, 1) / torch.sum(mask_item.float(), -1).unsqueeze(-1)
        
        # sum
        # sum_item_emb = torch.sum(item_emb, 1)
        
        sum_item_emb = sum_item_emb.unsqueeze(-2)
        for i in range(self.hop):
            session_info.append(sum_item_emb.repeat(1, entity_vectors[i].shape[1], 1))

        for n_hop in range(self.hop):
            entity_vectors_next_iter = []
            shape = [batch_size, -1, self.sample_num, self.dim]
            for hop in range(self.hop - n_hop):
                aggregator = self.global_agg[n_hop]
                vector = aggregator(self_vectors=entity_vectors[hop],
                                    neighbor_vector=entity_vectors[hop+1].view(shape),
                                    masks=None,
                                    batch_size=batch_size,
                                    neighbor_weight=weight_vectors[hop].view(batch_size, -1, self.sample_num),
                                    extra_vector=session_info[hop])
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        h_global = entity_vectors[0].view(batch_size, seqs_len, self.dim)

        # combine
        h_local = F.dropout(h_local, self.dropout_local, training=self.training)
        h_global = F.dropout(h_global, self.dropout_global, training=self.training)
        output = h_local + h_global

        return output


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, data):
    alias_inputs, adj, items, mask, targets, inputs, explicit_responses, latent_responses = data
    # alias_inputs, adj, items, mask, targets, inputs = data
    alias_inputs = trans_to_cuda(alias_inputs).long()
    items = trans_to_cuda(items).long()
    adj = trans_to_cuda(adj).float()
    mask = trans_to_cuda(mask).long()
    inputs = trans_to_cuda(inputs).long()
    # intent_num = trans_to_cuda(intent_num).long()
    # mul_responses = trans_to_cuda(mul_responses).float()
    # mul_responses = mul_responses.squeeze(2)

    explicit_responses = trans_to_cuda(explicit_responses).float()
    latent_responses = trans_to_cuda(latent_responses).float()

    hidden = model(items, adj, mask, inputs)
    get = lambda index: hidden[index][alias_inputs[index]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    scores, con_loss = model.compute_scores(seq_hidden, mask, explicit_responses, latent_responses)
    # scores, con_loss = model.compute_scores(seq_hidden, mask)
    return targets, scores, con_loss


def train_test(model, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=model.batch_size,
                                               shuffle=True, pin_memory=True)
    for data in tqdm(train_loader):
        model.optimizer.zero_grad()
        targets, scores, con_loss = forward(model, data)
        targets = trans_to_cuda(targets).long()
        loss = model.loss_function(scores, targets - 1) + con_loss
        loss.backward()
        model.optimizer.step()
        total_loss += loss
    print('\tLoss:\t%.3f' % total_loss)
    model.scheduler.step()
    # test_candidate_ml_items = []
    print('start predicting: ', datetime.datetime.now())
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=model.batch_size,
                                              shuffle=False, pin_memory=True)
    result = []
    hit_5, mrr_5, hit_10, mrr_10, hit_20, mrr_20 = [], [], [], [], [], []
    for data in test_loader:
        targets, scores, con_loss = forward(model, data)
        sub_scores_5 = scores.topk(5)[1]
        sub_scores_5 = trans_to_cpu(sub_scores_5).detach().numpy()

        sub_scores_10 = scores.topk(10)[1]
        sub_scores_10 = trans_to_cpu(sub_scores_10).detach().numpy()

        sub_scores_20 = scores.topk(20)[1]
        # test_candidate_ml_items.append(sub_scores_20)
        sub_scores_20 = trans_to_cpu(sub_scores_20).detach().numpy()

        targets = targets.numpy()
        for score, target, mask in zip(sub_scores_5, targets, test_data.mask):
            hit_5.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr_5.append(0)
            else:
                mrr_5.append(1 / (np.where(score == target - 1)[0][0] + 1))
        
        for score, target, mask in zip(sub_scores_10, targets, test_data.mask):
            hit_10.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr_10.append(0)
            else:
                mrr_10.append(1 / (np.where(score == target - 1)[0][0] + 1))
        
        for score, target, mask in zip(sub_scores_20, targets, test_data.mask):
            hit_20.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr_20.append(0)
            else:
                mrr_20.append(1 / (np.where(score == target - 1)[0][0] + 1))
    
    # combined_tensor = torch.cat(test_candidate_ml_items, dim=0)
    # flattened_tensor = combined_tensor.view(-1, 50)
    # flattened_list = flattened_tensor.tolist()
    # with open("test_candidate_ml_items.txt", 'wb') as f:
    #     pickle.dump(flattened_list, f)
    result.append(np.mean(hit_5) * 100)
    result.append(np.mean(mrr_5) * 100)
    result.append(np.mean(hit_10) * 100)
    result.append(np.mean(mrr_10) * 100)
    result.append(np.mean(hit_20) * 100)
    result.append(np.mean(mrr_20) * 100)

    return result
