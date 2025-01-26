#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F


class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gnn = GNN(self.hidden_size, step=opt.step)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 4, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        
        self.w = nn.Parameter(torch.Tensor(self.hidden_size, 768))
        self.b = nn.Parameter(torch.Tensor(self.hidden_size))
        # self.linear_transform = nn.Linear(self.dim * 3, self.dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    
    def KLAlignmentModel(self, hidden1, hidden2):
        # hidden1_n = F.normalize(hidden1, p=2, dim=1)
        # hidden2_n = F.normalize(hidden2, p=2, dim=1)
        # la = F.mse_loss(hidden1_n, hidden2_n)
        p_hidden1 = F.softmax(hidden1, dim=-1) + 1e-8
        p_hidden2 = F.softmax(hidden2, dim=-1) + 1e-8
        kl_loss = F.kl_div(p_hidden2.log(), p_hidden1, reduction='batchmean')
        # kl_loss = F.kl_div(torch.log(p_hidden2), p_hidden1, reduction='batchmean')
        return kl_loss

    def compute_scores(self, hidden, mask, explicit_responses, latent_responses):
        explicit_responses = F.dropout(explicit_responses, 0.2, training=self.training)
        latent_responses = F.dropout(latent_responses, 0.2, training=self.training)
        explicit_responses = torch.matmul(explicit_responses, self.w.T) + self.b
        latent_responses = torch.matmul(latent_responses, self.w.T) + self.b

        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        con_loss = self.KLAlignmentModel(a, explicit_responses) + self.KLAlignmentModel(a, latent_responses)
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht, explicit_responses, latent_responses], 1))
        b = self.embedding.weight[1:]  # n_nodes x latent_size
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores, con_loss * 0.0

    def forward(self, inputs, A):
        hidden = self.embedding(inputs)
        hidden = self.gnn(A, hidden)
        return hidden


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


def forward(model, i, data):
    # alias_inputs, A, items, mask, targets = data.get_slice(i)
    alias_inputs, A, items, mask, targets, explicit_responses, latent_responses = data.get_slice(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())

    explicit_responses = trans_to_cuda(torch.Tensor(explicit_responses)).float()
    latent_responses = trans_to_cuda(torch.Tensor(latent_responses)).float()
    hidden = model(items, A)
    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    scores, con_loss = model.compute_scores(seq_hidden, mask, explicit_responses, latent_responses)
    # scores, con_loss = model.compute_scores(seq_hidden, mask)
    return targets, scores, con_loss


def train_test(model, train_data, test_data):
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        targets, scores, con_loss = forward(model, i, train_data)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets - 1) + con_loss
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit_5, mrr_5, hit_10, mrr_10, hit_20, mrr_20 = [], [], [], [], [], []
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets, scores, con_loss = forward(model, i, test_data)
        
        sub_scores_5 = scores.topk(5)[1]
        sub_scores_5 = trans_to_cpu(sub_scores_5).detach().numpy()

        sub_scores_10 = scores.topk(10)[1]
        sub_scores_10 = trans_to_cpu(sub_scores_10).detach().numpy()

        sub_scores_20 = scores.topk(20)[1]
        # test_candidate_items.append(sub_scores_20)
        sub_scores_20 = trans_to_cpu(sub_scores_20).detach().numpy()

        # sub_scores = scores.topk(20)[1]
        # sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        # for score, target, mask in zip(sub_scores, targets, test_data.mask):
        #     hit.append(np.isin(target - 1, score))
        #     if len(np.where(score == target - 1)[0]) == 0:
        #         mrr.append(0)
        #     else:
        #         mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
        
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
    hit_5 = np.mean(hit_5) * 100
    mrr_5 = np.mean(mrr_5) * 100
    hit_10 = np.mean(hit_10) * 100
    mrr_10 = np.mean(mrr_10) * 100
    hit_20 = np.mean(hit_20) * 100
    mrr_20 = np.mean(mrr_20) * 100
    return hit_5, mrr_5, hit_10, mrr_10, hit_20, mrr_20
