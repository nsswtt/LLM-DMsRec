import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from tqdm import tqdm
from layers import *
from Modules import *

def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)



class HGNN_ATT(nn.Module):
    def __init__(self, dataset, input_size, n_hid, output_size, step, dropout=0.3):
        super(HGNN_ATT, self).__init__()
        self.dropout = dropout
        self.step = step
        self.dataset = dataset
        self.gat1 = HyperGraphAttentionLayerSparse(input_size, n_hid, self.dropout, 0.2, transfer=False, concat=False)
        self.gat2 = HyperGraphAttentionLayerSparse(n_hid, output_size, self.dropout, 0.2, transfer=True,  concat=False)
        
    def forward(self, x, H, G, EG):   

        residual = x

        x,y = self.gat1(x, H)

        if self.step == 2:

            x = F.dropout(x, self.dropout, training=self.training)
            x += residual
            x,y = self.gat2(x, H)

        x = F.dropout(x, self.dropout, training=self.training)
        x += residual

        return x, x



class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.embedding2 = nn.Embedding(self.n_node, self.hidden_size)
        self.dropout = opt.dropout
        self.dataset = opt.dataset
        # for self-attention
        n_layers = 1
        n_head = 1
   
        
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.layer_norm1 = nn.LayerNorm(self.hidden_size, eps=1e-6)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(self.hidden_size, self.hidden_size, n_head, self.hidden_size, self.hidden_size, dropout=opt.dropout)
            for _ in range(n_layers)])
        
        self.w = nn.Parameter(torch.Tensor(self.hidden_size, 768))
        self.b = nn.Parameter(torch.Tensor(self.hidden_size))
        self.linear_transform = nn.Linear(self.hidden_size * 4, self.hidden_size, bias=True)

        self.reset_parameters()
        

        self.hgnn = HGNN_ATT(self.dataset, self.hidden_size, self.hidden_size, self.hidden_size, opt.step, dropout = self.dropout)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

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

    def compute_scores(self, enc_output, enc_output2, mask, edge_mask, hidden, explicit_responses, latent_responses):
        explicit_responses = F.dropout(explicit_responses, 0.3, training=self.training)
        latent_responses = F.dropout(latent_responses, 0.3, training=self.training)
        explicit_responses = torch.matmul(explicit_responses, self.w.T) + self.b
        latent_responses = torch.matmul(latent_responses, self.w.T) + self.b

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask = get_pad_mask(mask, 0))
                    
        ht1 = enc_output[torch.arange(mask.shape[0]).long(), mask.shape[1]-1]  # batch_size x latent_size

        ht = self.layer_norm(ht1)

        con_loss = self.KLAlignmentModel(ht1, explicit_responses) + self.KLAlignmentModel(ht1, latent_responses)
        ht = self.linear_transform(torch.cat([ht1, ht, explicit_responses, latent_responses], 1))

        hidden = ht

        b = self.embedding.weight[1:]  # n_nodes x latent_size
        scores = torch.matmul(hidden, b.transpose(1, 0))

        return scores, con_loss * 0.2




    def forward(self, inputs, HT, G, EG): 
        nodes = self.embedding(inputs) 
        #nodes = self.layer_norm1(nodes)       
        nodes, hidden = self.hgnn(nodes, HT, G, EG)
        nodes2 = self.embedding2(inputs) 
        return nodes,hidden,nodes2


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


def forward(model, alias_inputs, H, HT, G, EG, items, targets, node_masks, edge_mask, edge_inputs, explicit_responses, latent_responses):
    
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    HT = trans_to_cuda(torch.Tensor(HT).float())
    G = trans_to_cuda(torch.Tensor(G).float())
    EG = trans_to_cuda(torch.Tensor(EG).float())
    node_masks = trans_to_cuda(torch.Tensor(node_masks).long())
    edge_mask = trans_to_cuda(torch.Tensor(edge_mask).long())
    
    explicit_responses = trans_to_cuda(torch.Tensor(explicit_responses).float())
    latent_responses = trans_to_cuda(torch.Tensor(latent_responses).float())

    nodes, hidden, nodes2 = model(items, HT, G, EG)
    get = lambda i: nodes[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])

    get2 = lambda i: nodes2[i][alias_inputs[i]]
    seq_hidden2 = torch.stack([get2(i) for i in torch.arange(len(alias_inputs)).long()])
    scores, con_loss = model.compute_scores(seq_hidden, seq_hidden2, node_masks, edge_mask, hidden, explicit_responses, latent_responses)
    # scores, con_loss = model.compute_scores(seq_hidden, seq_hidden2, node_masks, edge_mask, hidden)
    return targets, scores, con_loss


def train_model(model, train_data, opt):
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(opt.batchSize, True)
    for step in tqdm(range(len(slices)), total=len(slices), ncols=70, leave=False, unit='b'):
        i = slices[step]
        alias_inputs, H, HT, G, EG, items, targets, node_masks, edge_mask, edge_inputs, explicit_responses, latent_responses = train_data.get_slice(i)
        # alias_inputs, H, HT, G, EG, items, targets, node_masks, edge_mask, edge_inputs = train_data.get_slice(i)    
        model.optimizer.zero_grad()
        targets, scores, con_loss = forward(model, alias_inputs, H, HT, G, EG, items, targets, node_masks, edge_mask, edge_inputs, explicit_responses, latent_responses)
        # targets, scores, con_loss = forward(model, alias_inputs, H, HT, G, EG, items, targets, node_masks, edge_mask, edge_inputs)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets - 1) + con_loss
        loss.backward()
        model.optimizer.step()
        total_loss += loss
    print('\tLoss:\t%.3f' % total_loss)

def test_model(model, test_data, opt):
    
    model.eval()
    hit20, mrr20, hit10, mrr10, hit5, mrr5 = [], [], [], [], [], []
    slices = test_data.generate_batch(min(128,test_data.length), False)
    for step in tqdm(range(len(slices)), total=len(slices), ncols=70, leave=False, unit='b'):
        i = slices[step]
        alias_inputs, H, HT, G, EG, items, targets, node_masks, edge_mask, edge_inputs, explicit_responses, latent_responses = test_data.get_slice(i)
        # alias_inputs, H, HT, G, EG, items, targets, node_masks, edge_mask, edge_inputs = test_data.get_slice(i)
        targets, scores, con_loss = forward(model, alias_inputs, H, HT, G, EG, items, targets, node_masks, edge_mask, edge_inputs, explicit_responses, latent_responses)
        # targets, scores, con_loss = forward(model, alias_inputs, H, HT, G, EG, items, targets, node_masks, edge_mask, edge_inputs)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()

        for score, target in zip(sub_scores, targets):
            hit20.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr20.append(0)
            else:
                mrr20.append(1.0 / (np.where(score == target - 1)[0][0] + 1))

            hit10.append(np.isin(target - 1, score[:10]))
            if len(np.where(score[:10] == target - 1)[0]) == 0:
                mrr10.append(0)
            else:
                mrr10.append(1.0 / (np.where(score[:10] == target - 1)[0][0] + 1))

            hit5.append(np.isin(target - 1, score[:5]))
            if len(np.where(score[:5] == target - 1)[0]) == 0:
                mrr5.append(0)
            else:
                mrr5.append(1.0 / (np.where(score[:5] == target - 1)[0][0] + 1))
    hit20 = np.mean(hit20) * 100
    mrr20 = np.mean(mrr20) * 100
    hit10 = np.mean(hit10) * 100
    mrr10 = np.mean(mrr10) * 100
    hit5 = np.mean(hit5) * 100
    mrr5 = np.mean(mrr5) * 100
    return hit20, mrr20, hit10, mrr10, hit5, mrr5
