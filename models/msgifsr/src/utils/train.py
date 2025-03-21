import time

import numpy as np
import torch as th
from sklearn.metrics import accuracy_score
from torch import nn, optim
from tqdm import tqdm
import wandb


# ignore weight decay for parameters in bias, batch norm and activation
def fix_weight_decay(model):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(map(lambda x: x in name, ['bias', 'batch_norm', 'activation'])):
            no_decay.append(param)
        else:
            decay.append(param)
    params = [{'params': decay}, {'params': no_decay, 'weight_decay': 0}]
    return params


def prepare_batch(batch, device):
    inputs, labels, explicit, latent = batch
    # inputs, labels = batch
    inputs_gpu  = [x.to(device) for x in inputs]
    labels_gpu  = labels.to(device)
    explicit_gpu = explicit.to(device)
    latent_gpu = latent.to(device)
   
    return inputs_gpu, labels_gpu, explicit_gpu, latent_gpu
    # return inputs_gpu, 0, labels_gpu, 0


def evaluate(model, data_loader, device, cutoff=20):
    model.eval()
    mrr = 0
    hit = 0
    # mrrs = []
    # hits = []
    num_samples = 0

    with th.no_grad():
        for batch in data_loader:
            inputs, labels, explicit, latent = prepare_batch(batch, device)
            logits, con_loss = model(*inputs, explicit, latent)
        
            batch_size   = logits.size(0)
            num_samples += batch_size
            topk         = logits.topk(k=cutoff)[1]
            labels       = labels.unsqueeze(-1)
            hit_ranks    = th.where(topk == labels)[1] + 1
            hit          += hit_ranks.numel()
            # print("topk: ", topk, "labels: ", labels, "hit_ranks: ", hit_ranks, "hit: ", hit)
            mrr         += hit_ranks.float().reciprocal().sum().item()
            # hit += (hit_ranks != 0).sum().item()  # 累计每个批次是否至少有一个命中
            # mrr = (hit_ranks.numel() > 0) * hit_ranks.float().reciprocal().sum().item()  # 只对命中样本计算MRR
            # hits.append(hit)
            # mrrs.append(mrr)
            
    return mrr / num_samples, hit / num_samples
    # return np.mean(mrrs), np.mean(hits)
class TrainRunner:
    def __init__(
        self,
        dataset,
        model,
        train_loader,
        test_loader,
        device,
        lr=1e-3,
        weight_decay=0,
        patience=3,
    ):
        self.dataset = dataset
        self.model = model
        if weight_decay > 0:
            params = fix_weight_decay(model)
        else:
            params = model.parameters()
        self.optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)
        
        self.train_loader = train_loader
        self.test_loader  = test_loader
        self.device       = device
        self.epoch        = 0
        self.batch        = 0
        self.patience     = patience

    def train(self, epochs, log_interval=100):
        max_mrr = 0
        max_hit = 0
        bad_counter = 0
        t = time.time()
        mean_loss = 0

        mrr, hit = evaluate(self.model, self.test_loader, self.device)
        for epoch in tqdm(range(epochs)):
            self.model.train()
            for batch in self.train_loader:
                inputs, labels, explicit, latent = prepare_batch(batch, self.device)
                self.optimizer.zero_grad()
                scores, con_loss = self.model(*inputs, explicit, latent)
                assert not th.isnan(scores).any()
                loss   = nn.functional.nll_loss(scores, labels) + con_loss
                loss.backward()
                self.optimizer.step()
                
                mean_loss += loss.item() / log_interval
                
                if self.batch > 0 and self.batch % log_interval == 0:
                    print(f'Batch {self.batch}: Loss = {mean_loss:.4f}, Time Elapsed = {time.time() - t:.2f}s')
                    t = time.time()
                    mean_loss = 0
                    
                self.batch += 1
            self.scheduler.step()
            mrr, hit = evaluate(self.model, self.test_loader, self.device)
            
            # wandb.log({"hit": hit, "mrr": mrr})

            print(f'Epoch {self.epoch}: MRR = {mrr * 100:.4f}%, Hit = {hit * 100:.4f}%')

            if mrr < max_mrr and hit < max_hit:
                bad_counter += 1
                if bad_counter == self.patience:
                    break
            else:
                bad_counter = 0
            max_mrr = max(max_mrr, mrr)
            max_hit = max(max_hit, hit)
            self.epoch += 1
        return max_mrr, max_hit
