import argparse
import os
import numpy as np
import torch
import random
import sys
import pickle

sys.path.append('..')
sys.path.append('../..')

def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    
seed_torch(123)

# def get_freer_gpu():
#     os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
#     memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
#     # memory_available = memory_available[0:5]
#     if len(memory_available) == 0:
#         return -1
#     return int(np.argmax(memory_available))

# os.environ["CUDA_VISIBLE_DEVICES"] = str(5)
torch.cuda.set_device(5)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--dataset-dir', default='/datasets/ml-1m', help='the dataset directory'
)
parser.add_argument('--embedding-dim', type=int, default=256, help='the embedding size') #256
parser.add_argument('--num-layers', type=int, default=1, help='the number of layers')
parser.add_argument(
    '--feat-drop', type=float, default=0.1, help='the dropout ratio for features'
)
parser.add_argument('--lr', type=float, default=1e-3, help='the learning rate')
parser.add_argument(
    '--batch-size', type=int, default=512, help='the batch size for training' #512
)
parser.add_argument(
    '--epochs', type=int, default=30, help='the number of training epochs'
)
parser.add_argument(
    '--weight-decay',
    type=float,
    default=1e-4,
    help='the parameter for L2 regularization',
)
parser.add_argument(
    '--patience',
    type=int,
    default=3,
    help='the number of epochs that the performance does not improves after which the training stops',
)
parser.add_argument(
    '--num-workers',
    type=int,
    default=4,
    help='the number of processes to load the input graphs',
)
parser.add_argument(
    '--valid-split',
    type=float,
    default=None,
    help='the fraction for the validation set',
)
parser.add_argument(
    '--log-interval',
    type=int,
    default=100,
    help='print the loss after this number of iterations',
)
parser.add_argument(
    '--order',
    type=int,
    default=3,
    help='order of msg',
)
parser.add_argument(
    '--reducer',
    type=str,
    default='mean',
    help='method for reducer',
)
parser.add_argument(
    '--norm',
    type=bool,
    default=True,
    help='whether use l2 norm',
)

parser.add_argument(
    '--extra',
    action='store_true',
    help='whether use REnorm.',
)

parser.add_argument(
    '--fusion',
    action='store_true',
    help='whether use IFR.',
)

args = parser.parse_args()
print(args)


from pathlib import Path
import os
import numpy as np
import torch as th
from torch.utils.data import DataLoader, SequentialSampler
from src.utils.data.dataset import read_dataset, AugmentedDataset
from src.utils.data.collate import (
    seq_to_ccs_graph,
    collate_fn_factory_ccs
)
from src.utils.train import TrainRunner
from src.models import MSGIFSR


device = th.device('cuda:5' if th.cuda.is_available() else 'cpu')
dataset_dir = Path(args.dataset_dir)
print('reading dataset')
train_sessions, test_sessions, num_items = read_dataset(dataset_dir)
# num_items += 5

if args.valid_split is not None:
    num_valid      = int(len(train_sessions) * args.valid_split)
    test_sessions  = train_sessions[-num_valid:]
    train_sessions = train_sessions[:-num_valid]

train_datas = []
train_datas.append(train_sessions)
train_response_explicit = pickle.load(open('/workspace/LLM-GNN/LLM/data/Ml-1M/train_response_explicit.txt', 'rb'))
train_response_latent = pickle.load(open('/workspace/LLM-GNN/LLM/data/Ml-1M/train_response_latent.txt', 'rb'))
train_datas.append([item.detach().cpu().numpy() for item in train_response_explicit])
train_datas.append([item.detach().cpu().numpy() for item in train_response_latent])

test_datas = []
test_datas.append(test_sessions)
test_response_explicit = pickle.load(open('/workspace/LLM-GNN/LLM/data/Ml-1M/test_response_explicit.txt', 'rb'))
test_response_latent = pickle.load(open('/workspace/LLM-GNN/LLM/data/Ml-1M/test_response_latent.txt', 'rb'))
test_datas.append([item.detach().cpu().numpy() for item in test_response_explicit])
test_datas.append([item.detach().cpu().numpy() for item in test_response_latent])

train_set = AugmentedDataset(train_datas)
test_set  = AugmentedDataset(test_datas)
print(len(train_set))
print(len(test_set))


collate_fn = collate_fn_factory_ccs((seq_to_ccs_graph,), order=args.order)

train_loader = DataLoader(
    train_set,
    batch_size=args.batch_size,
    shuffle=True,
    # drop_last=True,
    num_workers=args.num_workers,
    collate_fn=collate_fn,
    pin_memory=True,
    # sampler=SequentialSampler(train_set)
)

test_loader = DataLoader(
    test_set,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    collate_fn=collate_fn,
    pin_memory=True
)

model = MSGIFSR(num_items, args.dataset_dir, args.embedding_dim, args.num_layers, dropout=args.feat_drop, reducer=args.reducer, order=args.order, norm=args.norm, extra=args.extra, fusion=args.fusion, device=device)

model = model.to(device)

print(model)

runner = TrainRunner(
    args.dataset_dir,
    model,
    train_loader,
    test_loader,
    device=device,
    lr=args.lr,
    weight_decay=args.weight_decay,
    patience=args.patience,
)

print('start training')
mrr, hit = runner.train(args.epochs, args.log_interval)
print('MRR@20\tHR@20')
print(f'{mrr * 100:.4f}%\t{hit * 100:.4f}%')
