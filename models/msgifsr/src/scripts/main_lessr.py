import argparse
import sys

sys.path.append('..')
sys.path.append('../..')

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--dataset-dir', default='../../datasets/ml-1m', help='the dataset directory'
)
parser.add_argument('--embedding-dim', type=int, default=32, help='the embedding size')
parser.add_argument('--num-layers', type=int, default=3, help='the number of layers')
parser.add_argument(
    '--feat-drop', type=float, default=0.2, help='the dropout ratio for features'
)
parser.add_argument('--lr', type=float, default=1e-3, help='the learning rate')
parser.add_argument(
    '--batch-size', type=int, default=512, help='the batch size for training'
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
    default=2,
    help='the number of epochs that the performance does not improves after which the training stops',
)
parser.add_argument(
    '--num-workers',
    type=int,
    default=0,
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
args = parser.parse_args()
print(args)


from pathlib import Path

import torch as th
from torch.utils.data import DataLoader, SequentialSampler
from src.models import LESSR
from src.utils.data.collate import (collate_fn_factory, seq_to_eop_multigraph,
                                seq_to_shortcut_graph)
from src.utils.data.dataset import AugmentedDataset, read_dataset
from src.utils.train import TrainRunner

dataset_dir = Path(args.dataset_dir)
print('reading dataset')
train_sessions, test_sessions, num_items = read_dataset(dataset_dir)

if args.valid_split is not None:
    num_valid = int(len(train_sessions) * args.valid_split)
    test_sessions = train_sessions[-num_valid:]
    train_sessions = train_sessions[:-num_valid]

train_set = AugmentedDataset(train_sessions)
test_set = AugmentedDataset(test_sessions)

if args.num_layers > 1:
    collate_fn = collate_fn_factory(seq_to_eop_multigraph, seq_to_shortcut_graph)
else:
    collate_fn = collate_fn_factory(seq_to_eop_multigraph)

train_loader = DataLoader(
    train_set,
    batch_size=args.batch_size,
    # shuffle=True,
    # drop_last=True,
    num_workers=args.num_workers,
    collate_fn=collate_fn,
    pin_memory=True,
    sampler=SequentialSampler(train_set)
)

test_loader = DataLoader(
    test_set,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    collate_fn=collate_fn,
)

model = LESSR(num_items, args.embedding_dim, args.num_layers, feat_drop=args.feat_drop)
device = th.device('cuda' if th.cuda.is_available() else 'cpu')
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
print(f'{mrr * 100:.3f}%\t{hit * 100:.3f}%')
