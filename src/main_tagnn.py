import argparse
import pickle
import time
from utils import build_graph, Data, split_validation
from model import *
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='beauty2014', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
opt = parser.parse_args()
torch.cuda.set_device(5)
print(opt)

logging.basicConfig(filename='evaluation_results.log',  # log name
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    init_seed(2024)
    train_data = pickle.load(open('datasets/' + opt.dataset + '/train_beauty.txt', 'rb'))

    train_response_explicit = pickle.load(open('/workspace/LLM-GNN/LLM/data/beauty2014/train_response_explicit.txt', 'rb'))
    train_response_latent = pickle.load(open('/workspace/LLM-GNN/LLM/data/beauty2014/train_response_latent.txt', 'rb'))
    train_data.append([item.detach().cpu().numpy() for item in train_response_explicit])
    train_data.append([item.detach().cpu().numpy() for item in train_response_latent])

    test_response_explicit = pickle.load(open('/workspace/LLM-GNN/LLM/data/beauty2014/test_response_explicit.txt', 'rb'))
    test_response_latent = pickle.load(open('/workspace/LLM-GNN/LLM/data/beauty2014/test_response_latent.txt', 'rb'))

    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('datasets/' + opt.dataset + '/test_beauty.txt', 'rb'))
        test_data.append([item.detach().cpu().numpy() for item in test_response_explicit])
        test_data.append([item.detach().cpu().numpy() for item in test_response_latent])
    # all_train_seq = pickle.load(open('../datasets/' + opt.dataset + '/all_train_seq.txt', 'rb'))
    # g = build_graph(all_train_seq)
    train_data = Data(train_data, shuffle=True)
    test_data = Data(test_data, shuffle=False)
    # del all_train_seq, g
    if opt.dataset == 'diginetica':
        n_node = 43098
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        n_node = 37484
    elif opt.dataset == 'beauty2014':
        n_node = 12102
    elif opt.dataset == 'ml-1m':
        n_node = 3417
    else:
        n_node = 310

    model = trans_to_cuda(SessionGraph(opt, n_node))

    start = time.time()
    best_result = [0, 0, 0, 0, 0, 0]
    best_epoch = [0, 0, 0, 0, 0, 0]
    bad_counter = 0
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit_5, mrr_5, hit_10, mrr_10, hit_20, mrr_20 = train_test(model, train_data, test_data)
        flag = 0
        # if hit >= best_result[0]:
        #     best_result[0] = hit
        #     best_epoch[0] = epoch
        #     flag = 1
        # if mrr >= best_result[1]:
        #     best_result[1] = mrr
        #     best_epoch[1] = epoch
        #     flag = 1
        if hit_5 >= best_result[0]:
            best_result[0] = hit_5
            best_epoch[0] = epoch
            flag = 1
        if mrr_5 >= best_result[1]:
            best_result[1] = mrr_5
            best_epoch[1] = epoch
            flag = 1
        if hit_10 >= best_result[2]:
            best_result[2] = hit_10
            best_epoch[2] = epoch
            flag = 1
        if mrr_10 >= best_result[3]:
            best_result[3] = mrr_10
            best_epoch[3] = epoch
            flag = 1
        if hit_20 >= best_result[4]:
            # best_model_wts = model.state_dict()
            best_result[4] = hit_20
            best_epoch[4] = epoch
            flag = 1
        if mrr_20 >= best_result[5]:
            best_result[5] = mrr_20
            best_epoch[5] = epoch
            flag = 1
        print('Current Result:')
        print('\tRecall@5:\t%.4f\tMMR@5:\t%.4f\tRecall@10:\t%.4f\tMMR@10:\t%.4f\tRecall@20:\t%.4f\tMMR@20:\t%.4f' % (hit_5, mrr_5, hit_10, mrr_10, hit_20, mrr_20))
        print('Best Result:')
        print('\tRecall@5:\t%.4f\tMMR@5:\t%.4f\tRecall@10:\t%.4f\tMMR@10:\t%.4f\tRecall@20:\t%.4f\tMMR@20:\t%.4f' % (best_result[0], best_result[1], best_result[2], best_result[3], best_result[4], best_result[5]))
        print('Epoch:')
        print('\t%d,\t%d,\t%d,\t%d,\t%d,\t%d' % (best_epoch[0], best_epoch[1], best_epoch[2], best_epoch[3], best_epoch[4], best_epoch[5]))
        logging.info('Recall@5:\t%.4f\tMMR@5:\t%.4f\tRecall@10:\t%.4f\tMMR@10:\t%.4f\tRecall@20:\t%.4f\tMMR@20:\t%.4f' % 
             (best_result[0], best_result[1], best_result[2], best_result[3], best_result[4], best_result[5]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
