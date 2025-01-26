import time
import argparse
import pickle
from model import *
from utils import *
import logging


def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ml-1m', help='diginetica/Nowplaying/Tmall')
parser.add_argument('--hiddenSize', type=int, default=100)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--activate', type=str, default='relu')
parser.add_argument('--n_sample_all', type=int, default=12)
parser.add_argument('--n_sample', type=int, default=12)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay.')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay.')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty ')
parser.add_argument('--n_iter', type=int, default=1)                                    # [1, 2]
parser.add_argument('--dropout_gcn', type=float, default=0, help='Dropout rate.')       # [0, 0.2, 0.4, 0.6, 0.8]
parser.add_argument('--dropout_local', type=float, default=0, help='Dropout rate.')     # [0, 0.5]
parser.add_argument('--dropout_global', type=float, default=0.5, help='Dropout rate.')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=3)

opt = parser.parse_args()

torch.cuda.set_device(2)

logging.basicConfig(filename='evaluation_results.log',  # log name
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s') 


def main():
    init_seed(2024)

    if opt.dataset == 'diginetica':
        num_node = 43098
        opt.n_iter = 2
        opt.dropout_gcn = 0.2
        opt.dropout_local = 0.0
    elif opt.dataset == 'Nowplaying':
        num_node = 60417
        opt.n_iter = 1
        opt.dropout_gcn = 0.0
        opt.dropout_local = 0.0
    elif opt.dataset == 'Tmall':
        num_node = 40728
        opt.n_iter = 1
        opt.dropout_gcn = 0.6
        opt.dropout_local = 0.5
    elif opt.dataset == 'beauty2014':
        num_node = 12102
        opt.n_iter = 1
        opt.dropout_gcn = 0
        opt.dropout_local = 0
    elif opt.dataset == 'ml-1m':
        num_node = 3417
        opt.n_iter = 1
        opt.dropout_gcn = 0
        opt.dropout_local = 0
    else:
        num_node = 310

    train_data = pickle.load(open('datasets/' + opt.dataset + '/train_ML.txt', 'rb'))
    train_response_explicit = pickle.load(open('../LLM/data/Ml-1M/train_response_explicit.txt', 'rb'))
    train_response_latent = pickle.load(open('../LLM/data/Ml-1M/train_response_latent.txt', 'rb'))
    train_data.append([item.detach().cpu().numpy() for item in train_response_explicit])
    train_data.append([item.detach().cpu().numpy() for item in train_response_latent])

    test_response_explicit = pickle.load(open('../LLM/data/Ml-1M/test_response_explicit.txt', 'rb'))
    test_response_latent = pickle.load(open('../LLM/data/Ml-1M/test_response_latent.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('datasets/' + opt.dataset + '/test_ML.txt', 'rb'))
        test_data.append([item.detach().cpu().numpy() for item in test_response_explicit])
        test_data.append([item.detach().cpu().numpy() for item in test_response_latent])

    adj = pickle.load(open('datasets/' + opt.dataset + '/adj_' + str(opt.n_sample_all) + '.pkl', 'rb'))
    num = pickle.load(open('datasets/' + opt.dataset + '/num_' + str(opt.n_sample_all) + '.pkl', 'rb'))
    train_data = Data(train_data)
    test_data = Data(test_data)

    adj, num = handle_adj(adj, num_node, opt.n_sample_all, num)
    model = trans_to_cuda(CombineGraph(opt, num_node, adj, num))

    # model.load_state_dict(torch.load('best_model_ml.pth'))

    print(opt)
    start = time.time()
    best_result = [0, 0, 0, 0, 0, 0]
    best_epoch = [0, 0, 0, 0, 0, 0]
    bad_counter = 0

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit_5, mrr_5, hit_10, mrr_10, hit_20, mrr_20 = train_test(model, train_data, test_data)
        flag = 0
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
    # torch.save(best_model_wts, 'best_model_ml.pth')


if __name__ == '__main__':
    main()
