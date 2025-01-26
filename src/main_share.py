import argparse
import pickle
import time
from utils import Data, split_validation, translation
from model import *
import logging

def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ml-1m', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--worker', type=int, default=3, help='number of worker')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=0.0, help='l2 penalty')  #
parser.add_argument('--step', type=int, default=2, help='propogation steps')
parser.add_argument('--window', type=int, default=1, help='window size')
parser.add_argument('--patience', type=int, default=3, help='the number of epoch to wait before early stop')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
torch.cuda.set_device(5)
opt = parser.parse_args()
print(opt)

logging.basicConfig(filename='evaluation_results2.log',  # log name
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    init_seed(2024)
    train_data = pickle.load(open('datasets/' + opt.dataset + '/train_ML.txt', 'rb'))
    train_response_explicit = pickle.load(open('/workspace/LLM-GNN/LLM/data/Ml-1M/train_response_explicit.txt', 'rb'))
    train_response_latent = pickle.load(open('/workspace/LLM-GNN/LLM/data/Ml-1M/train_response_latent.txt', 'rb'))
    train_data.append([item.detach().cpu().numpy() for item in train_response_explicit])
    train_data.append([item.detach().cpu().numpy() for item in train_response_latent])

    test_response_explicit = pickle.load(open('/workspace/LLM-GNN/LLM/data/Ml-1M/test_response_explicit.txt', 'rb'))
    test_response_latent = pickle.load(open('/workspace/LLM-GNN/LLM/data/Ml-1M/test_response_latent.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('datasets/' + opt.dataset + '/test_ML.txt', 'rb'))
        test_data.append([item.detach().cpu().numpy() for item in test_response_explicit])
        test_data.append([item.detach().cpu().numpy() for item in test_response_latent])

    llen = [len(train_data[0][i]) for i in range(len(train_data[0]))] + [len(test_data[0][i]) for i in range(len(test_data[0]))]
    print(max(llen),sum(llen)*1.0/len(llen))
    if not opt.dataset == 'diginetica':
        l = []
        for i in range(len(train_data[0])):
            l += list(train_data[0][i])
        l += list(train_data[1])

        for i in range(len(test_data[0])):
            l += list(test_data[0][i])
        l += list(test_data[1])
        l = set(l)
        print('total number of items', len(l))

        item_dic = {}
        for i in l:
           item_dic[i] = len(item_dic) + 1 #start from 1

        del l
        train_data = translation(train_data, item_dic)
        test_data = translation(test_data, item_dic)
        
        n_node = len(item_dic) + 1

    

    train_data = Data(train_data, opt.window)
    test_data = Data(test_data, opt.window)


    if opt.dataset == 'diginetica':
        n_node = 43098
    


    model = trans_to_cuda(SessionGraph(opt, n_node))

    start = time.time()
    best_result = [0, 0, 0, 0, 0, 0]
    best_epoch = [0, 0, 0, 0, 0, 0]
    bad_counter = 0




    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)

        train_model(model, train_data, opt)

        hit, mrr, hit10, mrr10, hit5, mrr5 = test_model(model, test_data, opt)


        flag = 0
        # if hit >= best_result[0]:
        #     best_result[0] = hit
        #     best_result[2] = hit10
        #     best_result[4] = hit5
        #     best_epoch[0] = epoch
        #     flag = 1
        # if mrr >= best_result[1]:
        #     best_result[1] = mrr
        #     best_result[3] = mrr10
        #     best_result[5] = mrr5
        #     best_epoch[1] = epoch
        #     flag = 1
        
        if hit5 >= best_result[0]:
            best_result[0] = hit5
            best_epoch[0] = epoch
            flag = 1
        if mrr5 >= best_result[1]:
            best_result[1] = mrr5
            best_epoch[1] = epoch
            flag = 1
        if hit10 >= best_result[2]:
            best_result[2] = hit10
            best_epoch[2] = epoch
            flag = 1
        if mrr10 >= best_result[3]:
            best_result[3] = mrr10
            best_epoch[3] = epoch
            flag = 1
        if hit >= best_result[4]:
            # best_model_wts = model.state_dict()
            best_result[4] = hit
            best_epoch[4] = epoch
            flag = 1
        if mrr >= best_result[5]:
            best_result[5] = mrr
            best_epoch[5] = epoch
            flag = 1

        print('Current Result:')
        print('\tRecall@5:\t%.4f\tMMR@5:\t%.4f\tRecall@10:\t%.4f\tMMR@10:\t%.4f\tRecall@20:\t%.4f\tMMR@20:\t%.4f' % (hit5, mrr5, hit10, mrr10, hit, mrr))
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
