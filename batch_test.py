import utils.metrics as metrics
import multiprocessing
import heapq

import numpy as np
import torch
from torch.autograd import Variable
import math
import heapq

#cores = multiprocessing.cpu_count() // 2
#config = Config()
#data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size)
#BATCH_SIZE = config.batch_size

def ranklist_by_heapq(group_pos_test, test_items, rating, Ks):
    item_score ={}
    for i in range(len(test_items)):
        item_score[test_items[i]] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in group_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc

def get_auc(item_score, group_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv:kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in group_pos_test:
            r.append(1)
        else:
            r.append(1)
    auc = metrics.auc(ground_truth=r, prediction=posterior)
    return auc

def ranklist_by_sorted(group_pos_test, test_items, rating, Ks):
    item_score = {}

    for i in range(len(test_items)):
        item_score[test_items[i]] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in group_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, group_pos_test)
    return r, auc

def get_performance(group_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, len(group_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K))
        hit_ratio.append(metrics.hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}


def test(model, test_groups, Ks, group_trainRatings, num_items, device):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}

    #pool = multiprocessing.Pool(cores)

    #g_batch_size = BATCH_SIZE
    model = model.cpu()
    temp_dev = model.device
    model.device = torch.device("cpu")
    n_test_groups = len(test_groups)
    #n_group_batchs = n_test_groups // g_batch_size + 1
    #print(n_group_batchs)
    #count = 0

    for g_id in test_groups:
        try:
            training_items = group_trainRatings[g_id]
        except Exception:
            training_items = []

        group_pos_test = test_groups[g_id]

        all_items = set(range(num_items))

        test_items = list(all_items-set(training_items))
        one_group_batch = np.full(len(test_items), g_id)
        group_var = torch.LongTensor(one_group_batch)
        item_var = torch.LongTensor(test_items)

        rating, _ = model(group_var, item_var)
        rating = rating.data.numpy()


        #if config.test_flag == 'part':
        #    r, auc = ranklist_by_heapq(group_pos_test, test_items, rating, Ks)
        #else:
        r, auc = ranklist_by_sorted(group_pos_test, test_items, rating, Ks)        
        re = get_performance(group_pos_test, r, auc, Ks)
        #for re in batch_result:
        result['precision'] += re['precision']
        result['recall'] += re['recall']
        result['ndcg'] += re['ndcg']
        result['hit_ratio'] += re['hit_ratio']
        result['auc'] += re['auc']
        #print(re['hit_ratio'])
    model.to(temp_dev)
    model.device = temp_dev
    #print(n_test_groups)
    #print(result['hit_ratio'])
    result['precision'] = result['precision']/n_test_groups
    result['recall'] = result['recall']/n_test_groups
    result['ndcg'] = result['ndcg']/n_test_groups
    result['hit_ratio'] = result['hit_ratio']/n_test_groups
    result['auc'] = result['auc']/n_test_groups
    #print(result['hit_ratio'])
    return result

