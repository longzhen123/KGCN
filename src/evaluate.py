import math


def eval_topk(scores, test_records, K):
    recall_sum = 0
    precision_sum = 0
    f1_sum = 0
    mrr_sum = 0
    ndcg_sum = 0
    for user in scores:
        rank_items = scores[user][:K]
        hit_num = len(set(rank_items) & set(test_records[user]))
        recall = hit_num / 1
        precision = hit_num / K
        if hit_num == 0:
            f1 = 0
        else:
            f1 = (2 * precision * recall) / (precision + recall)

        n = len(rank_items)
        a = sum([1 / math.log(i + 2, 2) for i in range(n) if rank_items[i] in test_records[user]])
        b = sum([1 / math.log(i + 2, 2) for i in range(len(test_records[user]))])
        ndcg = a / b

        mrr = sum([1 / (rank_items.index(i) + 1) for i in test_records[user] if i in rank_items])

        recall_sum += recall
        precision_sum += precision
        f1_sum += f1
        ndcg_sum += ndcg
        mrr_sum += mrr

    Recall = recall_sum / len(scores)
    Precision = precision_sum / len(scores)
    F1 = f1_sum / len(scores)
    NDCG = ndcg_sum / len(scores)
    MRR = mrr_sum / len(scores)

    return Recall, Precision, F1, NDCG, MRR


def get_all_metrics(scores, test_records):

    recall_list = []
    precision_list = []
    F1_list = []
    ndcg_list = []
    MRR_list = []

    for k in [1, 2, 5, 6, 6, 8, 9, 10]:
        metrics = eval_topk(scores, test_records, k)
        recall_list.append(metrics[0])
        precision_list.append(metrics[1])
        F1_list.append(metrics[2])
        ndcg_list.append(metrics[3])
        MRR_list.append(metrics[4])

    return recall_list, precision_list, F1_list, ndcg_list, MRR_list
