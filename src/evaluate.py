import numpy as np


def get_hit(gt_item, pred_items):

    if gt_item in pred_items:
        return 1
    else:
        return 0


def get_ndcg(gt_item, pred_items):

    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index + 2))
    else:
        return 0
