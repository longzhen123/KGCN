import numpy as np
import pandas as pd


def load_kg(data_dir):
    # print(rel_dict)
    # print('load_kg...')
    edges = pd.read_csv(data_dir + 'kg.txt', delimiter='\t', header=None).values
    kg_dict = {}
    relation_set = set()
    entity_set = set()
    for edge in edges:
        head = edge[0]
        tail = edge[1]
        relation = edge[2]

        if head not in kg_dict:
            kg_dict[head] = []

        kg_dict[head].append([relation, tail])

        entity_set.add(head)
        entity_set.add(tail)
        relation_set.add(relation)

    n_entity = len(entity_set)

    return kg_dict, n_entity, len(relation_set)


def data_split(ratings_np, ratio):
    # print('data split...')

    positive_records, negative_records = convert_dict(ratings_np)
    train_set = []
    eval_set = []
    test_set = []

    for user in positive_records:
        pos_record = positive_records[user]
        neg_record = negative_records[user]
        size = len(pos_record)

        test_indices = np.random.choice(size, int(size * 0.2), replace=False)
        rem_indices = list(set(range(size)) - set(test_indices))
        eval_indices = np.random.choice(rem_indices, int(size * 0.2), replace=False)
        train_indices = list(set(rem_indices) - set(eval_indices))

        if ratio < 1:
            size = int(len(train_indices) * ratio)
            if size < 1:
                size = 1
            train_indices = np.random.choice(train_indices, size, replace=False)

        train_set.extend([user, pos_record[i], 1] for i in train_indices)
        train_set.extend([user, neg_record[i], 0] for i in train_indices)

        eval_set.extend([user, pos_record[i], 1] for i in eval_indices)
        eval_set.extend([user, neg_record[i], 0] for i in eval_indices)

        test_set.extend([user, pos_record[i], 1] for i in test_indices)
        test_set.extend([user, neg_record[i], 0] for i in test_indices)

    return train_set, eval_set, test_set


def get_rec(train_records, eval_records,  test_records, item_set):

    rec = dict()
    users = list(test_records.keys())
    for user in users:
        neg_items = list(item_set - set(train_records[user]) - set(eval_records[user]) - set(test_records[user]))
        rec[user] = np.random.choice(neg_items, 100).tolist() + np.random.choice(test_records[user], 1).tolist()
    return rec


def load_ratings(data_dir):

    data_np = pd.read_csv(data_dir + 'ratings.txt', delimiter='\t', header=None).values

    return data_np


def convert_dict(ratings_np):

    positive_records = dict()
    negative_records = dict()

    for pair in ratings_np:
        user = pair[0]
        item = pair[1]
        label = pair[2]

        if label == 1:
            if user not in positive_records:
                positive_records[user] = []
            positive_records[user].append(item)
        else:
            if user not in negative_records:
                negative_records[user] = []
            negative_records[user].append(item)

    return positive_records, negative_records


def get_records(data_set):

    records = dict()

    for pair in data_set:
        user = pair[0]
        item = pair[1]
        label = pair[2]

        if label == 1:
            if user not in records:
                records[user] = []

            records[user].append(item)

    return records


def load_data(args):
    data_dir = './data/' + args.dataset + '/'
    ratings_np = load_ratings(data_dir)
    train_set, eval_set, test_set = data_split(ratings_np, args.ratio)
    item_set = set(ratings_np[:, 1])
    user_set = set(ratings_np[:, 0])
    train_records = get_records(train_set)
    eval_records = get_records(eval_set)
    test_records = get_records(test_set)
    rec = get_rec(train_records, eval_records, test_records, item_set)
    kg_dict, n_entity, n_relation = load_kg(data_dir)
    n_entity = n_entity
    n_user = len(user_set)
    n_item = len(item_set)
    data = [n_entity, n_user, n_item, n_relation, train_set, eval_set, test_set, rec, kg_dict]

    return data
