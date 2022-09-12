import time
import numpy as np
import torch as t
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score

from src.evaluate import get_all_metrics
from src.load_base import load_data, get_records


class KGCN(nn.Module):

    def __init__(self, n_entity, n_user, n_relation, dim, n_iter, n_neighbors):
        super(KGCN, self).__init__()
        entity_emb_matrix = t.randn(n_entity, dim)
        relation_emb_matrix = t.randn(n_relation, dim)
        user_emb_matrix = t.randn(n_user, dim)
        nn.init.xavier_uniform_(entity_emb_matrix)
        nn.init.xavier_uniform_(relation_emb_matrix)
        nn.init.xavier_uniform_(user_emb_matrix)
        self.entity_emb_matrix = nn.Parameter(entity_emb_matrix)
        self.relation_emb_matrix = nn.Parameter(relation_emb_matrix)
        self.user_emb_matrix = nn.Parameter(user_emb_matrix)
        self.W_sum = nn.Linear(dim, dim)
        self.dim = dim
        self.n_iter = n_iter
        self.n_neighbors = n_neighbors

    def forward(self, pairs, adj_entity_np, adj_relation_np):

        users = [pair[0] for pair in pairs]
        items = [pair[1] for pair in pairs]
        entities, relations = self.get_neighbors(items, adj_entity_np, adj_relation_np)

        user_embeddings = self.user_emb_matrix[users]
        entity_vectors = [self.entity_emb_matrix[entities[i]].reshape(len(pairs), -1, self.dim) for i in range(len(entities))]
        relation_vectors = [self.relation_emb_matrix[relations[i]].reshape(len(pairs), -1, self.dim) for i in range(len(relations))]

        for i in range(self.n_iter):

            entity_vectors = self.gcn_layer(user_embeddings, entity_vectors, relation_vectors, i)

        item_embeddings = entity_vectors[0].reshape(-1, self.dim)

        predict = (user_embeddings * item_embeddings).sum(dim=1)

        return t.sigmoid(predict)

    def gcn_layer(self, user_embeddings, entity_vectors, relation_vectors, l):

        n = user_embeddings.shape[0]
        next_entity_vectors = []
        shape = [n, -1, self.n_neighbors, self.dim]
        for i in range(self.n_iter - l):

            relation_scores = (user_embeddings.view(n, 1, 1, self.dim) * relation_vectors[i].view(shape)).sum(dim=-1)
            relation_scores = relation_scores.reshape(n, -1, self.n_neighbors, 1)
            normalize_relation_scores = t.softmax(relation_scores, dim=-2)
            neighbor_embeddings = normalize_relation_scores * entity_vectors[i + 1].reshape(shape)
            neighbor_embeddings = neighbor_embeddings.sum(dim=-2)
            entity_and_neighbors = entity_vectors[i].view(n, -1, self.dim) + neighbor_embeddings
            if l == self.n_iter-1:
                next_entity_vectors.append(t.tanh(self.W_sum(entity_and_neighbors)))
            else:
                next_entity_vectors.append(t.relu(self.W_sum(entity_and_neighbors)))

        return next_entity_vectors

    def get_neighbors(self, seeds, adj_entity_np, adj_relation_np):

        entities = [seeds]
        relations = []
        for i in range(self.n_iter):
            entities.append(adj_entity_np[entities[i]].reshape(-1))
            relations.append(adj_relation_np[entities[i]].reshape(-1))

        return entities, relations


def get_scores(model, rec, adj_entity_np, adj_relation_np):
    scores = {}
    model.eval()
    for user in (rec):
        items = list(rec[user])
        pairs = [[user, item] for item in items]
        predict = model.forward(pairs, adj_entity_np, adj_relation_np).cpu().view(-1).detach().numpy().tolist()
        n = len(pairs)
        user_scores = {items[i]: predict[i] for i in range(n)}
        user_list = list(dict(sorted(user_scores.items(), key=lambda x: x[1], reverse=True)).keys())
        scores[user] = user_list
    model.train()
    return scores


def eval_ctr(model, pairs, adj_entity_np, adj_relation_np, batch_size):

    model.eval()
    pred_label = []
    for i in range(0, len(pairs), batch_size):
        batch_label = model(pairs[i: i+batch_size], adj_entity_np, adj_relation_np).cpu().detach().numpy().tolist()
        pred_label.extend(batch_label)
    model.train()

    true_label = [pair[2] for pair in pairs]
    auc = roc_auc_score(true_label, pred_label)

    pred_np = np.array(pred_label)
    pred_np[pred_np >= 0.5] = 1
    pred_np[pred_np < 0.5] = 0
    pred_label = pred_np.tolist()
    acc = accuracy_score(true_label, pred_label)
    return round(auc, 3), round(acc, 3)


def construct_adj(kg_dict, n_neighbors, n_entity):

    adj_entity_np = np.zeros([n_entity, n_neighbors], dtype=np.int)
    adj_relation_np = np.zeros([n_entity, n_neighbors], dtype=np.int)
    # print(adj_entity_np.dtype)
    for head in kg_dict:
        neighbors = kg_dict[head]

        replace = len(neighbors) < n_neighbors
        indices = np.random.choice(len(neighbors), n_neighbors, replace=replace)

        adj_relation_np[head] = np.array([int(neighbors[i][0]) for i in indices])
        adj_entity_np[head] = np.array([int(neighbors[i][1]) for i in indices])

    return adj_entity_np, adj_relation_np


def train(args, is_topk=False):
    np.random.seed(555)

    data = load_data(args)
    n_entity, n_user, n_item, n_relation = data[0], data[1], data[2], data[3]
    train_set, eval_set, test_set, rec, kg_dict = data[4], data[5], data[6], data[7], data[8]
    test_records = get_records(test_set)

    adj_entity_np, adj_relation_np = construct_adj(kg_dict, args.n_neighbors, n_entity)
    criterion = nn.BCELoss()
    model = KGCN(n_entity, n_user, n_relation, args.dim, args.n_iter, args.n_neighbors)
    if t.cuda.is_available():
        model = model.to(args.device)

    optimizer = t.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    print(args.dataset + '-----------------------------------------')

    print('dim: %d' % args.dim, end='\t')
    print('n_iter: %d' % args.n_iter, end='\t')
    print('n_neighbors: %d' % args.n_neighbors, end='\t')
    print('lr: %1.0e' % args.lr, end='\t')
    print('l2: %1.0e' % args.l2, end='\t')
    print('batch_size: %d' % args.batch_size)
    train_auc_list = []
    train_acc_list = []
    eval_auc_list = []
    eval_acc_list = []
    test_auc_list = []
    test_acc_list = []
    all_precision_list = []

    for epoch in (range(args.epochs)):

        start = time.clock()
        loss_sum = 0
        np.random.shuffle(train_set)
        for i in range(0, len(train_set), args.batch_size):

            if (i + args.batch_size + 1) > len(train_set):
                batch_uvls = train_set[i:]
            else:
                batch_uvls = train_set[i: i + args.batch_size]

            pairs = [[uvl[0], uvl[1]] for uvl in batch_uvls]
            labels = t.tensor([int(uvl[2]) for uvl in batch_uvls]).view(-1).float()
            if t.cuda.is_available():
                labels = labels.to(args.device)

            predicts = model(pairs, adj_entity_np, adj_relation_np)

            loss = criterion(predicts, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
        train_auc, train_acc = eval_ctr(model, train_set, adj_entity_np, adj_relation_np, args.batch_size)
        eval_auc, eval_acc = eval_ctr(model, eval_set,  adj_entity_np, adj_relation_np, args.batch_size)
        test_auc, test_acc = eval_ctr(model, test_set, adj_entity_np, adj_relation_np, args.batch_size)

        print('epoch: %d \t train_auc: %.3f \t train_acc: %.3f \t '
              'eval_auc: %.3f \t eval_acc: %.3f \t test_auc: %.3f \t test_acc: %.3f \t' %
              ((epoch + 1), train_auc, train_acc, eval_auc, eval_acc, test_auc, test_acc), end='\t')

        precision_list = []
        if is_topk:
            scores = get_scores(model, rec, adj_entity_np, adj_relation_np)
            precision_list = get_all_metrics(scores, test_records)[0]
            print(precision_list, end='\t')

        train_auc_list.append(train_auc)
        train_acc_list.append(train_acc)
        eval_auc_list.append(eval_auc)
        eval_acc_list.append(eval_acc)
        test_auc_list.append(test_auc)
        test_acc_list.append(test_acc)
        all_precision_list.append(precision_list)
        end = time.clock()
        print('time: %d' % (end - start))

    indices = eval_auc_list.index(max(eval_auc_list))
    print(args.dataset, end='\t')
    print('train_auc: %.3f \t train_acc: %.3f \t eval_auc: %.3f \t eval_acc: %.3f \t '
          'test_auc: %.3f \t test_acc: %.3f \t' %
          (train_auc_list[indices], train_acc_list[indices], eval_auc_list[indices], eval_acc_list[indices],
           test_auc_list[indices], test_acc_list[indices]), end='\t')

    print(all_precision_list[indices])

    return eval_auc_list[indices], eval_acc_list[indices], test_auc_list[indices], test_acc_list[indices]




