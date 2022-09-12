from src.KGCN import train
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # parser.add_argument('--dataset', type=str, default='music', help='dataset')
    # parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
    # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    # parser.add_argument('--epochs', type=int, default=20, help='epochs')
    # parser.add_argument("--device", type=str, default='cuda:0', help='device')
    # parser.add_argument('--dim', type=int, default=20, help='embedding size')
    # parser.add_argument('--n_neighbors', type=int, default=40, help='the number of neighbors')
    # parser.add_argument('--n_iter', type=int, default=1, help='the number of layers of KGCN')
    # parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')

    # parser.add_argument('--dataset', type=str, default='book', help='dataset')
    # parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
    # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    # parser.add_argument('--epochs', type=int, default=20, help='epochs')
    # parser.add_argument("--device", type=str, default='cuda:0', help='device')
    # parser.add_argument('--dim', type=int, default=50, help='embedding size')
    # parser.add_argument('--n_neighbors', type=int, default=5, help='the number of neighbors')
    # parser.add_argument('--n_iter', type=int, default=2, help='the number of layers of KGCN')
    # parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')

    # parser.add_argument('--dataset', type=str, default='ml', help='dataset')
    # parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
    # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    # parser.add_argument('--epochs', type=int, default=20, help='epochs')
    # parser.add_argument("--device", type=str, default='cuda:0', help='device')
    # parser.add_argument('--dim', type=int, default=40, help='embedding size')
    # parser.add_argument('--n_neighbors', type=int, default=20, help='the number of neighbors')
    # parser.add_argument('--n_iter', type=int, default=1, help='the number of layers of KGCN')
    # parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')
    #
    parser.add_argument('--dataset', type=str, default='yelp', help='dataset')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--l2', type=float, default=1e-4, help='L2')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--epochs', type=int, default=20, help='epochs')
    parser.add_argument("--device", type=str, default='cuda:0', help='device')
    parser.add_argument('--dim', type=int, default=30, help='embedding size')
    parser.add_argument('--n_neighbors', type=int, default=40, help='the number of neighbors')
    parser.add_argument('--n_iter', type=int, default=1, help='the number of layers of KGCN')
    parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')

    args = parser.parse_args()

    train(args, True)

'''
music	train_auc: 0.909 	 train_acc: 0.816 	 eval_auc: 0.814 	 eval_acc: 0.740 	 test_auc: 0.816 	 test_acc: 0.741 		[0.14, 0.27, 0.46, 0.48, 0.48, 0.51, 0.55, 0.56]
book	train_auc: 0.861 	 train_acc: 0.768 	 eval_auc: 0.729 	 eval_acc: 0.670 	 test_auc: 0.727 	 test_acc: 0.666 		[0.09, 0.14, 0.26, 0.28, 0.28, 0.33, 0.34, 0.36]
ml	train_auc: 0.938 	 train_acc: 0.865 	 eval_auc: 0.904 	 eval_acc: 0.827 	 test_auc: 0.907 	 test_acc: 0.829 		[0.25, 0.33, 0.56, 0.6, 0.6, 0.67, 0.69, 0.73]
yelp	train_auc: 0.867 	 train_acc: 0.793 	 eval_auc: 0.838 	 eval_acc: 0.764 	 test_auc: 0.837 	 test_acc: 0.763 		[0.12, 0.23, 0.35, 0.37, 0.37, 0.43, 0.45, 0.47]
'''