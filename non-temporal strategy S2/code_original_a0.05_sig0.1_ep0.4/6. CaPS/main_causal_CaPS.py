import argparse
import torch.nn.parallel
import torch.utils.data
from castle.metrics import MetricsDAG
from scipy.io import loadmat, savemat
from cdt.metrics import SID
from pathlib import Path
from cdt.utils.R import launch_R_script
import tempfile
import scipy.io
import os
import random
import torch
import pandas as pd
import numpy as np
import networkx as nx
import uuid

def is_acyclic(adj_matrix):
    G = nx.DiGraph(adj_matrix)
    return nx.is_directed_acyclic_graph(G)

def adj2order(adj_matrix):
    G = nx.DiGraph(adj_matrix)
    return list(nx.topological_sort(G))

def full_DAG(top_order):
    d = len(top_order)
    A = np.zeros((d,d))
    for i, var in enumerate(top_order):
        A[var, top_order[i+1:]] = 1
    return A

def fullAdj2Order(A):
    order = list(A.sum(axis=1).argsort())
    order.reverse()
    return order

def np_to_csv(array, save_path):
        """
        Convert np array to .csv
        array: numpy array
            the numpy array to convert to csv
        save_path: str
            where to temporarily save the csv
        Return the path to the csv file
        """
        id = str(uuid.uuid4())
        #output = os.path.join(os.path.dirname(save_path), 'tmp_' + id + '.csv')
        output = os.path.join(save_path, 'tmp_' + id + '.csv')

        df = pd.DataFrame(array)
        df.to_csv(output, header=False, index=False)

        return output

def normlize_data(data):
    mu = data.mean(axis=0)
    sigma = data.std(axis=0)
    data = (data-mu) / sigma
    return data

def get_parents_score(curr_H, full_H, i):
    full_H = torch.hstack([full_H[0:i], full_H[i+1:]])
    parents_score = np.abs(curr_H - full_H)
    parents_score = torch.cat([parents_score[:i], torch.tensor([0.0]), parents_score[i:]])
    return parents_score

def add_edge_with_parents_score(dag, parents_score, lambda2):
    avg_parent_score = (dag * parents_score.T).mean()
    add_edge = (parents_score.T >= lambda2 * avg_parent_score).astype(np.int32)
    idx = np.transpose(np.nonzero(add_edge))
    val = np.array([parents_score.T[i[0]][i[1]] for i in idx])
    sorted_idx = np.argsort(-val)
    idx = idx[sorted_idx]
    for i in idx:
        if dag[i[0], i[1]] == 1:
            continue
        else:
            dag[i[0], i[1]] = 1
            if is_acyclic(dag):
                continue
            else:
                dag[i[0], i[1]] = 0
    return dag

def pre_pruning_with_parents_score(init_dag, parents_score, lambda1):
    d = init_dag.shape[0]
    threshold = (np.max(parents_score, axis=1)/lambda1).reshape(d, 1)
    threshold = np.tile(threshold, (1, d))
    mask = (parents_score >= threshold).T.astype(np.int32)
    return init_dag * mask
torch.set_printoptions(linewidth=1000)
np.set_printoptions(linewidth=1000)


def blue(x): return '\033[94m' + x + '\033[0m'


def red(x): return '\033[31m' + x + '\033[0m'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model', type=str, default='CaPS')
    parser.add_argument('--pre_pruning', type=str, default=True)
    parser.add_argument('--add_edge', type=str, default=True)
    parser.add_argument('--lambda1', type=float, default=50.0)
    parser.add_argument('--lambda2', type=float, default=50.0)

    parser.add_argument('--dataset', type=str, default='sachs')
    parser.add_argument('--norm', type=bool, default=False)
    parser.add_argument('--num_nodes', type=int, default=10)
    parser.add_argument('--num_samples', type=int, default=2000)
    parser.add_argument('--method', type=str, default='mixed')
    parser.add_argument('--linear_sem_type', type=str, default='gauss')
    parser.add_argument('--nonlinear_sem_type', type=str, default='gp')
    parser.add_argument('--linear_rate', type=float, default=1.0)

    parser.add_argument('--manualSeed', type=str, default='False')
    parser.add_argument('--runs', type=int, default=1)

    args = parser.parse_args()
    args.manualSeed = True if args.manualSeed == 'True' else False
    return args


def order_divergence(order, adj):
    err = 0
    for i in range(len(order)):
        err += adj[order[i + 1:], order[i]].sum()
    return err


def evaluate(dag, GT_DAG):
    print('pred_dag:\n', dag)
    print('gt_dag:\n', GT_DAG.astype(int))
    print(blue('edge_num: ' + str(np.sum(dag))))
    mt = MetricsDAG(dag, GT_DAG)
    sid = SID(GT_DAG, dag)
    mt.metrics['sid'] = sid.item()
    print(blue(str(mt.metrics)))
    return mt.metrics


def cam_pruning(A, X, cutoff, only_pruning=True):
    with tempfile.TemporaryDirectory() as save_path:
        if only_pruning:
            pruning_path = Path(__file__).parent / "pruning_R_files/cam_pruning.R"

        data_np = X  # np.array(X.detach().cpu().numpy())
        data_csv_path = np_to_csv(data_np, save_path)
        dag_csv_path = np_to_csv(A, save_path)

        arguments = dict()
        arguments['{PATH_DATA}'] = data_csv_path
        arguments['{PATH_DAG}'] = dag_csv_path
        arguments['{PATH_RESULTS}'] = os.path.join(save_path, "results.csv")
        arguments['{ADJFULL_RESULTS}'] = os.path.join(save_path, "adjfull.csv")
        arguments['{CUTOFF}'] = str(cutoff)
        arguments['{VERBOSE}'] = "FALSE"  # TRUE, FALSE

        def retrieve_result():
            A = pd.read_csv(arguments['{PATH_RESULTS}']).values
            os.remove(arguments['{PATH_RESULTS}'])
            os.remove(arguments['{PATH_DATA}'])
            os.remove(arguments['{PATH_DAG}'])
            return A

        dag = launch_R_script(str(pruning_path), arguments, output_function=retrieve_result)

    return dag, adj2order(dag)


def train_test(args, train_set, GT_DAG, data_ls, runs):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if args.dataset in ['sachs', 'syntren']:
        x_in = train_set[0].shape[-1] - 1
    elif args.dataset.startswith('Syn'):
        x_in = args.num_nodes
    else:
        raise Exception('Dataset not recognized.')

    if args.model == 'CaPS':
        def Stein_hess(X, eta_G, eta_H, s=None):
            """
            Estimates the diagonal of the Hessian of log p_X at the provided samples points
            X, using first and second-order Stein identities
            """
            n, d = X.shape
            X = X.to(device)
            X_diff = X.unsqueeze(1) - X
            if s is None:
                D = torch.norm(X_diff, dim=2, p=2)
                s = D.flatten().median()
            K = torch.exp(-torch.norm(X_diff, dim=2, p=2) ** 2 / (2 * s ** 2)) / s

            nablaK = -torch.einsum('kij,ik->kj', X_diff, K) / s ** 2
            G = torch.matmul(torch.inverse(K + eta_G * torch.eye(n).to(device)), nablaK)

            nabla2K = torch.einsum('kij,ik->kj', -1 / s ** 2 + X_diff ** 2 / s ** 4, K)
            return (-G ** 2 + torch.matmul(torch.inverse(K + eta_H * torch.eye(n).to(device)), nabla2K)).to('cpu')

        def compute_top_order(X, eta_G, eta_H, dispersion="mean"):
            n, d = X.shape
            full_X = X
            order = []
            layer = [0 for _ in range(d)]
            active_nodes = list(range(d))
            for i in range(d - 1):
                H = Stein_hess(X, eta_G, eta_H)
                if dispersion == "mean":  # Lemma 1 of CaPS
                    l = int(H.mean(axis=0).argmax())
                else:
                    raise Exception("Unknown dispersion criterion")

                order.append(active_nodes[l])
                active_nodes.pop(l)

                X = torch.hstack([X[:, 0:l], X[:, l + 1:]])
            order.append(active_nodes[0])
            order.reverse()

            # compute parents score
            active_nodes = list(range(d))
            full_H = Stein_hess(full_X, eta_G, eta_H).mean(axis=0)
            parents_score = np.zeros((d, d))
            for i in range(d):
                curr_X = torch.hstack([full_X[:, 0:i], full_X[:, i + 1:]])
                curr_H = Stein_hess(curr_X, eta_G, eta_H).mean(axis=0)
                parents_score[i] = get_parents_score(curr_H, full_H, i)
            print(parents_score)

            return order, parents_score

        cutoff = 0.001
        train_set = torch.Tensor(train_set[:, 1:])

        order, parents_score = compute_top_order(train_set, eta_G=0.001, eta_H=0.001, dispersion="mean")
        print(blue(str(order)))

        if args.pre_pruning:
            init_dag = pre_pruning_with_parents_score(full_DAG(order), parents_score, args.lambda1)
        else:
            init_dag = full_DAG(order)

        dag, _ = cam_pruning(init_dag, train_set, cutoff)

        if args.add_edge:
            dag = add_edge_with_parents_score(dag, parents_score, args.lambda2)

    return dag


if __name__ == '__main__':
    parent_dir = os.path.join(os.getcwd(), '..')
    mat_files = [file for file in os.listdir(parent_dir) if
                 file.endswith('.mat') and file.startswith('data_numerical_var3')]

    for mat_file in mat_files:
        print(blue(f"Processing file: {mat_file}"))

        data_path = os.path.join(parent_dir, mat_file)
        data = loadmat(data_path)
        data_diff = data['data_diff']


        args = get_args()
        metrics_list_dict = {'fdr': [], 'tpr': [], 'fpr': [], 'shd': [], 'sid': [], 'nnz': [], 'precision': [],
                             'recall': [], 'F1': [], 'gscore': []}
        metrics_res_dict = {}
        t_ls = []
        D_top = []
        simulation_seeds = [1, 2, 3, 4, 5]

        num_columns = data_diff.shape[1]

        metrics_list = []  # 用于存储每次训练的 metrics_dict

        for i in range(num_columns):
            data = data_diff[0, i]  # 获取每个 cell 的矩阵
            num_samples = data.shape[0]
            train_set = np.concatenate([np.zeros([num_samples, 1]), data], axis=1)

            GT_DAG = np.array([
                [0, 1],
                [0, 0]
            ])

            data_ls = None

            if args.manualSeed:
                Seed = args.random_seed
            else:
                Seed = random.randint(1, 10000)

            random.seed(Seed)
            torch.manual_seed(Seed)
            np.random.seed(Seed)

            metrics_matrix = train_test(args, train_set, GT_DAG, data_ls, runs=i)
            metrics_list.append(metrics_matrix)


        # 构造 1x2 的 ndarray，每个元素是一个 array
        matlab_ready_array = np.empty((1, num_columns), dtype=object)  # 显式创建 1x2 的结构
        for i in range(num_columns):
            matlab_ready_array[0, i] = metrics_list[i]


        # 保存为 .mat 文件，文件名与当前处理的文件对应
        save_filename = f"result_CaPS_{mat_file.split('.')[0]}.mat"
        data_to_save = {
            'result': matlab_ready_array,
        }
        scipy.io.savemat(save_filename, data_to_save)
        print(blue(f"Results saved to {save_filename}"))
