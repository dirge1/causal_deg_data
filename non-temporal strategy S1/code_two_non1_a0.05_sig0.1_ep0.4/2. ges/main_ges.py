import os
from scipy.io import loadmat
from castle.algorithms.ges.ges import GES
from castle.metrics import MetricsDAG
from cdt.metrics import SID
import numpy as np
import scipy.io


def blue(x): return '\033[94m' + x + '\033[0m'


def evaluate(dag, GT_DAG):
    print('pred_dag:\n', dag)
    print('gt_dag:\n', GT_DAG.astype(int))
    print(blue('edge_num: ' + str(np.sum(dag))))
    mt = MetricsDAG(dag, GT_DAG)
    sid = SID(GT_DAG, dag)
    mt.metrics['sid'] = sid.item()
    print(blue(str(mt.metrics)))
    return mt.metrics


if __name__ == '__main__':

    # 获取当前文件夹中的所有 .mat 文件
    parent_dir = os.path.join(os.getcwd(), '..')
    mat_files = [file for file in os.listdir(parent_dir) if
                 file.endswith('.mat') and file.startswith('data_numerical_var3')]

    GT_DAG = np.array([
        [0, 1],
        [0, 0]
    ])

    for mat_file in mat_files:
        print(blue(f"Processing file: {mat_file}"))

        data_path = os.path.join(parent_dir, mat_file)
        data = loadmat(data_path)
        data_diff = data['data_origin']

        num_columns = data_diff.shape[1]

        metrics_list = []  # 用于存储每次训练的 metrics_dict
        metrics_list_dict = {'fdr': [], 'tpr': [], 'fpr': [], 'shd': [], 'sid': [], 'nnz': [], 'precision': [],
                             'recall': [], 'F1': [], 'gscore': []}
        metrics_res_dict = {}

        for i in range(num_columns):
            cell_data = data_diff[0, i]  # 获取每个 cell 的矩阵

            # 使用 GES 算法进行因果学习
            algo = GES(criterion='bic', method='scatter')
            algo.learn(cell_data)
            metrics_matrix = np.array(algo.causal_matrix)
            metrics_list.append(metrics_matrix)

        # 构造 1x2 的 ndarray，每个元素是一个 array
        matlab_ready_array = np.empty((1, num_columns), dtype=object)  # 显式创建 1x2 的结构

        for i in range(num_columns):
            matlab_ready_array[0, i] = metrics_list[i]

        # 保存为 .mat 文件，文件名与当前处理的文件对应
        save_filename = f"result_ges_{mat_file.split('.')[0]}.mat"
        data_to_save = {
            'result': matlab_ready_array
        }
        scipy.io.savemat(save_filename, data_to_save)
        print(blue(f"Results saved to {save_filename}"))
