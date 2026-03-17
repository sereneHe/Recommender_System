import numpy as np
import pandas as pd


def read_graph(graph_file):
    with open(graph_file, "r") as file:
        lines = file.readlines()
    groundtruth = [tuple(line.strip().split()) for line in lines]
    return set(groundtruth)

def create_adj_matrix(vertices, edges):
    A = np.zeros((len(vertices), len(vertices)))
    A_inter = np.zeros((len(vertices), len(vertices)))
    for row_idx, row in enumerate(vertices):
        for col_idx, column in enumerate(vertices):
            if (f'{row}_lag0', f'{column}_lag0') in edges:
                A[row_idx, col_idx] = 1
            if (f'{row}_lag1', f'{column}_lag0') in edges:
                A_inter[row_idx, col_idx] = 1
    assert len(edges) == A.sum() + A_inter.sum()
    return A, A_inter


def load_data(n, granularity, p, data_path):
    from os import listdir
    from os.path import isfile, join
    data_files = [f for f in listdir(data_path) if isfile(join(data_path, f)) and f.endswith('.csv')]
    data = [pd.read_table(join(data_path, file), sep=',',index_col=0) for file in data_files]
    data_names = [file.split('_')[3] for file in data_files]

    result_df = None
    for i in range(len(data_names)):
        df = data[i].iloc[:(n+p)*granularity:granularity, [0]]
        df.rename(columns={df.columns[0]: data_names[i]}, inplace=True)
        if result_df is None:
            result_df = df
        else:
            result_df = pd.merge(result_df, df, left_index=True, right_index=True, how='inner')
    print(result_df.head().to_string())


    variables = data_names
    intra_nodes = [f'{v}_lag0' for v in variables]
    W_true = np.zeros((len(variables), len(variables))) # TODO: get the ground truth
    X = result_df.iloc[:n,:].to_numpy(copy=True)
    X = 1000 * X
    inter_nodes = []
    A_true = []
    Y = []
    for lag in range(p):
        inter_nodes.extend([f'{v}_lag{lag + 1}' for v in variables])
        A_true_i = np.zeros((len(variables), len(variables)))
        A_true.append(A_true_i)
        offset = lag + 1
        Y_i = result_df.iloc[offset:n+offset,:].to_numpy(copy=True)
        Y_i = 1000 * Y_i
        Y.append(Y_i)

    tabu_edges = generate_tabu_edges(intra_nodes)
    return W_true, W_true, A_true, X, Y, intra_nodes, inter_nodes, tabu_edges


def generate_tabu_edges(intra_nodes):
    intra_nodes = [name.split("_lag")[0] for name in intra_nodes]
    banks = {'48DGFE','05ABBF','8B69AP','06DABK','EFAGG9','2H6677','FH49GG','8D8575'}
    insurers = {'GG6EBT','DD359M', 'FF667M'}
    industrial = {'0H99B7','2H66B7','8A87AG','NN2A8G','6A516F'}
    groups = [banks, insurers, industrial]
    def check_if_common_group(node_i, node_j):
        for group in groups:
            if node_i in group and node_j in group:
                return True
        return False
    tabu_edges = []
    for i, node_i in enumerate(intra_nodes):
        for j, node_j in enumerate(intra_nodes):
            if not check_if_common_group(node_i, node_j) and i != j:
                tabu_edges.append((i, j))
    return tabu_edges