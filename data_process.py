import numpy as np
import Levenshtein
import torch
import pandas as pd
import itertools
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_txt(path):
    with open(path, 'r', newline='') as txt_file:
        md_data = []
        reader = txt_file.readlines()
        for row in reader:
            line = row.split( )
            row = []
            for k in line:
                row.append(float(k))
            md_data.append(row)
        md_data = np.array(md_data)
        return md_data


def calculate_bandwidth(interaction_profile):
    squared_sum = np.sum(interaction_profile ** 2)
    n = len(interaction_profile)
    if n * squared_sum == 0:
        return np.inf
    bandwidth = 1 / (n * squared_sum)
    return bandwidth


def gaussian_kernel_similarity(IP1, IP2, bandwidth):
    diff = IP1 - IP2
    norm_squared = np.dot(diff, diff)
    if bandwidth == np.inf:
        return 0.0
    similarity = np.exp(-bandwidth * norm_squared)
    return similarity


def build_gaussian_similarity_matrix(IP, is_lncRNA=True):
    n = IP.shape[0] if is_lncRNA else IP.shape[1]
    GSM = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if is_lncRNA:
                IP_i = IP[i, :]
                IP_j = IP[j, :]
            else:
                IP_i = IP[:, i]
                IP_j = IP[:, j]
            bandwidth = calculate_bandwidth(IP_i) if is_lncRNA else calculate_bandwidth(IP_j)
            GSM[i, j] = gaussian_kernel_similarity(IP_i, IP_j, bandwidth)
    return GSM


def read_sequences(filepath):
    sequences = []
    with open(filepath, 'r') as file:
        for line in file:
            sequences.append(line.strip())
    return sequences


def calculate_similarity(seq1, seq2):
    len1, len2 = len(seq1), len(seq2)
    lev_distance = Levenshtein.distance(seq1, seq2)
    similarity = (max(len1, len2) - lev_distance) / max(len1, len2)
    return similarity


def build_similarity_matrix(sequences):
    n = len(sequences)
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            similarity = calculate_similarity(sequences[i], sequences[j])
            similarity_matrix[i, j] = similarity_matrix[j, i] = similarity
    return similarity_matrix


def graph(network, p):
    rows, cols = network.shape
    np.fill_diagonal(network, 0)
    PNN = np.zeros((rows, cols))
    graph = np.zeros((rows, cols))

    for i in range(rows):
        idx = np.argsort(-network[i, :])
        PNN[i, idx[:p]] = network[i, idx[:p]]
    for i in range(rows):
        idx_i = np.nonzero(PNN[i, :])[0]
        for j in range(rows):
            idx_j = np.nonzero(PNN[j, :])[0]
            if j in idx_i and i in idx_j:
                graph[i, j] = 1
            elif j not in idx_i and i not in idx_j:
                graph[i, j] = 0
            else:
                graph[i, j] = 0.5
    return graph


def read_expression_profiles(file_path):
    df = pd.read_csv(file_path, sep=' ')
    return df


def express_similarity(profile_a, profile_b):
    pa = np.array(profile_a, dtype=float)
    pb = np.array(profile_b, dtype=float)
    pa_mean = np.mean(pa)
    pb_mean = np.mean(pb)
    pa_diff = pa - pa_mean
    pb_diff = pb - pb_mean
    numerator = np.sum(pa_diff * pb_diff)
    denominator = np.sqrt(np.sum(pa_diff ** 2) * np.sum(pb_diff ** 2))
    correlation = numerator / denominator if denominator != 0 else 0
    similarity = (correlation + 1) / 2
    return similarity


def maxLength(file_path):
    with open(file_path, 'r') as f:
        lines = [line.strip().split(',') for line in f]
    miRNA_sequences = []
    lncRNA_sequences = []
    for line in lines:
        miRNA_name, lncRNA_name, miRNA_seq, lncRNA_seq, label = line
        miRNA_sequences.append(miRNA_seq.replace('U', 'T'))
        lncRNA_sequences.append(lncRNA_seq.replace('U', 'T'))
    mi_max_length = max(len(seq) for seq in miRNA_sequences)
    lnc_max_length = max(len(seq) for seq in lncRNA_sequences)
    return mi_max_length, lnc_max_length


def pad_sequences(sequences, max_length):
    padded_sequences = []
    for seq in sequences:
        n = max_length // len(seq)
        c = max_length % len(seq)
        padded_seq = (seq * n) + seq[:c]
        padded_sequences.append(padded_seq)
    return padded_sequences, max_length


def encode_high_order_one_hot(sequences, k):
    bases = ['A', 'G', 'C', 'T']
    k_mers = []
    for j in range(1, k + 1):
        k_mers.extend(''.join(p) for p in itertools.product(bases, repeat=j))
    encoding_map_length = len(k_mers)
    k_mer_to_index = {k_mer: i for i, k_mer in enumerate(k_mers)}
    k_mer_to_vector = {
        k_mer: np.eye(1, encoding_map_length, i, dtype=int)[0]
        for i, k_mer in enumerate(k_mers)
    }
    encoded_sequences = []
    for seq in sequences:
        encoded_seq = [
            k_mer_to_vector[seq[i:i + k]]
            for i in range(0, len(seq) - k + 1, k)
        ]
        remaining = len(seq) % k
        if remaining != 0:
            last_k_mer = seq[-remaining:]
            if last_k_mer in k_mer_to_vector:
                encoded_seq.append(k_mer_to_vector[last_k_mer])
            else:
                encoded_seq.append(np.zeros(encoding_map_length))
        encoded_sequences.append(np.array(encoded_seq))
    return encoded_sequences, k_mer_to_index, k_mer_to_vector


def adj_to_edge_index_weighted(adj_matrix):
    edge_indices = torch.nonzero(adj_matrix, as_tuple=False).t().contiguous()
    edge_weights = adj_matrix[edge_indices[0], edge_indices[1]]
    return edge_indices, edge_weights


def load_adj_matrix_from_file(file_path):
    with open(file_path, 'r') as file:
        matrix = []
        for line in file:
            row = [float(value) for value in line.strip().split(' ')]
            matrix.append(row)
        tensor_matrix = torch.tensor(matrix, dtype=torch.float)
    return tensor_matrix


def process_GCNdata(files):
    adj_matrices = [load_adj_matrix_from_file(file) for file in files]
    num_nodes = adj_matrices[0].size(0)
    num_features = adj_matrices[0].size(1)
    num_matrices = len(adj_matrices)
    edge_indices = []
    edge_weights = []
    for adj in adj_matrices:
        edge_index, edge_weight = adj_to_edge_index_weighted(adj)
        edge_indices.append(edge_index)
        edge_weights.append(edge_weight)
    feature_matrices = [torch.randn(adj_matrix.size(0), adj_matrix.size(1)) for adj_matrix in adj_matrices]
    return adj_matrices, num_nodes, num_features, num_matrices, feature_matrices, edge_indices, edge_weights


def move_tensors_to_device(data, device):
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            data[key] = value.to(device)
        elif isinstance(value, list):
            data[key] = [v.to(device) if isinstance(v, torch.Tensor) else v for v in value]
        elif isinstance(value, dict):
            data[key] = move_tensors_to_device(value, device)
    return data


def read_fasta_file(fasta_file_path):
    data = []
    with open(fasta_file_path, 'r') as file:
        for line in file:
            parts = line[1:].strip().split(',')
            miRNA_name = parts[0]
            lncRNA_name = parts[1]
            miRNA_sequence = parts[2]
            lncRNA_sequence = parts[3]
            label = parts[4]
            data.append([miRNA_name, lncRNA_name, miRNA_sequence, lncRNA_sequence, label])
    column_names = ['miRNA_name', 'lncRNA_name', 'miRNA_sequence', 'lncRNA_sequence', 'Label']
    df = pd.DataFrame(data, columns=column_names)
    unique_miRNAs = df['miRNA_name'].unique()
    unique_lncRNAs = df['lncRNA_name'].unique()
    miRNA_to_index = {miRNA: index for index, miRNA in enumerate(unique_miRNAs)}
    lncRNA_to_index = {lncRNA: index for index, lncRNA in enumerate(unique_lncRNAs)}
    return df, miRNA_to_index, lncRNA_to_index


def lnc_indicesTensor(df, lncRNA_to_index):
    indices = df['lncRNA_name'].apply(lambda x: lncRNA_to_index[x]).tolist()
    tensor = torch.tensor(indices, dtype=torch.long)
    return tensor


def mi_indicesTensor(df, miRNA_to_index):
    indices = df['miRNA_name'].apply(lambda x: miRNA_to_index[x]).tolist()
    tensor = torch.tensor(indices, dtype=torch.long)
    return tensor


def parse_txt(txt_file):
    sequences = []
    labels = []
    with open(txt_file, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) != 5:
                continue
            miRNAname = parts[0]
            lncRNAname = parts[1]
            miRNAseq = parts[2]
            lncRNAseq = parts[3]
            label = int(parts[4])
            sequences.append((miRNAname, lncRNAname, miRNAseq, lncRNAseq))
            labels.append(label)
    return sequences, labels


def normalize_tensor(tensor):
    min_val = np.min(tensor)
    max_val = np.max(tensor)
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor


