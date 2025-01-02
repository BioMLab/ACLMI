from data_process import *
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def process_file(file_path, k, mi_max_length, lnc_max_length):
    with open(file_path, 'r') as f:
        lines = [line.strip().split(',') for line in f]

    miRNA_sequences = []
    lncRNA_sequences = []
    labels = []
    for line in lines:
        miRNA_name, lncRNA_name, miRNA_seq, lncRNA_seq, label = line
        miRNA_sequences.append(miRNA_seq.replace('U', 'T'))
        lncRNA_sequences.append(lncRNA_seq.replace('U', 'T'))
        labels.append(label)

    padded_miRNA, max_length_miRNA = pad_sequences(miRNA_sequences, mi_max_length)
    padded_lncRNA, max_length_lncRNA = pad_sequences(lncRNA_sequences, lnc_max_length)
    encoded_miRNA, miRNA_k_mer_to_index, miRNA_k_mer_to_vector = encode_high_order_one_hot(padded_miRNA, k)
    encoded_lncRNA, lncRNA_k_mer_to_index, lncRNA_k_mer_to_vector = encode_high_order_one_hot(padded_lncRNA, k)

    encoded_miRNA = np.array(encoded_miRNA)
    encoded_miRNA = torch.from_numpy(encoded_miRNA).float()
    encoded_miRNA = encoded_miRNA.to(device)
    encoded_lncRNA = np.array(encoded_lncRNA)
    encoded_lncRNA = torch.from_numpy(encoded_lncRNA).float()
    encoded_lncRNA = encoded_lncRNA.to(device)
    label_integers = [int(label) for label in labels]
    y_labels = torch.tensor(label_integers, dtype=torch.float)
    y_labels = y_labels.to(device)

    mi_seq_len = encoded_miRNA.shape[1]
    lnc_seq_len = encoded_lncRNA.shape[1]
    d_input = encoded_miRNA.shape[2]
    return encoded_miRNA, mi_seq_len, encoded_lncRNA, lnc_seq_len, y_labels, d_input


def getUseDate(filepath):
    data_dict = {}
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) != 5:
                continue
            miRNA_name = parts[0]
            lncRNA_name = parts[1]
            label = parts[4]
            if lncRNA_name not in data_dict:
                data_dict[lncRNA_name] = {}
            data_dict[lncRNA_name][miRNA_name] = label
    lncRNA_list = sorted(data_dict.keys())
    miRNA_list = sorted({miRNA for lncRNA_data in data_dict.values() for miRNA in lncRNA_data.keys()})
    matrix = []
    for lncRNA in lncRNA_list:
        row = []
        for miRNA in miRNA_list:
            if miRNA in data_dict[lncRNA]:
                row.append(data_dict[lncRNA][miRNA])
            else:
                row.append('2')
        matrix.append(row)
    # with open('./database/ExampleData/processedData/lncRNA-miRNA interaction.txt', 'w') as output_file:
    with open('./database/dataset/processedData/lncRNA-miRNA interaction.txt', 'w') as output_file:
        for row in matrix:
            output_file.write(' '.join(row) + '\n')

    miRNA_sequences = {}
    lncRNA_sequences = {}
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) != 5:
                continue
            miRNA_name = parts[0]
            miRNA_seq = parts[2]
            if miRNA_name not in miRNA_sequences:
                miRNA_sequences[miRNA_name] = miRNA_seq
            lncRNA_name = parts[1]
            lncRNA_seq = parts[3]
            if lncRNA_name not in lncRNA_sequences:
                lncRNA_sequences[lncRNA_name] = lncRNA_seq

    unique_miRNA_names = list(miRNA_sequences.keys())
    unique_lncRNA_names = list(lncRNA_sequences.keys())
    print(f'不重复的miRNA数量: {len(unique_miRNA_names)}')
    print(f'不重复的lncRNA数量: {len(unique_lncRNA_names)}')

    # with open('./database/ExampleData/rawData/miRNASequence.txt', 'w') as output_file:
    with open('./database/dataset/rawData/miRNASequence.txt', 'w') as output_file:
        for miRNA_name, miRNA_seq in miRNA_sequences.items():
            output_file.write(f"{miRNA_seq}\n")
    # with open('./database/ExampleData/rawData/lncRNASequence.txt', 'w') as output_file:
    with open('./database/dataset/rawData/lncRNASequence.txt', 'w') as output_file:
        for lncRNA_name, lncRNA_seq in lncRNA_sequences.items():
            output_file.write(f"{lncRNA_seq}\n")

    # miRNA_expression_profiles = pd.read_csv('./database/ExampleData/rawData/miRNAExpression.txt', sep=',')
    # lncRNA_expression_profiles = pd.read_csv('./database/ExampleData/rawData/lncRNAExpression.txt', sep=',')
    miRNA_expression_profiles = pd.read_csv('./database/dataset/rawData/miRNAExpression.txt', sep=',')
    lncRNA_expression_profiles = pd.read_csv('./database/dataset/rawData/lncRNAExpression.txt', sep=',')
    miRNA_profiles = miRNA_expression_profiles[miRNA_expression_profiles['miRNA名称'].isin(unique_miRNA_names)]
    lncRNA_profiles = lncRNA_expression_profiles[lncRNA_expression_profiles['lncRNA名称'].isin(unique_lncRNA_names)]
    # miRNA_profiles.to_csv('./database/ExampleData/processedData/used_miRNAExpression.txt', index=False, sep=' ')
    # lncRNA_profiles.to_csv('./database/ExampleData/processedData/used_lncRNAExpression.txt', index=False, sep=' ')
    miRNA_profiles.to_csv('./database/dataset/processedData/used_miRNAExpression.txt', index=False, sep=' ')
    lncRNA_profiles.to_csv('./database/dataset/processedData/used_lncRNAExpression.txt', index=False, sep=' ')

    # sequences similarity
    # lncRNA_sequences = read_sequences('./database/ExampleData/rawData/lncRNASequence.txt')
    # miRNA_sequences = read_sequences('./database/ExampleData/rawData/miRNASequence.txt')
    lncRNA_sequences = read_sequences('./database/dataset/rawData/lncRNASequence.txt')
    miRNA_sequences = read_sequences('./database/dataset/rawData/miRNASequence.txt')
    lncRNA_sequences_similarity = build_similarity_matrix(lncRNA_sequences)
    miRNA_sequences_similarity = build_similarity_matrix(miRNA_sequences)
    # np.savetxt('./database/ExampleData/processedData/lncRNA_sequences_similarity.txt', lncRNA_sequences_similarity, fmt='%f')
    # np.savetxt('./database/ExampleData/processedData/miRNA_sequences_similarity.txt', miRNA_sequences_similarity, fmt='%f')
    np.savetxt('./database/dataset/processedData/lncRNA_sequences_similarity.txt', lncRNA_sequences_similarity, fmt='%f')
    np.savetxt('./database/dataset/processedData/miRNA_sequences_similarity.txt', miRNA_sequences_similarity, fmt='%f')

    lncRNA_sequences_P = graph(lncRNA_sequences_similarity, 5)
    miRNA_sequences_P = graph(miRNA_sequences_similarity, 5)
    lncRNA_sequences_P = lncRNA_sequences_P * lncRNA_sequences_similarity
    miRNA_sequences_P = miRNA_sequences_P * miRNA_sequences_similarity
    # np.savetxt('./database/ExampleData/processedData/lncRNA_sequences_P.txt', lncRNA_sequences_P, fmt='%f')
    # np.savetxt('./database/ExampleData/processedData/miRNA_sequences_P.txt', miRNA_sequences_P, fmt='%f')
    np.savetxt('./database/dataset/processedData/lncRNA_sequences_P.txt', lncRNA_sequences_P, fmt='%f')
    np.savetxt('./database/dataset/processedData/miRNA_sequences_P.txt', miRNA_sequences_P, fmt='%f')
    print("miRNA_sequences_similarity:", miRNA_sequences_P.shape)
    print("lncRNA_sequences_similarity:", lncRNA_sequences_P.shape)


    # GIP similarity
    # A = read_txt('./database/ExampleData/processedData/lncRNA-miRNA interaction.txt')
    A = read_txt('./database/dataset/processedData/lncRNA-miRNA interaction.txt')
    GSL = build_gaussian_similarity_matrix(A, is_lncRNA=True)
    GSM = build_gaussian_similarity_matrix(A, is_lncRNA=False)
    # np.savetxt('./database/ExampleData/processedData/lncRNA_GIP_similarity.txt', GSL, fmt='%f')
    # np.savetxt('./database/ExampleData/processedData/miRNA_GIP_similarity.txt', GSM, fmt='%f')
    np.savetxt('./database/dataset/processedData/lncRNA_GIP_similarity.txt', GSL, fmt='%f')
    np.savetxt('./database/dataset/processedData/miRNA_GIP_similarity.txt', GSM, fmt='%f')

    lncRNA_GIP_P = graph(GSL, 5)
    miRNA_GIP_P = graph(GSM, 5)
    lncRNA_GIP_P = lncRNA_GIP_P * GSL
    miRNA_GIP_P = miRNA_GIP_P * GSM
    # np.savetxt('./database/ExampleData/processedData/lncRNA_GIP_P.txt', lncRNA_GIP_P, fmt='%f')
    # np.savetxt('./database/ExampleData/processedData/miRNA_GIP_P.txt', miRNA_GIP_P, fmt='%f')
    np.savetxt('./database/dataset/processedData/lncRNA_GIP_P.txt', lncRNA_GIP_P, fmt='%f')
    np.savetxt('./database/dataset/processedData/miRNA_GIP_P.txt', miRNA_GIP_P, fmt='%f')
    print("miRNA_GIP_similarity:", miRNA_GIP_P.shape)
    print("lncRNA_GIP_similarity:", lncRNA_GIP_P.shape)


    # express similarity
    # miRNA_profiles = read_expression_profiles('./database/ExampleData/processedData/used_miRNAExpression.txt').set_index('miRNA名称').values
    # lncRNA_profiles = read_expression_profiles('./database/ExampleData/processedData/used_lncRNAExpression.txt').set_index('lncRNA名称').values
    miRNA_profiles = read_expression_profiles('./database/dataset/processedData/used_miRNAExpression.txt').set_index('miRNA名称').values
    lncRNA_profiles = read_expression_profiles('./database/dataset/processedData/used_lncRNAExpression.txt').set_index('lncRNA名称').values
    miRNA_express_similarity = np.zeros((len(unique_miRNA_names), len(unique_miRNA_names)))
    lncRNA_express_similarity = np.zeros((len(unique_lncRNA_names), len(unique_lncRNA_names)))
    for i in range(len(unique_miRNA_names)):
        for j in range(len(unique_miRNA_names)):
            miRNA_express_similarity[i, j] = express_similarity(miRNA_profiles[i], miRNA_profiles[j])
    for i in range(len(unique_lncRNA_names)):
        for j in range(len(unique_lncRNA_names)):
            lncRNA_express_similarity[i, j] = express_similarity(lncRNA_profiles[i], lncRNA_profiles[j])
    # np.savetxt('./database/ExampleData/processedData/miRNA_express_similarity.txt', miRNA_express_similarity, delimiter=' ')
    # np.savetxt('./database/ExampleData/processedData/lncRNA_express_similarity.txt', lncRNA_express_similarity, delimiter=' ')
    np.savetxt('./database/dataset/processedData/miRNA_express_similarity.txt', miRNA_express_similarity, delimiter=' ')
    np.savetxt('./database/dataset/processedData/lncRNA_express_similarity.txt', lncRNA_express_similarity, delimiter=' ')

    lncRNA_express_P = graph(lncRNA_express_similarity, 5)
    miRNA_express_P = graph(miRNA_express_similarity, 5)
    lncRNA_express_P = lncRNA_express_P * lncRNA_express_similarity
    miRNA_express_P = miRNA_express_P * miRNA_express_similarity
    # np.savetxt('./database/ExampleData/processedData/lncRNA_express_P.txt', lncRNA_express_P, fmt='%f')
    # np.savetxt('./database/ExampleData/processedData/miRNA_express_P.txt', miRNA_express_P, fmt='%f')
    np.savetxt('./database/dataset/processedData/lncRNA_express_P.txt', lncRNA_express_P, fmt='%f')
    np.savetxt('./database/dataset/processedData/miRNA_express_P.txt', miRNA_express_P, fmt='%f')
    print("miRNA_express_similarity:", miRNA_express_P.shape)
    print("lncRNA_express_similarity:", lncRNA_express_P.shape)


    # mi_files = ['./database/ExampleData/processedData/miRNA_sequences_P.txt',
    #             './database/ExampleData/processedData/miRNA_GIP_P.txt',
    #             './database/ExampleData/processedData/miRNA_express_P.txt']
    mi_files = ['./database/dataset/processedData/miRNA_sequences_P.txt',
                './database/dataset/processedData/miRNA_GIP_P.txt',
                './database/dataset/processedData/miRNA_express_P.txt']
    mi_adj_matrices, mi_num_nodes, mi_num_features, mi_num_matrices, mi_feature_matrices,\
        mi_edge_indices, mi_edge_weights = process_GCNdata(mi_files)

    # lnc_files = ['./database/ExampleData/processedData/lncRNA_sequences_P.txt',
    #              './database/ExampleData/processedData/lncRNA_GIP_P.txt',
    #              './database/ExampleData/processedData/lncRNA_express_P.txt']
    lnc_files = ['./database/dataset/processedData/lncRNA_sequences_P.txt',
                 './database/dataset/processedData/lncRNA_GIP_P.txt',
                 './database/dataset/processedData/lncRNA_express_P.txt']
    lnc_adj_matrices, lnc_num_nodes, lnc_num_features, lnc_num_matrices, lnc_feature_matrices,\
        lnc_edge_indices, lnc_edge_weights = process_GCNdata(lnc_files)

    # l_m_adj_matrices = read_txt("./database/ExampleData/processedData/lncRNA-miRNA interaction.txt")
    l_m_adj_matrices = read_txt("./database/dataset/processedData/lncRNA-miRNA interaction.txt")
    l_m_adj_matrices = torch.tensor(l_m_adj_matrices, dtype=torch.float32)

    data = {'mi_x': mi_feature_matrices,
            'mi_edge_index': mi_edge_indices,
            'mi_edge_weights': mi_edge_weights,
            'mi_num_nodes': mi_num_nodes,
            'mi_num_features': mi_num_features,
            'mi_num_matrices': mi_num_matrices,

            'lnc_x': lnc_feature_matrices,
            'lnc_edge_index': lnc_edge_indices,
            'lnc_edge_weights': lnc_edge_weights,
            'lnc_num_nodes': lnc_num_nodes,
            'lnc_num_features': lnc_num_features,
            'lnc_num_matrices': lnc_num_matrices,

            'l_m_adj_matrices': l_m_adj_matrices
            }
    return data




