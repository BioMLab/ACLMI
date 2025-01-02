import torch.optim as optim
import copy
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import KFold
from model import *
from getUseData import *
from data_utils import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def train_GCN(labels, train_losses, idx_train):
    model.train()
    optimizer.zero_grad()
    output = model(data, 64, mi_indices, lnc_indices).to(device)
    labels = torch.tensor(labels, dtype=torch.float)
    train_labels = labels[idx_train]
    train_labels = torch.tensor(train_labels) if type(train_labels) is list else train_labels
    train_labels = train_labels.float().to(device)
    train_output = output[idx_train]
    loss_fn = torch.nn.BCELoss()
    loss = loss_fn(train_output, train_labels)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())


def validate_GCN(labels, val_losses, idx_val):
    model.eval()
    with torch.no_grad():
        output = model(data, 64, mi_indices, lnc_indices).to(device)
        loss_fn = torch.nn.BCELoss()
        labels_tensor = torch.tensor(labels, dtype=torch.float).to(device)
        labels_selected = labels_tensor[idx_val]
        output_selected = output[idx_val]
        val_loss = loss_fn(output_selected, labels_selected)
        val_losses.append(val_loss.item())
        labels_np = labels_selected.cpu().numpy()
        predictions_np = output_selected.cpu().numpy()
        calc_metrics_results = metrics1(labels_np, predictions_np)
    model.train()
    return val_loss.item(), calc_metrics_results, labels_np, predictions_np


def train_transformer(model, train_loader, optimizer, loss_function, device):
    model.train()
    total_loss = 0
    for mi_seq1, lnc_seq1, mi_seq2, lnc_seq2, mi_seq3, lnc_seq3, labels in train_loader:
        mi_seq1, lnc_seq1, mi_seq2, lnc_seq2, mi_seq3, lnc_seq3, labels = \
            mi_seq1.to(device), lnc_seq1.to(device), \
                mi_seq2.to(device), lnc_seq2.to(device), \
                mi_seq3.to(device), lnc_seq3.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(mi_seq1, lnc_seq1, mi_seq2, lnc_seq2, mi_seq3, lnc_seq3)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def validate_transformer(model, valid_loader, loss_function, device):
    model.eval()
    valid_loss = 0
    all_targets = []
    all_outputs = []
    with torch.no_grad():
        for mi_seq1, lnc_seq1, mi_seq2, lnc_seq2, mi_seq3, lnc_seq3, labels in valid_loader:
            mi_seq1, lnc_seq1, mi_seq2, lnc_seq2, mi_seq3, lnc_seq3, labels = \
                mi_seq1.to(device), lnc_seq1.to(device), \
                    mi_seq2.to(device), lnc_seq2.to(device), \
                    mi_seq3.to(device), lnc_seq3.to(device), labels.to(device)
            outputs = model(mi_seq1, lnc_seq1, mi_seq2, lnc_seq2, mi_seq3, lnc_seq3)
            loss = loss_function(outputs, labels)
            valid_loss += loss.item()
            all_targets.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.cpu().detach().numpy())
    metrics = metrics2(all_targets, all_outputs)
    return valid_loss / len(valid_loader), metrics


torch.random.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# file_path = './database/dataset/crossValidation.fasta'
file_path = 'database/ExampleData/ExampleCrossValidation.fasta'

# GCN
data = getUseDate(file_path)
data = move_tensors_to_device(data, device)
df, miRNAIndex, lncRNAIndex = read_fasta_file(file_path)
mi_indices = mi_indicesTensor(df, miRNAIndex)
lnc_indices = lnc_indicesTensor(df, lncRNAIndex)
sequences, labels = parse_txt(file_path)
print("GCN data already load")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_performance = []
best_fold_metrics = []
model_states = {}
fold_predictions = []
fold_labels = []
for fold, (idx_train, idx_val) in enumerate(kf.split(sequences)):
    model = GCNNet(data, 0.2, 64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    train_losses = []
    val_losses = []
    best_model_wts = None
    best_val_loss = float('inf')
    best_epoch_metrics = {}
    patience = 20
    epochs_since_improvement = 0
    for epoch in range(100):
        train_GCN(labels, train_losses, idx_train)
        val_loss, epoch_metrics, y_true, y_score = validate_GCN(labels, val_losses, idx_val)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = model.state_dict()
            best_epoch_metrics = epoch_metrics
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
        if epochs_since_improvement >= patience:
            break
    fold_performance.append({
        'fold': fold,
        'metrics': best_epoch_metrics,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    })
    model_states[fold] = best_model_wts

best_fold_index = np.argmin([fm['best_val_loss'] for fm in fold_performance])
best_fold_performance = fold_performance[best_fold_index]
# torch.save(model_states[best_fold_index], f'./database/ExampleData/saveModel/best_model1.pth')
torch.save(model_states[best_fold_index], f'./database/dataset/saveModel/best_model1.pth')

GCN_model = GCNNet(data, 0.2, 64).to(device)
# GCN_model.load_state_dict(torch.load('./database/ExampleData/saveModel/best_model1.pth'))
GCN_model.load_state_dict(torch.load('./database/dataset/saveModel/best_model1.pth'))
GCN_model.eval()
GCN_output = GCN_model(data, 64, mi_indices, lnc_indices, return_features=True)
GCN_features_mi, GCN_features_lnc = GCN_output
GCN_features_mi = GCN_features_mi.detach().cpu().numpy()
GCN_features_lnc = GCN_features_lnc.detach().cpu().numpy()
# np.save('./database/ExampleData/feature/GCN_features_mi.npy', GCN_features_mi)
# np.save('./database/ExampleData/feature/GCN_features_lnc.npy', GCN_features_lnc)
np.save('./database/dataset/feature/GCN_features_mi.npy', GCN_features_mi)
np.save('./database/dataset/feature/GCN_features_lnc.npy', GCN_features_lnc)
print("GCN feature already get!")


# transformer
mi_max_length, lnc_max_length = maxLength(file_path)
mi_seq1, mi_seq_len1, lnc_seq1, lnc_seq_len1, y_labels, d_input1 = process_file(file_path, 2, mi_max_length, lnc_max_length)
mi_seq2, mi_seq_len2, lnc_seq2, lnc_seq_len2, _, d_input2 = process_file(file_path, 3, mi_max_length, lnc_max_length)
mi_seq3, mi_seq_len3, lnc_seq3, lnc_seq_len3, _, d_input3 = process_file(file_path, 4, mi_max_length, lnc_max_length)
train_dataset = TensorDataset(mi_seq1, lnc_seq1, mi_seq2, lnc_seq2, mi_seq3, lnc_seq3, y_labels)
print("Transformer data already load")

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
best_models = []
best_loss = []
best_metrics_per_fold = []
valid_loaders = []
fold_predictions = []
fold_labels = []
for fold, (train_ids, valid_ids) in enumerate(kfold.split(train_dataset)):
    train_subsampler = Subset(train_dataset, train_ids)
    valid_subsampler = Subset(train_dataset, valid_ids)
    train_loader = DataLoader(train_subsampler, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_subsampler, batch_size=64, shuffle=False)
    valid_loaders.append(valid_loader)
    model = TransformerNet(mi_seq_len1=mi_seq_len1, lnc_seq_len1=lnc_seq_len1,
                            mi_seq_len2=mi_seq_len2, lnc_seq_len2=lnc_seq_len2,
                            mi_seq_len3=mi_seq_len3, lnc_seq_len3=lnc_seq_len3,
                            d_input1=d_input1, d_input2=d_input2, d_input3=d_input3,
                            d_model=64, dropout=0.05, num_matrices=3, batch_size=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = torch.nn.BCELoss()
    best_fold_loss = float('inf')
    best_fold_model = None
    best_fold_metrics = {}
    train_losses = []
    valid_losses = []
    patience = 20
    no_improvement_count = 0
    for epoch in range(100):
        train_loss = train_transformer(model, train_loader, optimizer, loss_function, device)
        valid_loss, metrics = validate_transformer(model, valid_loader, loss_function, device)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        if valid_loss < best_fold_loss:
            best_fold_loss = valid_loss
            best_fold_model = copy.deepcopy(model)
            best_fold_metrics = metrics
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                break
    best_metrics_per_fold.append(best_fold_metrics)
    best_models.append(best_fold_model)
    best_loss.append(best_fold_loss)

    best_fold_model.eval()
    fold_predictions.append([])
    fold_labels.append([])
    with torch.no_grad():
        for mi_seq1, lnc_seq1, mi_seq2, lnc_seq2, mi_seq3, lnc_seq3, labels in valid_loader:
            mi_seq1, lnc_seq1, mi_seq2, lnc_seq2, mi_seq3, lnc_seq3, labels = mi_seq1.to(device), lnc_seq1.to(
                device), mi_seq2.to(device), lnc_seq2.to(device), mi_seq3.to(device), lnc_seq3.to(
                device), labels.to(device)
            mi_seq1.to(device), lnc_seq1.to(device), \
                mi_seq2.to(device), lnc_seq2.to(device), \
                mi_seq3.to(device), lnc_seq3.to(device), labels.to(device)

            outputs = best_fold_model(mi_seq1, lnc_seq1, mi_seq2, lnc_seq2, mi_seq3, lnc_seq3)
            fold_predictions[-1].append(outputs.cpu().numpy())
            fold_labels[-1].append(labels.cpu().numpy())

best_index = best_loss.index(min(best_loss))
best_model = best_models[best_index]
# torch.save(best_model.state_dict(), './database/ExampleData/saveModel/best_model2.pth')
torch.save(best_model.state_dict(), './database/dataset/saveModel/best_model2.pth')

transformer_model = TransformerNet(mi_seq_len1=mi_seq_len1, lnc_seq_len1=lnc_seq_len1,
                                    mi_seq_len2=mi_seq_len2, lnc_seq_len2=lnc_seq_len2,
                                    mi_seq_len3=mi_seq_len3, lnc_seq_len3=lnc_seq_len3,
                                    d_input1=d_input1, d_input2=d_input2, d_input3=d_input3,
                                    d_model=64, dropout=0.05, num_matrices=3, batch_size=64).to(device)
# transformer_model.load_state_dict(torch.load('./database/ExampleData/saveModel/best_model2.pth'))
transformer_model.load_state_dict(torch.load('./database/dataset/saveModel/best_model2.pth'))

transformer_model.eval()
transformer_features_mi = []
transformer_features_lnc = []
for valid_loader in valid_loaders:
    for mi_seq1, lnc_seq1, mi_seq2, lnc_seq2, mi_seq3, lnc_seq3, labels in valid_loader:
        mi_seq1, lnc_seq1, mi_seq2, lnc_seq2, mi_seq3, lnc_seq3, labels = \
            mi_seq1.to(device), lnc_seq1.to(device), \
                mi_seq2.to(device), lnc_seq2.to(device), \
                mi_seq3.to(device), lnc_seq3.to(device), labels.to(device)
        transformer_output = transformer_model(mi_seq1, lnc_seq1, mi_seq2, lnc_seq2, mi_seq3, lnc_seq3, return_features=True)
        transformer_feat_mi, transformer_feat_lnc = transformer_output
        transformer_features_mi.append(transformer_feat_mi.detach().cpu().numpy())
        transformer_features_lnc.append(transformer_feat_lnc.detach().cpu().numpy())
transformer_features_mi = np.concatenate(transformer_features_mi, axis=0)
transformer_features_lnc = np.concatenate(transformer_features_lnc, axis=0)
# np.save('./database/ExampleData/feature/transformer_features_mi.npy', transformer_features_mi)
# np.save('./database/ExampleData/feature/transformer_features_lnc.npy', transformer_features_lnc)
np.save('./database/dataset/feature/transformer_features_mi.npy', transformer_features_mi)
np.save('./database/dataset/feature/transformer_features_lnc.npy', transformer_features_lnc)
print("Transformer feature already get!")