from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import KFold
import copy
import argparse
from sklearn.model_selection import StratifiedKFold
from data_process import parse_txt, normalize_tensor
from data_utils import *
from model import *


# 单层
# class KAN(nn.Module):
#     def __init__(self, input_dim):
#         super(KAN, self).__init__()
#         self.kan_layer = KANLinear(input_dim, 1)
#         self.bn = nn.BatchNorm1d(input_dim)
#         # self.ln = nn.LayerNorm(input_dim)
#         self.constrast = SupConLoss()

#     def forward(self, GCN_mi, trans_mi, GCN_lnc, trans_lnc):
#         mi_con = self.constrast(GCN_mi, trans_mi, labels=None, mask=None)
#         lnc_con = self.constrast(GCN_lnc, trans_lnc, labels=None, mask=None)
#         con = mi_con + lnc_con
#         mi = torch.cat((GCN_mi, trans_mi), dim=1)
#         lnc = torch.cat((GCN_lnc, trans_lnc), dim=1)
#         features = torch.cat((mi, lnc), dim=1)

#         features = self.bn(features)
#         # features = self.ln(features)

#         x = self.kan_layer(features)
#         return x, con


# 多层
class KAN(nn.Module):
    def __init__(self, input_dim, num_layers=2, hidden_dim=256, dropout_prob=0.5):
        super(KAN, self).__init__()
        self.kan_layers = nn.ModuleList([
            nn.Sequential(
                KANLinear(input_dim if i == 0 else hidden_dim, hidden_dim), 
                nn.Dropout(dropout_prob) 
            ) for i in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.bn = nn.BatchNorm1d(input_dim) 
        self.constrast = SupConLoss()

    def forward(self, GCN_mi, trans_mi, GCN_lnc, trans_lnc):
        mi_con = self.constrast(GCN_mi, trans_mi, labels=None, mask=None)
        lnc_con = self.constrast(GCN_lnc, trans_lnc, labels=None, mask=None)
        con = mi_con + lnc_con

        mi = torch.cat((GCN_mi, trans_mi), dim=1)
        lnc = torch.cat((GCN_lnc, trans_lnc), dim=1)
        features = torch.cat((mi, lnc), dim=1)
        features = self.bn(features)  
        
        for layer in self.kan_layers:
            # residual = features.clone()
            features = layer(features) 
            # features = features + residual  

        output = self.output_layer(features)
        return output, con
    

def train_KAN(model, loader, optimizer, loss_function, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for GCN_mi, trans_mi, GCN_lnc, trans_lnc, labels in loader:
        GCN_mi, trans_mi, GCN_lnc, trans_lnc, labels = GCN_mi.to(device), trans_mi.to(device), GCN_lnc.to(
            device), trans_lnc.to(device), labels.to(device)
        optimizer.zero_grad()
        labels = labels.unsqueeze(1)
        outputs, con = model(GCN_mi, trans_mi, GCN_lnc, trans_lnc)
        outputs = torch.sigmoid(outputs)
        loss = loss_function(outputs, labels) + con
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)
    return total_loss / total_samples


def validate_KAN(model, loader, loss_function, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for GCN_mi, trans_mi, GCN_lnc, trans_lnc, labels in loader:
            GCN_mi, trans_mi, GCN_lnc, trans_lnc, labels = GCN_mi.to(device), trans_mi.to(device), GCN_lnc.to(
                device), trans_lnc.to(device), labels.to(device)
            outputs, con = model(GCN_mi, trans_mi, GCN_lnc, trans_lnc)
            outputs = torch.sigmoid(outputs)
            labels = labels.unsqueeze(1)
            loss = loss_function(outputs, labels) + con
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
            all_predictions.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    metrics = metrics_util(all_predictions, all_labels)
    return total_loss / total_samples, metrics


class MyDataset(Dataset):
    def __init__(self, mi_features_GCN, mi_features_trans, lnc_features_GCN, lnc_features_trans, labels):
        self.mi_features_GCN = mi_features_GCN
        self.mi_features_trans = mi_features_trans
        self.lnc_features_GCN = lnc_features_GCN
        self.lnc_features_trans = lnc_features_trans
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mi_feature_GCN = self.mi_features_GCN[idx]
        lnc_feature_GCN = self.lnc_features_GCN[idx]
        mi_feature_trans = self.mi_features_trans[idx]
        lnc_feature_trans = self.lnc_features_trans[idx]
        label = self.labels[idx]
        return mi_feature_GCN, lnc_feature_GCN, mi_feature_trans, lnc_feature_trans, label


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args')
    parser.add_argument('--seed', dest='seed', default=42, type=int, help='Random seed')
    parser.add_argument('--lr', dest='lr', default=0.001, type=float)
    parser.add_argument('--wd', dest='wd', default=5e-4, type=float)
    parser.add_argument('--epoch', dest='epoch', default=100, type=int, help='Training epochs')
    parser.add_argument('--dropout', dest='dropout', default=0.2, type=int, help='dropout')
    args = parser.parse_args()
    torch.random.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # GCN_features_mi = np.load('./database/ExampleData/feature/GCN_features_mi.npy')
    # GCN_features_lnc = np.load('./database/ExampleData/feature/GCN_features_lnc.npy')
    # transformer_features_mi = np.load('./database/ExampleData/feature/transformer_features_mi.npy')
    # transformer_features_lnc = np.load('./database/ExampleData/feature/transformer_features_lnc.npy')
    GCN_features_mi = np.load('./database/dataset/feature/GCN_features_mi.npy')
    GCN_features_lnc = np.load('./database/dataset/feature/GCN_features_lnc.npy')
    transformer_features_mi = np.load('./database/dataset/feature/transformer_features_mi.npy')
    transformer_features_lnc = np.load('./database/dataset/feature/transformer_features_lnc.npy')

    GCN_features_mi = normalize_tensor(GCN_features_mi)
    GCN_features_lnc = normalize_tensor(GCN_features_lnc)
    transformer_features_mi = normalize_tensor(transformer_features_mi)
    transformer_features_lnc = normalize_tensor(transformer_features_lnc)

    GCN_features_mi = torch.tensor(GCN_features_mi, dtype=torch.float)
    transformer_features_mi = torch.tensor(transformer_features_mi, dtype=torch.float)
    GCN_features_lnc = torch.tensor(GCN_features_lnc, dtype=torch.float)
    transformer_features_lnc = torch.tensor(transformer_features_lnc, dtype=torch.float)

    # sequences, labels = parse_txt('./database/dataset/crossValidation.fasta')
    sequences, labels = parse_txt('database/ExampleData/ExampleCrossValidation.fasta')
    labels = torch.tensor(labels, dtype=torch.float)
    dataset = MyDataset(GCN_features_mi, transformer_features_mi, GCN_features_lnc, transformer_features_lnc, labels)
    print("Data already load")

    # kfold = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    fold_metrics = []
    best_metrics_per_fold = []
    val_datasets = []

    # for fold, (train_ids, valid_ids) in enumerate(kfold.split(dataset)):
    for fold, (train_ids, valid_ids) in enumerate(kfold.split(dataset, labels)):
        print(f"Fold {fold + 1}")
        train_subsampler = Subset(dataset, train_ids)
        valid_subsampler = Subset(dataset, valid_ids)
        train_loader = DataLoader(train_subsampler, batch_size=96, shuffle=True)
        valid_loader = DataLoader(valid_subsampler, batch_size=96, shuffle=False)
        val_datasets.append(valid_subsampler)

        kan_model = KAN(input_dim=256).to(device)
        optimizer = torch.optim.SGD(kan_model.parameters(), lr=args.lr, weight_decay=args.wd)
        loss_function = torch.nn.BCELoss()

        best_fold_loss = float('inf')
        best_fold_model = None
        best_fold_metrics = {}
        train_losses = []
        valid_losses = []
        patience = 5
        no_improvement_count = 0

        for epoch in range(args.epoch):
            train_loss = train_KAN(kan_model, train_loader, optimizer, loss_function, device)
            valid_loss, metrics = validate_KAN(kan_model, valid_loader, loss_function, device)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            if epoch % 20 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}\nMetrics: {metrics}")
            if valid_loss < best_fold_loss:
                best_fold_loss = valid_loss
                best_fold_model = copy.deepcopy(kan_model)
                best_fold_metrics = metrics
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    print(f"Early stopping at epoch {epoch + 1}.")
                    break
        best_metrics_per_fold.append(best_fold_metrics)

    # Average Metrics
    average_metrics = {key: np.mean([m[key] for m in best_metrics_per_fold]) for key in best_metrics_per_fold[0]}
    print("\nAverage Metrics Across All Folds:")
    for metric, value in average_metrics.items():
        print(f"{metric}: {value:.4f}\n")


