import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from sklearn.metrics import roc_auc_score
import time
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

import fairness

# class NeuralNetwork(nn.Module):
#   def __init__(self, feature_dim, output_dim, hidden_dims_arr=[64, 32], dropout_p=0.5, no_dropout_first_layer=False):
#     super().__init__()
#     mod_list = []
#     last_output_dim = feature_dim
#     for i in range(len(hidden_dims_arr)):
#       this_output_dim = hidden_dims_arr[i]
#       mod_list.append(nn.Linear(last_output_dim, this_output_dim))
#       mod_list.append(nn.ReLU())
#       if dropout_p > 0 and (i > 0 or not no_dropout_first_layer):
#         mod_list.append(nn.Dropout(dropout_p))
#       last_output_dim = this_output_dim
#     mod_list.append(nn.Linear(last_output_dim, output_dim))
#     self.mod_list = nn.Sequential(*mod_list)
#
#   def forward(self, x):
#     return self.mod_list(x)

class NeuralNetwork(nn.Module):
  def __init__(self, feature_dim,output_dim, hidden_dim=64, dropout_p=0.5):
    super(NeuralNetwork, self).__init__()

      # Define Graph Convolutional layers
    self.conv1 = GCNConv(feature_dim, hidden_dim)
    self.conv2 = GCNConv(hidden_dim, output_dim)

      # Dropout
    self.dropout = nn.Dropout(dropout_p)

  def forward(self, x,edge_index):

    # Apply first Graph Convolutional layer
    x = F.relu(self.conv1(x, edge_index))
    x = self.dropout(x)

    # Apply second Graph Convolutional layer
    x = self.conv2(x, edge_index)

    return F.log_softmax(x, dim=1)



# class MyDataset(Dataset):
#   def __init__(self, X, X_sensitive, y):
#     super().__init__()
#     self.X = X
#     self.X_sensitive = X_sensitive
#     self.y = y
#
#   def __len__(self):
#     return len(self.y)
#
  # def __getitem__(self, idx):
  #   return self.X[idx], self.y[idx], self.X_sensitive[idx], idx
#
#   def get_stats(self):
#     feature_dim = self.X.shape[1]
#     output_dim = len(np.unique(self.y))
#     return feature_dim, output_dim




class MyDataset(Dataset):
    def __init__(self, X, X_sensitive, y, edge_index):
        super().__init__()
        self.X = X
        self.X_sensitive = X_sensitive
        self.y = y
        self.edge_index = edge_index

    def __len__(self):
        return len(self.y)

    # def __getitem__(self, idx):
    #     return {
    #         'x': self.X[idx],
    #         'y': self.y[idx],
    #         'x_sensitive': self.X_sensitive[idx],
    #         'edge_index': self.edge_index,
    #         'idx': idx
    #     }

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.X_sensitive[idx], idx ,self.edge_index

    def get_stats(self):
        feature_dim = self.X.shape[1]
        output_dim = len(np.unique(self.y))
        return feature_dim, output_dim


def mark_clusters(dataset, model, device, num_pts_in_bin=30, ucb_factor=2.0):
  model.eval()
  all_best_dirs = []
  all_stress_idxs = torch.empty(0, dtype=int, device=device)
  all_okc_idxs = torch.empty(0, device=device)

  X, y, X_sensitive, _, edge_index = dataset[:]
  X, y, edge_index = torch.FloatTensor(X).to(device), torch.LongTensor(y).to(device),  torch.LongTensor(edge_index).to(device)

  # Predict
  pred = model(X, edge_index)

  # Find misclassifier points
  misc_mask = (pred.argmax(1) != y)
  if misc_mask.sum() > 0:
    ok_mask = (pred.argmax(1) == y)

    # Find clusters
    start_time_cluster = time.time()
    points_idx, best_dirs, okc_pts_idx, overall_goodness = \
      fairness.cluster_scanner(X[misc_mask],
                               baseline=X[ok_mask],
                               num_pts_in_bin=num_pts_in_bin,
                               ucb_factor=ucb_factor,
                               device=device,
                               )

    # Add extra loss for cluster points
    if not (len(best_dirs) == 0 or overall_goodness < -1):
      stress_ids = points_idx
      okc_ids = okc_pts_idx
      all_stress_idxs = torch.cat((all_stress_idxs, stress_ids))
      all_okc_idxs = torch.cat((all_okc_idxs, okc_ids))

      all_best_dirs.extend(best_dirs)

  all_stress_idxs = all_stress_idxs.cpu().numpy()
  all_okc_idxs = all_okc_idxs.cpu().numpy()
  return all_stress_idxs, all_okc_idxs


def train(model, optimizer, device, wt_vec, X, y, edge_index, loss_fn):
  model.train()

  X, y, edge_index = torch.FloatTensor(X).to(device), torch.LongTensor(y).to(device),  torch.LongTensor(edge_index).to(device)

  # Predict
  pred = model(X, edge_index)

  # Standard loss
  overall_loss = loss_fn(pred, y)

  this_wt_vec = wt_vec
  loss = (this_wt_vec * overall_loss).sum()

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()


def test(model, device, sensitive_cols, X, y, X_sensitive, edge_index, loss_fn, verbose):
  model.eval()
  size = len(y)
  correct = 0
  all_res = []

  X, y, edge_index =  torch.FloatTensor(X).to(device), torch.LongTensor(y).to(device),  torch.LongTensor(edge_index).to(device)

  # Predict
  pred = model(X, edge_index)

  # Create a DataFrame to store results
  # this_res = DataFrame(X_sensitive, columns=list(sensitive_cols.keys()))
  this_res = pd.DataFrame(X_sensitive, columns=[f"Sensitive_{i + 1}" for i in range(X_sensitive.shape[1])])
  this_res['y'] = y

  # Calculate loss and accuracy
  # this_res['loss'] = loss_fn(pred, y).cpu().numpy()
  this_res['loss'] = loss_fn(pred, y).detach().cpu().numpy()
  this_res['correct'] = (pred.argmax(1) == y).type(torch.float).cpu().numpy()
  this_res['score'] = pred[:, 1].type(torch.float).detach().cpu().numpy()



  correct += this_res['correct'].sum()
  all_res.append(this_res)

  correct /= size
  all_res = pd.concat(all_res, ignore_index=True)

  if verbose > -3:
    print(f" Accuracy: {(100 * correct):>0.1f}%, Avg loss: {this_res['loss'].mean():3.3f}")

  result_stats = DataFrame()
  attr_series = {}
  for s, l in sensitive_cols.items():
    for sval, _ in l:
      attr_series[sval] = s
      tmp = all_res[all_res[sval] == 1]

      try:
        this_d = {'count': len(tmp),
                  'accuracy': '{:2.2f}'.format(tmp['correct'].mean()),
                  'macro-auc': '{:2.2f}'.format(roc_auc_score(tmp['y'], tmp['score'], average='macro')),
                  'loss2': '{:3.3f}'.format(tmp['loss'].mean()),
                  }
      except ValueError:
        auc_score = -1
        this_d = {'count': len(tmp),
                  'accuracy': '{:2.2f}'.format(tmp['correct'].mean()),
                  'macro-auc': '{:2.2f}'.format(auc_score),
                  'loss2': '{:3.3f}'.format(tmp['loss'].mean()),
                  }

      result_stats[sval] = Series(this_d)
  attr_series = Series(attr_series)
  return result_stats, attr_series


def do_all(training_data, testing_data, sensitive_cols, num_pts_in_bin=30, cluster_loss_multiplier=1.0,
           min_wt_factor=1.0, batch_size=2708, cluster_batch_size=2708, epochs=10, skip_epochs=3, cluster_epochs=30,
           seed=0, verbose=0, dropout_p=0, lr=1e-2,
           train_error_check_epochs=[140, 200, 300, 400, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]):
  np.random.seed(seed)
  torch.manual_seed(seed)

  if cluster_batch_size < 0:
    cluster_batch_size = len(training_data)

  shuffle = True

  feature_dim, output_dim = training_data.get_stats()
  loss_fn = nn.CrossEntropyLoss(reduction='none')
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = NeuralNetwork(feature_dim=feature_dim, output_dim=output_dim, hidden_dim=64, dropout_p=dropout_p).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  wt_vec = (torch.ones(len(training_data)) / len(training_data)).to(device)
  all_stress_idxs, all_okc_idxs = [], []
  all_res_train = []

  for i in range(epochs):
    if verbose >= 1:
      print(f'Iter {i}')

    if (i >= skip_epochs) and (i - skip_epochs) % cluster_epochs == 0:
      # Find stress points
      all_stress_idxs, all_okc_idxs = \
        mark_clusters(training_data, model=model, device=device, num_pts_in_bin=num_pts_in_bin)

      # Update weights
      wt_vec2 = torch.ones(len(training_data)).to(device)
      wt_vec2[all_stress_idxs] += min_wt_factor
      wt_vec2[all_okc_idxs] += min_wt_factor
      wt_vec2 /= wt_vec2.sum()

      wt_vec = wt_vec / (1 + cluster_loss_multiplier) + (1 - 1 / (1 + cluster_loss_multiplier)) * wt_vec2

      # Set up the training dataloader so that enough stress+okc points are there in each mini-batch
      num_focus_pts = len(all_stress_idxs) + len(all_okc_idxs)
      effective_size = num_focus_pts
      this_batch_size = int(
        np.ceil(batch_size * len(training_data) / effective_size)) if num_focus_pts > 0 else batch_size

      # Training without DataLoader
      train(model=model, optimizer=optimizer, device=device, wt_vec=wt_vec,
            X=training_data.X, y=training_data.y, edge_index=training_data.edge_index, loss_fn=loss_fn)

    # Training without DataLoader
    train(model=model, optimizer=optimizer, device=device, wt_vec=wt_vec,
          X=training_data.X, y=training_data.y, edge_index=training_data.edge_index, loss_fn=loss_fn)

    if verbose >= 1 or i == epochs - 1 or ((i + 1) in train_error_check_epochs):
      res_train, attr_series = test(model=model, device=device, sensitive_cols=sensitive_cols,
                                    X=training_data.X, y=training_data.y, X_sensitive=training_data.X_sensitive,
                                    edge_index=training_data.edge_index, loss_fn=loss_fn, verbose=verbose)
      res_train['epoch'] = i + 1
      if verbose >= 0 or (i == epochs - 1 and verbose > -2):
        print('Train error:')
        print(res_train)
      all_res_train.append(res_train)

  all_res_train = pd.concat(all_res_train, ignore_index=False)
  if testing_data is not None:
    res_test, attr_series = test(model=model, device=device, sensitive_cols=sensitive_cols,
                                 X=testing_data.X, y=testing_data.y, X_sensitive=testing_data.X_sensitive,
                                 edge_index=testing_data.edge_index, loss_fn=loss_fn, verbose=verbose)
    res_test['epoch'] = epochs
    if verbose >= 1 or (i == epochs - 1 and verbose > -2):
      print('Test error:')
      print(res_test)

  return all_res_train, res_test, attr_series
