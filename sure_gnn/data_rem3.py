import torch
from torch_geometric.datasets import Planetoid
import numpy as np

# Load the Cora dataset
dataset = Planetoid(root='.', name='Cora')

# Extract data
data = dataset[0]

# Use masks to split the data into train and test sets
train_mask = data.train_mask.numpy()
test_mask = data.test_mask.numpy()

# Extract features and labels for training and testing
X_train = data.x[train_mask].numpy()
X_test = data.x[test_mask].numpy()
y_train = data.y[train_mask].numpy()
y_test = data.y[test_mask].numpy()

# Map original node indices to new indices
train_indices = np.where(train_mask)[0]
test_indices = np.where(test_mask)[0]

# Randomly select 2-3 sensitive features
num_sensitive_features = np.random.randint(2, 4)
selected_sensitive_features = np.random.choice(data.x.size(1), num_sensitive_features, replace=False)
X_train_sensitive = X_train[:, selected_sensitive_features]
X_test_sensitive = X_test[:, selected_sensitive_features]

# Adjust edge_index for training set
edge_index_train = data.edge_index.numpy()
mask_train_edges = np.isin(edge_index_train, train_indices).all(axis=0)
edge_index_train = edge_index_train[:, mask_train_edges]

# Map the edge indices to the new node index set
train_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(train_indices)}
edge_index_train = np.vectorize(train_mapping.get)(edge_index_train)

# Adjust edge_index for testing set
edge_index_test = data.edge_index.numpy()
mask_test_edges = np.isin(edge_index_test, test_indices).all(axis=0)
edge_index_test = edge_index_test[:, mask_test_edges]

test_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(test_indices)}
edge_index_test = np.vectorize(test_mapping.get)(edge_index_test)


# Save to npz file
np.savez('cora_data2.npz',
         sensitive_cols={'sensitive_features': [(f'Sensitive_{i+1}', i) for i in range(num_sensitive_features)]},
         X_train=X_train,
         X_test=X_test,
         X_train_sensitive=X_train_sensitive,
         X_test_sensitive=X_test_sensitive,
         edge_index_train=edge_index_train,
         edge_index_test=edge_index_test,
         y_train=y_train,
         y_test=y_test)

loaded_data = np.load('cora_data2.npz', allow_pickle=True)
# print("Keys in the saved file:")
# print(loaded_data.files)
# print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
# print(f"y_train unique classes: {np.unique(y_train)}, counts: {np.bincount(y_train)}")
# print(f"Max edge index in edge_index_train: {edge_index_train.max()}, X_train size: {len(X_train)}")
