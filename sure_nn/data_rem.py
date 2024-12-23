import torch
from torch_geometric.datasets import Planetoid
import numpy as np

# Load the Cora dataset
dataset = Planetoid(root='.', name='Cora')

# Extract data
data = dataset[0]

# Count the number of 1s in each feature
num_ones_per_feature = np.sum(data.x.numpy(), axis=0)
# Select the top 20 features with the most 1s
selected_sensitive_features = np.argsort(num_ones_per_feature)[-20:]

X_train_sensitive = data.x[:, selected_sensitive_features].numpy()
X_test_sensitive = data.x[:, selected_sensitive_features].numpy()

# Extract other information
X_train = data.x[data.train_mask].numpy()
X_test = data.x[data.test_mask].numpy()
y_train = data.y[data.train_mask].numpy()
y_test = data.y[data.test_mask].numpy()

# Save to npz file
np.savez('cora_test.npz',
         sensitive_cols={'sensitive_features': [(f'Sensitive_{i+1}', i) for i in range(20)]},
         X_train=X_train,
         X_test=X_test,
         X_train_sensitive=X_train_sensitive,
         X_test_sensitive=X_test_sensitive,
         # edge_index=edge_index,
         y_train=y_train,
         y_test=y_test)

loaded_data = np.load('cora_test.npz', allow_pickle=True)
