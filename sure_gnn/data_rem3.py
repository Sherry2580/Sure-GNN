import torch
from torch_geometric.datasets import Planetoid
import numpy as np

# Load the Cora dataset
dataset = Planetoid(root='.', name='Cora')


# Extract data
data = dataset[0]

# Randomly select 2-3 sensitive features
num_sensitive_features = np.random.randint(2, 4)
selected_sensitive_features = np.random.choice(data.x.size(1), num_sensitive_features, replace=False)
X_train_sensitive = data.x[:, selected_sensitive_features].numpy()
X_test_sensitive = data.x[:, selected_sensitive_features].numpy()

# Extract other information
X_train = data.x.numpy()
X_test = data.x.numpy()
edge_index = data.edge_index.numpy()
y_train = data.y.numpy()
y_test = data.y.numpy()

# Save to npz file
np.savez('cora_data2.npz',
         sensitive_cols={'sensitive_features': [(f'Sensitive_{i+1}', i) for i in range(num_sensitive_features)]},
         X_train=X_train,
         X_test=X_test,
         X_train_sensitive=X_train_sensitive,
         X_test_sensitive=X_test_sensitive,
         edge_index=edge_index,
         y_train=y_train,
         y_test=y_test)

loaded_data = np.load('cora_data2.npz', allow_pickle=True)
#
# print("sensitive_cols:")
# print(loaded_data['sensitive_cols'].item())
#
# print("\nX_train:")
# print(loaded_data['X_train'])
#
# print("\nX_test:")
# print(loaded_data['X_test'])
#
# print("\nX_train_sensitive:")
# print(loaded_data['X_train_sensitive'])
#
# print("\nX_test_sensitive:")
# print(loaded_data['X_test_sensitive'])
#
# print("\ny_train:")
# print(loaded_data['y_train'])
#
# print("\ny_test:")
# print(loaded_data['y_test'])
