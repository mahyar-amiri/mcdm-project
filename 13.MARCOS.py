import numpy as np
import pandas as pd

df = pd.read_excel('1.Normalization_Weight.xlsx', sheet_name=['Players', 'Type', 'Weight'])

# Get Data Matrix
dm = df['Players']
ids = dm.pop('ID')
# dm = dm.iloc[:10]  # Maximum 10 GKs are allowed
dm = dm.to_numpy(dtype=np.float64)
# Get Type
g = df['Type'].to_numpy()[0][1:]
# Get Weight
w = df['Weight'].to_numpy(dtype=np.float64)[0][1:]

# Number of Alternatives & Criteria
n, m = dm.shape

# Extended initial decision matrix
print('[CALCULATIONS] Extended Initial DM')
Matrix_extended = np.zeros((n + 2, m))
Matrix_extended[:-2] = dm

max_values, min_values = np.max(dm, axis=0), np.min(dm, axis=0)

for i in range(m):
    if g[i] == '+':
        Matrix_extended[-2, i] = max_values[i]
        Matrix_extended[-1, i] = min_values[i]
    else:
        Matrix_extended[-2, i] = min_values[i]
        Matrix_extended[-1, i] = max_values[i]
# print(Matrix_extended)

# Linear Normalization
print('[CALCULATIONS] Linear Normalized')
Matrix_normalized = np.empty_like(Matrix_extended)
for i in range(m):
    if g[i] == '+':
        Matrix_normalized[:, i] = Matrix_extended[:, i] / np.max(Matrix_extended[:, i])
    elif g[i] == '-':
        Matrix_normalized[:, i] = np.min(Matrix_extended[:, i]) / Matrix_extended[:, i]
# print(Matrix_normalized)

# Linear Normalization * W
print('[CALCULATIONS] Linear Normalized * W')
Matrix_normalized_w = Matrix_normalized * w
# print(Matrix_normalized_w)

# Utility degree
print('[CALCULATIONS] S')
S = Matrix_normalized_w.sum(axis=1)
print('[CALCULATIONS] K-')
k_neg = (S / S[-1])[:-2]
print('[CALCULATIONS] K+')
k_pos = (S / S[-2])[:-2]

# Utility functions
print('[CALCULATIONS] f(K+)')
f_k_pos = k_neg / (k_pos + k_neg)
print('[CALCULATIONS] f(K-)')
f_k_neg = k_pos / (k_pos + k_neg)
print('[CALCULATIONS] f(K)')
f_k = (k_pos + k_neg) / (1 + (1 - f_k_pos) / f_k_pos + (1 - f_k_neg) / f_k_neg)
# print(f_k)

print('\n[Ranking]')
ranks = sorted(zip(ids, f_k), key=lambda x: x[1], reverse=True)
for i, (j, k) in enumerate(ranks):
    print(f'[{i + 1}]: {j} ({k})')
