import numpy as np
import pandas as pd

df = pd.read_excel('1.Normalization_Weight.xlsx', sheet_name=['TOP 10', 'Type', 'Weight'])

# TAU Constant Value
tau = 0.02

# Get Data Matrix
dm = df['TOP 10']
ids = dm.pop('ID')
# dm = dm.iloc[:10]  # Maximum 10 GKs are allowed
dm = dm.to_numpy(dtype=np.float64)
# Get Type
g = df['Type'].to_numpy()[0][1:]
# Get Weight
w = df['Weight'].to_numpy(dtype=np.float64)[0][1:]

# Number of Alternatives & Criteria
n, m = dm.shape

# Linear Normalization
print('[CALCULATIONS] Linear Normalized')
Matrix_normalized = np.empty_like(dm)
for i in range(m):
    if g[i] == '+':
        Matrix_normalized[:, i] = dm[:, i] / np.max(dm[:, i])
    elif g[i] == '-':
        Matrix_normalized[:, i] = np.min(dm[:, i]) / dm[:, i]
# print(Matrix_normalized)

# Linear Normalization * W
print('[CALCULATIONS] Linear Normalized * W')
Matrix_normalized_w = Matrix_normalized * w
# print(Matrix_normalized_w)

# ns Matrix
print('[CALCULATIONS] NS')
Matrix_ns = np.min(Matrix_normalized_w, axis=0)
# print(Matrix_ns)

# Euclidean and Taxicab distances from negative-ideal solution
print('[CALCULATIONS] E')
Matrix_E = np.sqrt(np.sum((Matrix_normalized_w - Matrix_ns) ** 2, axis=1))
# print(Matrix_E)

print('[CALCULATIONS] T')
Matrix_T = np.sum(np.abs(Matrix_normalized_w - Matrix_ns), axis=1)
# print(Matrix_T)

print('[CALCULATIONS] Ra')
Matrix_Ra = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        psi = 1 if np.abs(Matrix_E[i] - Matrix_E[j]) >= tau else 0
        Matrix_Ra[i, j] = (Matrix_E[i] - Matrix_E[j]) + (psi * (Matrix_T[i] - Matrix_T[j]))
# print(Matrix_Ra)

print('[CALCULATIONS] H')
Matrix_H = np.sum(Matrix_Ra, axis=1)
# print(Matrix_H)

print('\n[Ranking]')
ranks = sorted(zip(ids, Matrix_H), key=lambda x: x[1], reverse=True)
for i, (j, k) in enumerate(ranks):
    print(f'[{i + 1}]: {j} ({k})')
