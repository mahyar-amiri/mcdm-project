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

# Feature Scaling Normalization
print('[CALCULATIONS] Feature Scaling Normalized')
Matrix_normalized = np.empty_like(dm)
for i in range(m):
    if g[i] == '+':
        Matrix_normalized[:, i] = (dm[:, i] - np.min(dm[:, i])) / (np.max(dm[:, i]) - np.min(dm[:, i]))
    elif g[i] == '-':
        Matrix_normalized[:, i] = (dm[:, i] - np.max(dm[:, i])) / (np.min(dm[:, i]) - np.max(dm[:, i]))
# print(Matrix_normalized)

Matrix_I = np.zeros(n)
Matrix_O = np.zeros(n)
for j in range(m):
    if g[j] == '-':
        Matrix_I += w[j] * Matrix_normalized[:, j]
    else:
        Matrix_O += w[j] * Matrix_normalized[:, j]

# Calculate linear preference ratings for cost and profit criteria
print('[CALCULATIONS] I')
Matrix_I -= np.min(Matrix_I)
# print(Matrix_I)
print('[CALCULATIONS] O')
Matrix_O -= np.min(Matrix_O)
# print(Matrix_O)

# Calculate overall preference rating
print('[CALCULATIONS] P')
Matrix_P = (Matrix_I + Matrix_O) - np.min(Matrix_I + Matrix_O)
# print(Matrix_P)

print('\n[Ranking]')
ranks = sorted(zip(ids, Matrix_P), key=lambda x: x[1], reverse=True)
for i, (j, k) in enumerate(ranks):
    print(f'[{i + 1}]: {j} ({k})')
