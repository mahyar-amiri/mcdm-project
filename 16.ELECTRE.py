import numpy as np
import pandas as pd

df = pd.read_excel('1.Normalization_Weight.xlsx', sheet_name=['TOP 10', 'Type', 'Weight'])

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

# DM ^ 2
Matrix_dm_2 = dm ** 2

# Vector Normalization
print('[CALCULATIONS] Vector Normalized')
Matrix_normalized = dm / (np.sum(Matrix_dm_2, axis=0) ** 0.5)
# print(Matrix_normalized)

# Vector Normalization * W
print('[CALCULATIONS] Vector Normalized * W')
Matrix_normalized_w = Matrix_normalized * w
# print(Matrix_normalized_w)

# Cij
Matrix_C = np.zeros((n * (n - 1), m))
Matrix_Names = np.zeros((n * (n - 1), 1))
print('[CALCULATIONS] Concordance Set]')
r_count = 0
for i, row1 in enumerate(Matrix_normalized_w):
    for j, row2 in enumerate(Matrix_normalized_w):

        # don't compare i = j rows (same rows)
        if i == j:
            continue

        Matrix_Names[r_count] = 10 * (i + 1) + (j + 1)

        for k, sign in enumerate(g):
            if sign == '+':
                Matrix_C[r_count, k] = 1 if row1[k] >= row2[k] else 0
            else:
                Matrix_C[r_count, k] = 1 if row1[k] <= row2[k] else 0
        r_count += 1

Matrix_C = Matrix_C.astype(int)
Matrix_Names = Matrix_Names.astype(int)
# [print(f'[C{Matrix_Names[i, 0]}] {Matrix_C[i]}') for i in range(n * (n - 1))]

# Concordance Set * W
print('[CALCULATIONS] Concordance Set * W')
Matrix_C_W = Matrix_C * w
Matrix_C_W_Sum = np.sum(Matrix_C_W, axis=1).reshape(n * (n - 1), 1)
c_avg = np.average(Matrix_C_W_Sum)
# [print(f'[C{Matrix_Names[i, 0]}] {Matrix_C_W[i]} {Matrix_C_W_Sum[i]}') for i in range(n * (n - 1))]

# Concordance Matrix
print('[CALCULATIONS] Concordance Matrix')
Matrix_Concordance = np.zeros((n, n))
Matrix_Concordance[:] = np.nan
for idx, name in enumerate(Matrix_Names):
    i, j = (name // 10) - 1, (name % 10) - 1
    Matrix_Concordance[i, j] = Matrix_C_W_Sum[idx]
# print(f'Average({c_avg})\n{Matrix_Concordance}')

# F Matrix
print('[CALCULATIONS] F Matrix')
Matrix_F = (Matrix_Concordance >= c_avg).astype(int)
# print(Matrix_F)

# delta
Matrix_delta = np.zeros((n * (n - 1), m))
print('[CALCULATIONS] Delta Set]')
r_count = 0
for i, row1 in enumerate(Matrix_normalized_w):
    for j, row2 in enumerate(Matrix_normalized_w):

        # don't compare i = j rows (same rows)
        if i == j:
            continue

        Matrix_delta[r_count] = np.abs(row1 - row2)
        r_count += 1
# [print(f'[Δ{Matrix_Names[i, 0]}] {Matrix_delta[i]}') for i in range(n * (n - 1))]

# dij
Matrix_d = np.zeros((n * (n - 1), m))
print('[CALCULATIONS] Discordance Set')
r_count = 0
for i, row1 in enumerate(Matrix_normalized_w):
    for j, row2 in enumerate(Matrix_normalized_w):

        # don't compare i = j rows (same rows)
        if i == j:
            continue

        for k, sign in enumerate(g):
            if sign == '+':
                Matrix_d[r_count, k] = 1 if row1[k] < row2[k] else 0
            else:
                Matrix_d[r_count, k] = 1 if row1[k] > row2[k] else 0
        r_count += 1

Matrix_d = Matrix_d.astype(int)
# [print(f'[d{Matrix_Names[i, 0]}] {Matrix_d[i]}') for i in range(n * (n - 1))]

# Dij
print('[CALCULATIONS] Discordance Set')
Matrix_D = Matrix_d * Matrix_delta
# [print(f'[D{Matrix_Names[i, 0]}] {Matrix_D[i]}') for i in range(n * (n - 1))]

# Discordance Matrix
print('[CALCULATIONS] Discordance Matrix')
Matrix_Discordance = np.zeros((n, n))
Matrix_Discordance[:] = np.nan
for idx, name in enumerate(Matrix_Names):
    i, j = (name // 10) - 1, (name % 10) - 1
    Matrix_Discordance[i, j] = np.max(
        Matrix_D[idx]) / np.max(Matrix_delta[idx])
d_avg = np.nanmean(Matrix_Discordance)
# print(f'Average({d_avg})\n{Matrix_Discordance}')

# G Matrix
print('[CALCULATIONS] G Matrix')
Matrix_G = (Matrix_Discordance <= d_avg).astype(int)
# print(Matrix_G)

# E Matrix
print('[CALCULATIONS] E Matrix')
Matrix_E = (Matrix_F * Matrix_G).astype(int)
# print(Matrix_E)

# Ranking
print('\n[Ranking]')
RANK = {ids[i]: np.sum(Matrix_E, axis=1)[i] - np.sum(Matrix_E, axis=0)[i] for i in range(n)}
SORTED_RANK = sorted(RANK.items(), key=lambda x: x[1], reverse=True)
[print(f'[{i[0]}]: {i[1]}') for i in SORTED_RANK]
