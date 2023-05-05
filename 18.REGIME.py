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

print('=' * 20, 'A', '=' * 20)

# Cij
Matrix_C = np.zeros((n * (n - 1), m))
Matrix_Names = np.zeros((n * (n - 1), 1))
print('[CALCULATIONS] Concordance Set')
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
# [print(f'[C{Matrix_Names[i,0]}] {Matrix_C[i]}') for i in range(n*(n-1))]

# Concordance Set * W
print('[CALCULATIONS] Concordance Set * W')
Matrix_C_W = Matrix_C * w
Matrix_C_W_Sum = np.sum(Matrix_C_W, axis=1).reshape(n * (n - 1), 1)
# [print(f'[C{Matrix_Names[i,0]}] {Matrix_C_W[i]} {Matrix_C_W_Sum[i]}') for i in range(n*(n-1))]


# Concordance Matrix
print('[CALCULATIONS] Concordance Matrix')
Matrix_Concordance = np.zeros((n, n))
Matrix_Concordance[:] = np.nan
for idx, name in enumerate(Matrix_Names):
    i, j = (name // 10) - 1, (name % 10) - 1
    Matrix_Concordance[i, j] = Matrix_C_W_Sum[idx]
# print(Matrix_Concordance)

# E Matrix
print('[CALCULATIONS] E Matrix')
Matrix_E = np.zeros((n, n))
Matrix_E[:] = np.nan
log = {}
for i in range(n):
    for j in range(n):
        # don't compare i = j rows (same rows)
        if i == j:
            continue

        if Matrix_Concordance[i, j] > Matrix_Concordance[j, i]:
            Matrix_E[i, j] = 1
            ls = log.setdefault(f'A{i + 1}', [])
            ls.append(f'A{j + 1}')
# print(Matrix_E)

# Ranking A
print('\n[Ranking A]')
[print(f'[{len(dm) - len(v)}]', k, '>', ', '.join(v)) for k, v in log.items()]
RANK = {k: len(dm) - len(v) for k, v in log.items()}
SORTED_RANK = sorted(RANK.items(), key=lambda x: x[1])
print('\n[Sorted Ranking A]')
[print(f'[{i[1]}]: {i[0]} ({ids[int(i[0][1:])-1]})') for i in SORTED_RANK]

print('=' * 20, 'B', '=' * 20)

# Rij
Matrix_R = np.zeros((n * (n - 1), m))
print('[CALCULATIONS] R Matrix')
r_count = 0
for i, row1 in enumerate(Matrix_normalized_w):
    for j, row2 in enumerate(Matrix_normalized_w):

        # don't compare i = j rows (same rows)
        if i == j:
            continue

        for k, sign in enumerate(g):
            if row1[k] > row2[k]:
                Matrix_R[r_count, k] = 1 if sign == '+' else -1
            elif row1[k] < row2[k]:
                Matrix_R[r_count, k] = -1 if sign == '+' else 1
            elif row1[k] == row2[k]:
                Matrix_R[r_count, k] = 0

        r_count += 1

Matrix_R = Matrix_R.astype(int)
# [print(f'[R{Matrix_Names[i,0]}] {Matrix_R[i]}') for i in range(n*(n-1))]

# Rij * W
print('[CALCULATIONS] R*W Matrix')
Matrix_R_W = Matrix_R * w
Matrix_R_W_Sum = np.sum(Matrix_R_W, axis=1).reshape(n * (n - 1), 1)
# [print(f'[R{Matrix_Names[i,0]}] {Matrix_R_W[i]} {Matrix_R_W_Sum[i]}') for i in range(n*(n-1))]

# Ranking
log = {}
for i in range(n):
    for j in range(n):
        # don't compare i = j rows (same rows)
        if i == j:
            continue

        if Matrix_Concordance[i, j] > Matrix_Concordance[j, i]:
            ls = log.setdefault(f'A{i + 1}', [])
            ls.append(f'A{j + 1}')

print('\n[Ranking B]')
[print(f'[{len(dm) - len(v)}]', k, '>', ', '.join(v)) for k, v in log.items()]
RANK = {k: len(dm) - len(v) for k, v in log.items()}
SORTED_RANK = sorted(RANK.items(), key=lambda x: x[1])
print('\n[Sorted Ranking B]')
[print(f'[{i[1]}]: {i[0]} ({ids[int(i[0][1:])-1]})') for i in SORTED_RANK]
