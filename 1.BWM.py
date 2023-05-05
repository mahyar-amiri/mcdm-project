from collections import OrderedDict

import numpy as np
from scipy.optimize import linprog


def calc_weight(b2o, o2w):
    cb = OrderedDict()
    cw = OrderedDict()
    allkeys = sorted(b2o.keys())
    for kk in allkeys:
        cb[kk] = b2o[kk]
        cw[kk] = o2w[kk]
    colSize = np.size(allkeys)
    rowSize = 4 * colSize - 5
    mat = np.zeros((rowSize - 1, colSize + 1), dtype=np.double)
    # print(allkeys)
    # get the best criteria location
    bkey = min(b2o, key=b2o.get)
    bloc = allkeys.index(bkey)
    # get the worst criteria location
    wkey = min(o2w, key=o2w.get)
    wloc = allkeys.index(wkey)
    cb_copy = cb.copy()
    cb_copy.pop(bkey, None)
    tmpmat = np.zeros((len(cb_copy.keys()), colSize + 1), dtype=np.double)
    tmpmat1 = np.zeros((len(cb_copy.keys()), colSize + 1), dtype=np.double)
    for idx in np.arange(len(cb_copy.keys())):
        itmp = allkeys.index(list(cb_copy.keys())[idx])
        tmpmat[idx, bloc] = 1.0
        tmpmat[idx, itmp] = -cb_copy[list(cb_copy.keys())[idx]]
        tmpmat[idx, colSize] = -1.0
    for idx in np.arange(len(cb_copy.keys())):
        itmp = allkeys.index(list(cb_copy.keys())[idx])
        tmpmat1[idx, bloc] = -1.0
        tmpmat1[idx, itmp] = cb_copy[list(cb_copy.keys())[idx]]
        tmpmat1[idx, colSize] = -1.0
    mat[0:2 * colSize - 2, :] = np.concatenate((tmpmat, tmpmat1), axis=0)

    cw_copy = cw.copy()
    cw_copy.pop(bkey, None)
    cw_copy.pop(wkey, None)
    tmpmat = np.zeros((len(cw_copy.keys()), colSize + 1), dtype=np.double)
    tmpmat1 = np.zeros((len(cw_copy.keys()), colSize + 1), dtype=np.double)
    for idx in np.arange(len(cw_copy.keys())):
        # find the location of the key of cw_copy in the keylist list
        itmp = allkeys.index(list(cw_copy.keys())[idx])
        tmpmat[idx, itmp] = 1
        tmpmat[idx, wloc] = -cw_copy[list(cw_copy.keys())[idx]]
        tmpmat[idx, colSize] = -1.0
    for idx in np.arange(len(cw_copy.keys())):
        # find the location of the key of cw_copy in the keylist list
        itmp = allkeys.index(list(cw_copy.keys())[idx])
        tmpmat1[idx, itmp] = -1
        tmpmat1[idx, wloc] = cw_copy[list(cw_copy.keys())[idx]]
        tmpmat1[idx, colSize] = -1.0
    mat[2 * colSize - 2:, :] = np.concatenate((tmpmat, tmpmat1), axis=0)
    Aeq = np.ones((1, colSize + 1), dtype=np.double)
    Aeq[0, -1] = 0.
    beq = np.array([1])
    bub = np.zeros((rowSize - 1), dtype=np.double)
    cc = np.zeros((colSize + 1), dtype=np.double)
    cc[-1] = 1
    res = linprog(cc, A_eq=Aeq, b_eq=beq, A_ub=mat, b_ub=bub, bounds=(0, None), options={'disp': False})
    sol1 = res['x']
    outp = dict()
    ii = 0
    for x in allkeys:
        outp[x] = sol1[ii].item()
        ii += 1
    return outp, sol1[-1].item()


Best2Others = {'Value': 6, 'Passing': 2, 'Finishing': 8, 'Defending': 3, 'Dribbling': 7, 'Pace': 9, 'Physical': 4, 'Goalkeeper': 1}  # Goalkeeper
Others2Worst = {'Value': 6, 'Passing': 8, 'Finishing': 2, 'Defending': 5, 'Dribbling': 5, 'Pace': 1, 'Physical': 8, 'Goalkeeper': 9}  # Pace

weights, zeta = calc_weight(Best2Others, Others2Worst)
[print(f'[{criteria}]: {weight}') for criteria, weight in weights.items()]

print('\nsum(Weights):', np.sum(list(weights.values())))
print('Consistency:', zeta)
print('Is Consistence:', zeta < 0.1)

# [Defending]: 0.14197343068654314
# [Dribbling]: 0.06084575600851823
# [Finishing]: 0.05324003650745308
# [Goalkeeper]: 0.32795862488591443
# [Pace]: 0.02555521752357775
# [Passing]: 0.2129601460298146
# [Physical]: 0.10648007301490725
# [Value]: 0.0709867153432715
