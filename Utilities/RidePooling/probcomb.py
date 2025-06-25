import numpy as np
def probcombN(a, waiting):
    waiting = waiting
    n = len(a)
    prob = 0
    if np.any(a==0):
        print('prob0')
    for ii in range(0,n):
        a_temp = np.array(a)
        a_temp = np.delete(a_temp, ii)
        prob = prob +(a[ii]/sum(a)) * np.prod(1 - np.exp(-a_temp*waiting))
    if np.isnan(prob):
        prob = 0
    if prob < 10e-10:
        prob = 0
    return prob
