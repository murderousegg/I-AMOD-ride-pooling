import numpy as np
from Utilities.RidePooling.probcomb import probcombN
def calculate_gamma_k2(FullList, DemandS, Delay, N_nodes, WaitingTime, Cumul_delay2, TotGamma2, iii, Demands_rp, idx_map):
    gamma = 0
    if FullList[iii][1] < Delay and FullList[iii][2] < Delay and DemandS[idx_map[int(FullList[iii][4])]][idx_map[int(FullList[iii][3])]] >= 10e-5 and DemandS[idx_map[int(FullList[iii][6])]][idx_map[int(FullList[iii][5])]] >= 10e-5:
        jj1 = idx_map[int(FullList[iii][3])]
        ii1 = idx_map[int(FullList[iii][4])]
        jj2 = idx_map[int(FullList[iii][5])]
        ii2 = idx_map[int(FullList[iii][6])]

        prob = probcombN([DemandS[ii1][jj1],DemandS[ii2][jj2]],WaitingTime)
        gamma = min([DemandS[ii1][jj1], DemandS[ii2][jj2]])*prob/2
        Gamma0 = np.zeros([N_nodes,N_nodes])
        if np.array_equal(FullList[iii][7:11], [1, 2, 1, 2]):
            Gamma0[jj2][jj1] = 1
            Gamma0[ii1][jj2] = 1
            Gamma0[ii2][ii1] = 1
        elif np.array_equal(FullList[iii][7:11], [1, 2, 2, 1]):
            Gamma0[jj2][jj1] = 1
            Gamma0[ii2][jj2] = 1
            Gamma0[ii1][ii2] = 1

        matrow = np.array([[jj1, ii1], [jj2, ii2]])
        # print(matrow)
        multip = np.unique(matrow, axis=0).shape[0]
        # if multip == 1:
        #     print(matrow)
        #     print(jj1)
        #     print(ii1)
        #     print(jj2)
        #     print(ii2)
        #     sdf
        # print(multip)
        Demands_rp = Demands_rp +  multip* gamma* Gamma0
        # print(Demands_rp)
        DemandS[ii1][jj1] -= multip*gamma
        DemandS[ii2][jj2] -= multip*gamma
        Cumul_delay2 += multip*gamma*(FullList[iii, 1] + FullList[iii, 2])
        # print(Cumul_delay2)
        TotGamma2 += multip*gamma
        # print(TotGamma2)

    return Cumul_delay2, TotGamma2, DemandS, Demands_rp, gamma