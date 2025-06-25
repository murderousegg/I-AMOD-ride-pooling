import numpy as np
from Utilities.RidePooling.probcomb import probcombN
def calculate_gamma(FullList, DemandS, Delay, N_nodes, WaitingTime, Cumul_delay2, TotGamma2, Cumul_delay3, TotGamma3, Cumul_delay4, TotGamma4, iii, Demands_rp, idx_map):
    gamma = 0
    if np.isnan(FullList[iii][3]) and np.isnan(FullList[iii][4]) and FullList[iii][1] < Delay and FullList[iii][2] < Delay and DemandS[idx_map[int(FullList[iii][6])]][idx_map[int(FullList[iii][5])]] >= 10e-5 and DemandS[idx_map[int(FullList[iii][8])]][idx_map[int(FullList[iii][7])]] >= 10e-5:
        jj1 = idx_map[int(FullList[iii][5])]
        ii1 = idx_map[int(FullList[iii][6])]
        jj2 = idx_map[int(FullList[iii][7])]
        ii2 = idx_map[int(FullList[iii][8])]

        prob = probcombN([DemandS[ii1][jj1],DemandS[ii2][jj2]],WaitingTime)
        gamma = min([DemandS[ii1][jj1], DemandS[ii2][jj2]])*prob/2
        Gamma0 = np.zeros([N_nodes,N_nodes])
        if np.array_equal(FullList[iii][13:17], [1, 2, 1, 2]):
            Gamma0[jj2][jj1] = 1
            Gamma0[ii1][jj2] = 1
            Gamma0[ii2][ii1] = 1
        elif np.array_equal(FullList[iii][13:17], [1, 2, 2, 1]):
            Gamma0[jj2][jj1] = 1
            Gamma0[ii2][jj2] = 1
            Gamma0[ii1][ii2] = 1

        matrow = np.array([[jj1, ii1], [jj2, ii2]])
        # print(matrow)
        multip = np.unique(matrow, axis=0).shape[0]
        # print(multip)
        Demands_rp = Demands_rp +  multip* gamma* Gamma0
        # print(Demands_rp)
        DemandS[ii1][jj1] -= multip*gamma
        DemandS[ii2][jj2] -= multip*gamma
        Cumul_delay2 += multip*gamma*(FullList[iii, 1] + FullList[iii, 2])
        # print(Cumul_delay2)
        TotGamma2 += multip*gamma
        # print(TotGamma2)
    
    elif ~np.isnan(FullList[iii][3]) and np.isnan(FullList[iii][4]) and FullList[iii][3] < Delay and FullList[iii][1] < Delay and FullList[iii][2] < Delay and DemandS[idx_map[int(FullList[iii][6])]][idx_map[int(FullList[iii][5])]] >= 10e-5 and DemandS[idx_map[int(FullList[iii][8])]][idx_map[int(FullList[iii][7])]] >= 10e-5 and DemandS[idx_map[int(FullList[iii][10])]][idx_map[int(FullList[iii][9])]]>= 10e-5:
        jj1 = idx_map[int(FullList[iii][5])]
        ii1 = idx_map[int(FullList[iii][6])]
        jj2 = idx_map[int(FullList[iii][7])]
        ii2 = idx_map[int(FullList[iii][8])]
        jj3 = idx_map[int(FullList[iii][9])]
        ii3 = idx_map[int(FullList[iii][10])]

        gamma = min([DemandS[ii1][jj1],DemandS[ii2][jj2],DemandS[ii3][jj3]])*probcombN([DemandS[ii1][jj1],DemandS[ii2][jj2],DemandS[ii3][jj3]],WaitingTime)/3
        Gamma0 = np.zeros([N_nodes,N_nodes])
        
        if np.array_equal(FullList[iii, 13:19], [1, 2, 3, 1, 2, 3]):
            Gamma0[jj2, jj1] = 1
            Gamma0[jj3, jj2] = 1
            Gamma0[ii1, jj3] = 1
            Gamma0[ii2, ii1] = 1
            Gamma0[ii3, ii2] = 1
        elif np.array_equal(FullList[iii, 13:19], [1, 2, 3, 1, 3, 2]):
            Gamma0[jj2, jj1] = 1
            Gamma0[jj3, jj2] = 1
            Gamma0[ii1, jj3] = 1
            Gamma0[ii3, ii1] = 1
            Gamma0[ii2, ii3] = 1
        elif np.array_equal(FullList[iii, 13:19], [1, 2, 3, 2, 1, 3]):
            Gamma0[jj2, jj1] = 1
            Gamma0[jj3, jj2] = 1
            Gamma0[ii2, jj3] = 1
            Gamma0[ii1, ii2] = 1
            Gamma0[ii3, ii1] = 1
        elif np.array_equal(FullList[iii, 13:19], [1, 2, 3, 2, 3, 1]):
            Gamma0[jj2, jj1] = 1
            Gamma0[jj3, jj2] = 1
            Gamma0[ii2, jj3] = 1
            Gamma0[ii3, ii2] = 1
            Gamma0[ii1, ii3] = 1
        elif np.array_equal(FullList[iii, 13:19], [1, 2, 3, 3, 1, 2]):
            Gamma0[jj2, jj1] = 1
            Gamma0[jj3, jj2] = 1
            Gamma0[ii3, jj3] = 1
            Gamma0[ii1, ii3] = 1
            Gamma0[ii2, ii1] = 1
        elif np.array_equal(FullList[iii, 13:19], [1, 2, 3, 3, 2, 1]):
            Gamma0[jj2, jj1] = 1
            Gamma0[jj3, jj2] = 1
            Gamma0[ii3, jj3] = 1
            Gamma0[ii2, ii3] = 1
            Gamma0[ii1, ii2] = 1
        matrow = np.array([[jj1, ii1], [jj2, ii2], [jj3, ii3]])
        multip = np.unique(matrow, axis=0).shape[0]

        Demands_rp += multip * gamma * Gamma0

        DemandS[ii1, jj1] -= multip * gamma
        DemandS[ii2, jj2] -= multip * gamma
        DemandS[ii3, jj3] -= multip * gamma

        Cumul_delay3 += multip * gamma * (FullList[iii, 1] + FullList[iii, 2] + FullList[iii, 3])

        TotGamma3 += gamma    
    


    elif (FullList[iii, 1] < Delay and FullList[iii, 2] < Delay and FullList[iii, 3] < Delay and FullList[iii, 4] < Delay and
        DemandS[int(FullList[iii, 6]), int(FullList[iii, 5])] >= 1e-5 and DemandS[int(FullList[iii, 8]), int(FullList[iii, 7])] >= 1e-5 and
        DemandS[int(FullList[iii, 10]), int(FullList[iii, 9])] >= 1e-5 and DemandS[int(FullList[iii, 12]), int(FullList[iii, 11])] >= 1e-5):
        
        jj1 = int(FullList[iii][5])
        ii1 = int(FullList[iii][6])
        jj2 = int(FullList[iii][7])
        ii2 = int(FullList[iii][8])
        jj3 = int(FullList[iii][9])
        ii3 = int(FullList[iii][10])
        jj4 = int(FullList[iii][11])
        ii4 = int(FullList[iii][12])
        
        gamma = min([DemandS[ii1, jj1], DemandS[ii2, jj2], DemandS[ii3, jj3], DemandS[ii4, jj4]]) * \
                probcombN([DemandS[ii1, jj1], DemandS[ii2, jj2], DemandS[ii3, jj3], DemandS[ii4, jj4]], WaitingTime) / 4
        Gamma0 = np.zeros([N_nodes, N_nodes])  # Assuming N_nodes is defined somewhere in the code
        
        if np.array_equal(FullList[iii,13:21], [1, 2, 3, 4, 1, 2, 3, 4]):
            Gamma0[jj2,jj1] = 1
            Gamma0[jj3,jj2] = 1
            Gamma0[jj4,jj3] = 1
            Gamma0[ii1,jj4] = 1
            Gamma0[ii2,ii1] = 1
            Gamma0[ii3,ii2] = 1
            Gamma0[ii4,ii3] = 1
        elif np.array_equal(FullList[iii,13:21], [1, 2, 3, 4, 1, 3, 2, 4]):
            Gamma0[jj2,jj1] = 1
            Gamma0[jj3,jj2] = 1
            Gamma0[jj4,jj3] = 1
            Gamma0[ii1,jj4] = 1
            Gamma0[ii3,ii1] = 1
            Gamma0[ii2,ii3] = 1
            Gamma0[ii4,ii2] = 1
        elif np.array_equal(FullList[iii,13:21], [1, 2, 3, 4, 1, 3, 4, 2]):
            Gamma0[jj2,jj1] = 1
            Gamma0[jj3,jj2] = 1
            Gamma0[jj4,jj3] = 1
            Gamma0[ii1,jj4] = 1
            Gamma0[ii3,ii1] = 1
            Gamma0[ii4,ii3] = 1
            Gamma0[ii2,ii4] = 1
        elif np.array_equal(FullList[iii,13:21], [1, 2, 3, 4, 1, 2, 4, 3]):
            Gamma0[jj2,jj1] = 1
            Gamma0[jj3,jj2] = 1
            Gamma0[jj4,jj3] = 1
            Gamma0[ii1,jj4] = 1
            Gamma0[ii2,ii1] = 1
            Gamma0[ii4,ii2] = 1
            Gamma0[ii3,ii4] = 1
        elif np.array_equal(FullList[iii,13:21], [1, 2, 3, 4, 1, 4, 2, 3]):
            Gamma0[jj2,jj1] = 1
            Gamma0[jj3,jj2] = 1
            Gamma0[jj4,jj3] = 1
            Gamma0[ii1,jj4] = 1
            Gamma0[ii4,ii1] = 1
            Gamma0[ii2,ii4] = 1
            Gamma0[ii3,ii2] = 1
        elif np.array_equal(FullList[iii,13:21], [1, 2, 3, 4, 1, 4, 3, 2]):
            Gamma0[jj2,jj1] = 1
            Gamma0[jj3,jj2] = 1
            Gamma0[jj4,jj3] = 1
            Gamma0[ii1,jj4] = 1
            Gamma0[ii4,ii1] = 1
            Gamma0[ii3,ii4] = 1
            Gamma0[ii2,ii3] = 1
        elif np.array_equal(FullList[iii,13:21], [1, 2, 3, 4, 2, 1, 3, 4]):
            Gamma0[jj2,jj1] = 1
            Gamma0[jj3,jj2] = 1
            Gamma0[jj4,jj3] = 1
            Gamma0[ii2,jj4] = 1
            Gamma0[ii1,ii2] = 1
            Gamma0[ii3,ii1] = 1
            Gamma0[ii4,ii3] = 1
        elif np.array_equal(FullList[iii,13:21], [1, 2, 3, 4, 2, 3, 1, 4]):
            Gamma0[jj2,jj1] = 1
            Gamma0[jj3,jj2] = 1
            Gamma0[jj4,jj3] = 1
            Gamma0[ii2,jj4] = 1
            Gamma0[ii3,ii2] = 1
            Gamma0[ii1,ii3] = 1
            Gamma0[ii4,ii1] = 1
        elif np.array_equal(FullList[iii,13:21], [1, 2, 3, 4, 2, 1, 4, 3]):
            Gamma0[jj2,jj1] = 1
            Gamma0[jj3,jj2] = 1
            Gamma0[jj4,jj3] = 1
            Gamma0[ii2,jj4] = 1
            Gamma0[ii1,ii2] = 1
            Gamma0[ii4,ii1] = 1
            Gamma0[ii3,ii4] = 1
        elif np.array_equal(FullList[iii,13:21], [1, 2, 3, 4, 2, 3, 4, 1]):
            Gamma0[jj2,jj1] = 1
            Gamma0[jj3,jj2] = 1
            Gamma0[jj4,jj3] = 1
            Gamma0[ii2,jj4] = 1
            Gamma0[ii3,ii2] = 1
            Gamma0[ii4,ii3] = 1
            Gamma0[ii1,ii4] = 1
        elif np.array_equal(FullList[iii,13:21], [1, 2, 3, 4, 2, 4, 3, 1]):
            Gamma0[jj2,jj1] = 1
            Gamma0[jj3,jj2] = 1
            Gamma0[jj4,jj3] = 1
            Gamma0[ii2,jj4] = 1
            Gamma0[ii4,ii2] = 1
            Gamma0[ii3,ii4] = 1
            Gamma0[ii1,ii3] = 1
        elif np.array_equal(FullList[iii,13:21], [1, 2, 3, 4, 2, 4, 1, 3]):
            Gamma0[jj2,jj1] = 1
            Gamma0[jj3,jj2] = 1
            Gamma0[jj4,jj3] = 1
            Gamma0[ii2,jj4] = 1
            Gamma0[ii4,ii2] = 1
            Gamma0[ii1,ii4] = 1
            Gamma0[ii3,ii1] = 1
        elif np.array_equal(FullList[iii,13:21], [1, 2, 3, 4, 3, 1, 2, 4]):
            Gamma0[jj2,jj1] = 1
            Gamma0[jj3,jj2] = 1
            Gamma0[jj4,jj3] = 1
            Gamma0[ii3,jj4] = 1
            Gamma0[ii1,ii3] = 1
            Gamma0[ii2,ii1] = 1
            Gamma0[ii4,ii2] = 1
        elif np.array_equal(FullList[iii,13:21], [1, 2, 3, 4, 3, 1, 4, 2]):
            Gamma0[jj2,jj1] = 1
            Gamma0[jj3,jj2] = 1
            Gamma0[jj4,jj3] = 1
            Gamma0[ii3,jj4] = 1
            Gamma0[ii1,ii3] = 1
            Gamma0[ii4,ii1] = 1
            Gamma0[ii2,ii4] = 1
        elif np.array_equal(FullList[iii,13:21], [1, 2, 3, 4, 3, 2, 1, 4]):
            Gamma0[jj2,jj1] = 1
            Gamma0[jj3,jj2] = 1
            Gamma0[jj4,jj3] = 1
            Gamma0[ii3,jj4] = 1
            Gamma0[ii2,ii3] = 1
            Gamma0[ii1,ii2] = 1
            Gamma0[ii4,ii1] = 1
        elif np.array_equal(FullList[iii,13:21], [1, 2, 3, 4, 3, 2, 4, 1]):
            Gamma0[jj2,jj1] = 1
            Gamma0[jj3,jj2] = 1
            Gamma0[jj4,jj3] = 1
            Gamma0[ii3,jj4] = 1
            Gamma0[ii2,ii3] = 1
            Gamma0[ii4,ii2] = 1
            Gamma0[ii1,ii4] = 1
        elif np.array_equal(FullList[iii,13:21], [1, 2, 3, 4, 3, 4, 1, 2]):
            Gamma0[jj2,jj1] = 1
            Gamma0[jj3,jj2] = 1
            Gamma0[jj4,jj3] = 1
            Gamma0[ii3,jj4] = 1
            Gamma0[ii4,ii3] = 1
            Gamma0[ii1,ii4] = 1
            Gamma0[ii2,ii1] = 1
        elif np.array_equal(FullList[iii,13:21], [1, 2, 3, 4, 3, 4, 2, 1]):
            Gamma0[jj2,jj1] = 1
            Gamma0[jj3,jj2] = 1
            Gamma0[jj4,jj3] = 1
            Gamma0[ii3,jj4] = 1
            Gamma0[ii4,ii3] = 1
            Gamma0[ii2,ii4] = 1
            Gamma0[ii1,ii2] = 1
        elif np.array_equal(FullList[iii,13:21], [1, 2, 3, 4, 4, 3, 2, 1]):
            Gamma0[jj2,jj1] = 1
            Gamma0[jj3,jj2] = 1
            Gamma0[jj4,jj3] = 1
            Gamma0[ii4,jj4] = 1
            Gamma0[ii3,ii4] = 1
            Gamma0[ii2,ii3] = 1
            Gamma0[ii1,ii2] = 1
        elif np.array_equal(FullList[iii,13:21], [1, 2, 3, 4, 4, 3, 1, 2]):
            Gamma0[jj2,jj1] = 1
            Gamma0[jj3,jj2] = 1
            Gamma0[jj4,jj3] = 1
            Gamma0[ii4,jj4] = 1
            Gamma0[ii3,ii4] = 1
            Gamma0[ii1,ii3] = 1
            Gamma0[ii2,ii1] = 1
        elif np.array_equal(FullList[iii,13:21], [1, 2, 3, 4, 4, 2, 1, 3]):
            Gamma0[jj2,jj1] = 1
            Gamma0[jj3,jj2] = 1
            Gamma0[jj4,jj3] = 1
            Gamma0[ii4,jj4] = 1
            Gamma0[ii2,ii4] = 1
            Gamma0[ii1,ii2] = 1
            Gamma0[ii3,ii1] = 1
        elif np.array_equal(FullList[iii,13:21], [1, 2, 3, 4, 4, 2, 3, 1]):
            Gamma0[jj2,jj1] = 1
            Gamma0[jj3,jj2] = 1
            Gamma0[jj4,jj3] = 1
            Gamma0[ii4,jj4] = 1
            Gamma0[ii2,ii4] = 1
            Gamma0[ii3,ii2] = 1
            Gamma0[ii1,ii3] = 1
        elif np.array_equal(FullList[iii,13:21], [1, 2, 3, 4, 4, 1, 3, 2]):
            Gamma0[jj2,jj1] = 1
            Gamma0[jj3,jj2] = 1
            Gamma0[jj4,jj3] = 1
            Gamma0[ii4,jj4] = 1
            Gamma0[ii1,ii4] = 1
            Gamma0[ii3,ii1] = 1
            Gamma0[ii2,ii3] = 1
        elif np.array_equal(FullList[iii,13:21], [1, 2, 3, 4, 4, 1, 2, 3]):
            Gamma0[jj2,jj1] = 1
            Gamma0[jj3,jj2] = 1
            Gamma0[jj4,jj3] = 1
            Gamma0[ii4,jj4] = 1
            Gamma0[ii1,ii4] = 1
            Gamma0[ii2,ii1] = 1
            Gamma0[ii3,ii2] = 1

        # Create matrow as a list of rows
        matrow = np.array([[jj1, ii1], [jj2, ii2], [jj3, ii3], [jj4, ii4]])

        # Calculate the unique rows and the number of unique rows (multip)
        unique_rows = np.unique(matrow, axis=0)
        multip = unique_rows.shape[0]

        # Update Demands_rp by adding multip * gamma * Gamma0
        Demands_rp = Demands_rp + multip * gamma * Gamma0

        # Update DemandS for the relevant indices
        DemandS[ii1, jj1] -= multip * gamma
        DemandS[ii2, jj2] -= multip * gamma
        DemandS[ii3, jj3] -= multip * gamma
        DemandS[ii4, jj4] -= multip * gamma

        # Update Cumul_delay4 using FullList
        Cumul_delay4 += multip * gamma * (FullList[iii, 1] + FullList[iii, 2] + FullList[iii, 3] + FullList[iii, 4])

        # Update TotGamma4 by adding gamma
        TotGamma4 += gamma

    return Cumul_delay2, TotGamma2, Cumul_delay3, TotGamma3, Cumul_delay4, TotGamma4, DemandS, Demands_rp, gamma