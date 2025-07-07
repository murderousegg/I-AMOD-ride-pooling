import numpy as np

def LTIFM2_SP(jj1, ii1, jj2, ii2, solPart, node_order):
    sol1 = {}
    sol1["obj"] = 0 #%solPart{jj1,ii1}.obj + solPart{jj2,ii2}.obj;  difference wrt no rp 
    sol1["order"]= [1, 1, 2, 2]
    sol1["Delay"] = [0,0];  #solPart{jj1,ii1}.obj

    sol2 = {}
    sol2["obj"] = solPart[jj1, jj2]["obj"] + solPart[jj2, ii1]["obj"] + solPart[ii1, ii2]["obj"] - solPart[jj1, ii1]["obj"] - solPart[jj2, ii2]["obj"]
    sol2["order"]= [1, 2, 1, 2]
    sol2["Delay"] = [solPart[jj1, jj2]["obj"] + solPart[jj2, ii1]["obj"] - solPart[jj1, ii1]["obj"], solPart[jj2, ii1]["obj"] + solPart[ii1, ii2]["obj"] - solPart[jj2, ii2]["obj"]]

    sol3 = {}
    sol3["obj"] = solPart[jj1, jj2]["obj"] + solPart[jj2, ii2]["obj"] + solPart[ii2, ii1]["obj"] - solPart[jj1, ii1]["obj"] - solPart[jj2, ii2]["obj"]
    sol3["order"]=[1, 2, 2, 1]
    sol3["Delay"] = [solPart[jj1, jj2]["obj"] + solPart[jj2, ii2]["obj"] + solPart[ii2, ii1]["obj"] - solPart[jj1, ii1]["obj"],0]

    # if jj1 == 3 and jj2 == 1 and ii1 == 24 and ii2 == 21:
    #     print(sol1["Delay"])
    #     print(sol2["Delay"])
    #     print(sol3["Delay"])
    #     print(sol2["obj"])
    #     print(sol3["obj"])
    #     asdfs
    # iterators = [jj1+1, ii1+1, jj2+1, ii2+1]
    iterators = [node_order[jj1], node_order[ii1], node_order[jj2], node_order[ii2]]
    if sol2["obj"] < 0:
        sol = [sol2["obj"]]
        sol.extend(sol2["Delay"])
        sol.extend(iterators)
        sol.extend(sol2["order"])
    elif  sol3["obj"] < 0:
        sol = [sol3["obj"]]
        sol.extend(sol3["Delay"])
        sol.extend(iterators)
        sol.extend(sol3["order"])
    else:
        sol = [sol1["obj"]]
        sol.extend(sol1["Delay"])
        sol.extend(iterators)
        sol.extend(sol1["order"])
    return sol

import numpy as np

def LTIFM2_SP_dests(jj1, ii1, jj2, ii2, solPart, node_order, dest_order):
    sol1 = {}
    sol1["obj"] = 0 #%solPart{jj1,ii1}.obj + solPart{jj2,ii2}.obj;  difference wrt no rp 
    sol1["order"]= [1, 1, 2, 2]
    sol1["Delay"] = [0,0];  #solPart{jj1,ii1}.obj

    sol2 = {}
    sol2["obj"] = solPart[jj1, jj2]["obj"] + solPart[jj2, ii1]["obj"] + solPart[ii1, ii2]["obj"] - solPart[jj1, ii1]["obj"] - solPart[jj2, ii2]["obj"]
    sol2["order"]= [1, 2, 1, 2]
    sol2["Delay"] = [solPart[jj1, jj2]["obj"] + solPart[jj2, ii1]["obj"] - solPart[jj1, ii1]["obj"], solPart[jj2, ii1]["obj"] + solPart[ii1, ii2]["obj"] - solPart[jj2, ii2]["obj"]]

    sol3 = {}
    sol3["obj"] = solPart[jj1, jj2]["obj"] + solPart[jj2, ii2]["obj"] + solPart[ii2, ii1]["obj"] - solPart[jj1, ii1]["obj"] - solPart[jj2, ii2]["obj"]
    sol3["order"]=[1, 2, 2, 1]
    sol3["Delay"] = [solPart[jj1, jj2]["obj"] + solPart[jj2, ii2]["obj"] + solPart[ii2, ii1]["obj"] - solPart[jj1, ii1]["obj"],0]

    # if jj1 == 3 and jj2 == 1 and ii1 == 24 and ii2 == 21:
    #     print(sol1["Delay"])
    #     print(sol2["Delay"])
    #     print(sol3["Delay"])
    #     print(sol2["obj"])
    #     print(sol3["obj"])
    #     asdfs
    # iterators = [jj1+1, ii1+1, jj2+1, ii2+1]
    iterators = [node_order[jj1], dest_order[ii1], node_order[jj2], dest_order[ii2]]
    if sol2["obj"] < 0:
        sol = [sol2["obj"]]
        sol.extend(sol2["Delay"])
        sol.extend(iterators)
        sol.extend(sol2["order"])
    elif  sol3["obj"] < 0:
        sol = [sol3["obj"]]
        sol.extend(sol3["Delay"])
        sol.extend(iterators)
        sol.extend(sol3["order"])
    else:
        sol = [sol1["obj"]]
        sol.extend(sol1["Delay"])
        sol.extend(iterators)
        sol.extend(sol1["order"])
    return sol

