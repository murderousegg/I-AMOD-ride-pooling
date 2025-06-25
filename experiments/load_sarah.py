import numpy as np
import itertools
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy import io
from scipy import sparse
import networkx as nx
from Utilities.RidePooling.LTIFM2_SP import LTIFM2_SP
from Utilities.RidePooling.LTIFM3_SP import LTIFM3_SP
from Utilities.RidePooling.LTIFM4_SP import LTIFM4_SP
from Utilities.RidePooling.LTIFM_reb import LTIFM_reb, LTIFM_reb_damp
from Utilities.RidePooling.calculate_gamma import calculate_gamma
from Utilities.RidePooling.probcomb import probcombN
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import h5py
import time

import src.tnet as tnet
import matplotlib.pyplot as plt
import src.CARS as cars
import networkx as nx
import re
import copy
from gurobipy import *

data = io.loadmat("Eindhoven/data_eind.mat", squeeze_me=True)
# print(data)
graph_data = io.loadmat("Eindhoven/graph_eind.mat", squeeze_me=True)
node_types = graph_data.get('node_type')
print(sum(node_types == 'b '))
print(sum(node_types == 'pt'))
print(sum(node_types == 'w '))
print(sum(node_types == 'c '))
ped_nodes = [i for i in range(len(node_types)) if node_types[i] == 'w ']
bike_nodes = [i for i in range(len(node_types)) if node_types[i] == 'b ']
pt_nodes = [i for i in range(len(node_types)) if node_types[i] == 'pt']
car_nodes = [i for i in range(len(node_types)) if node_types[i] == 'c ']
print(ped_nodes)