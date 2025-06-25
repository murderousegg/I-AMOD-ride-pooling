# I-AMOD-ride-pooling
This project proposes an approach to synthesize the works by [Paparella et al.](https://ieeexplore.ieee.org/document/10605118) on ride-pooling and [Wollenstein-Betech et al., 2021](https://ieeexplore.ieee.org/document/9541261) on Intermodal Autonomous Mobility on Demand (I-AMoD). Both approaches will be combined to form a model that optimizes traffic flow for NYC, with data taken from [source]. The paper for this project can be found here (add link when paper ready)

# File organisation and running the code

As of 20-12, the code is yet to be combined. The ride-pooling approach, which was originally written in MatLab, is included as Ride-pooling.py, and its uitlity functions are included in /Utilities/RidePooling/. The datasets stored in /NYC20/ and /NYC250/ are copied from the Paparalla, but the digraph data is modified beforehand using Utilities/convert_digraph.m, as python does not support the original matlab format for the digraph. To run this on one of the datasets, modify CITY_FOLDER at the top of Ride-pooling.py to the name of the folder that you want to run. If only parts of the algorithm need to be run, simply comment out the functions from main() at the bottom of the file. By default, the code analyses ride-pooling with a max of 2, 3 and 4 people.

## Intermodal Mobility on Demand
The code from [Wollenstein-betech et al., 2021] is currently copied into /intermodal/. To run an expirement, a command needs to be entered into the console:
`python3 -m experiments.{name of the file without extension}`
Note that there are many different types of experiments listed in the expiriments folder. In the future, only the necessary parts of the code will be reused. Additionally, all of data used in that paper are in a different structure than the data from [Paparella et al.]. To solve this issue, the code still needs to be modified to some extend in order to accept the new formatting. 

Requirements: gurobipy, networkx, scipy, numpy, pwlf, joblib, h5py

# Main functionality
The main expiriment can be run by using
`python3 -m experiments.ride-pooling`
First, a network is definied similar to the networks described by [Wollenstein-betech et al., 2021]. Along with the normal modes of transport (walking, public transport, cars), a ride-pooling layer is created. Then, all spatially feasible bags/sequences using the algorithm proposed by [Paparella et al., 2024] are computed for the ride-pooling layer. Each sequence corresponds to a selection variable in the optimization problem, and a demand matrix can be created as a linear combination of all sequences (that are selected). 
