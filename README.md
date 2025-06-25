# I-AMOD-ride-pooling
This project proposes an approach to synthesize the works by [Paparella et al.](https://ieeexplore.ieee.org/document/10605118) on ride-pooling and [Wollenstein-Betech et al., 2021](https://ieeexplore.ieee.org/document/9541261) on Intermodal Autonomous Mobility on Demand (I-AMoD). Both approaches will be combined to form a model that optimizes traffic flow for NYC, with data taken from [source]. The paper for this project can be found here (add link when paper ready)

# File organisation and running the code

Data for some networks is stored in the data folder. The main focus of the project is NYC, which has its own specific folder which also includes subway data. None of the other data sources has any public transit data. The code that runs the experiments can be found in the expirements folder. The code is currently setup to run NYC. To get this to work, make sure that you run reduce_roadgraph.py first. This removes many points from the roadrgraph and groups OD's to the closest point in the new roadgraph. This smaller graph is only used to create the ride-pooling layer.

Next, run create_save_NYC.py. This compiles the supergraph and stores it in a pickeled file. Now, 3 pickled files should exist under data/gml. By then running create_rp_list_small.py, a giant array can be created which contains all possible ridepooling sequences, along with the order, delay and cost. This is stored as a compressed numpy file (.npz) in NYC/. 

Now all the preparation is done, and the optimization can begin. To optimize for a 100/% penetration rate, run disjoint_refactor.py. This will perform the disjoint optimization strategy as outlined in the paper. To run the penetration rate problem, run disjoint_penrate_refactor.py. I should warn you, this will probably take an insane amount of time. To reduce runtime, play with the amount of penetration rates and the depth of the Stackelberg game.

Results are stored in the /results/ folder. All results are stored in .csv files, which can be used for plotting later. Additionally, a capture of the supergraph is stored as a pickled file. This can be used to visualize the network, or to find individual routes.

To run the files, simply type `python -m experiments.[experiment_name]` (or `python3` depending on installation). However, it is highly recommended to do most of the work on a server given the amount of memory required and the benefits of multiprocessing.