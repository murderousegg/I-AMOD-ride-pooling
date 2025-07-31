# I-AMOD-ride-pooling
This project proposes an approach to synthesize the works by [Paparella et al.](https://ieeexplore.ieee.org/document/10605118) on ride-pooling and [Wollenstein-Betech et al., 2021](https://ieeexplore.ieee.org/document/9541261) on Intermodal Autonomous Mobility on Demand (I-AMoD). The paper for this project can be found here (add link when paper ready)

# File organisation

Data is split into 3 parts: network data in `data\net`, positional data for nodes in `data\pos` and trip (demand) data in `data\trips`. While there are toy datasets for NYC and some other cities, the main focus is on the full dataset NYC. 
Preprocessing scripts, as well as the different scripts for optimizing the network can be found in `experiments\`. Here, `reduce_roadgraph.py` prunes nodes from the NYC dataset and stores the new roadgraph in `data\gml`. `create_save_NYC.py` then creates the supergraph for the NYC dataset, and stores it in `data\gml` as well. The final "preprocessing" step is performed in `create_rp_list_small.py`, which creates and stores the database of all possible ride-pooling combinations.
During this last step, a new folder `NYC` is created that stores all the data for the ride-pooling combinations. Note that `create_rp_list_small.py` can be run in parts, as some of the steps are saved. 
The `src\` folder contains some helper libraries. `src\tnet.py` contains the backbone of the supergraph functionality, `src\solvers.py` contains the I-AMoD solvers, and `src\LTIFM_reb.py` is the ride-pooling solver for the vehicle routing part. `src\simConfig.py` contains a configuration class that contains some tunable hyperparameters, and `src\simulationCore.py` contains the core class for the bi-level problem. 

# Running the scripts

Before running the expirements, make sure requirements are installed from the `requirements.txt` file:

`pip install -r requirements.txt`

Next, any of the experiments can be run using 

`python -m experimentst.[expirement_name]`

or `python3` depending on the installation. Since the new york dataset is quite large, it is recommended to offload the work onto a server, or at least use a computer with a decent amount of ram. (You can run the NYC dataset on a laptop, however only when the roadgraph is pruned to a low amount of nodes.)

