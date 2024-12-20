%% convert the digraph to a python compatible format
city = 'NYC250'; % 'SF'
load(strcat(city,'/Graphs.mat'));
digraph = G_road;

% export its edge list and nodes
edge_list = G_road.Edges.EndNodes;  % Get edge list (source, target)
weight_list = G_road.Edges.Weight;
edge_list = [edge_list weight_list];
adj_matrix = adjacency(G_road);     % Get adjacency matrix
save(strcat(city, '/digraph_data.mat'), 'edge_list', 'adj_matrix');