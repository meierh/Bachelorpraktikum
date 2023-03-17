import numpy as np

from network import Network
from rank import Rank


def random_3d_position(spacial_span: np.ndarray):
    """
    Computes a random position in the specified 3D-space.
    :param spacial_span: contains minimal and maximal values for all dimensions
    :return: the position as a tuple of x, y and z value
    """
    min_x, max_x = spacial_span[0][0], spacial_span[1][0]
    min_y, max_y = spacial_span[0][1], spacial_span[1][1]
    min_z, max_z = spacial_span[0][2], spacial_span[1][2]

    x = np.random.default_rng().uniform(min_x, max_x)
    y = np.random.default_rng().uniform(min_y, max_y)
    z = np.random.default_rng().uniform(min_z, max_z)

    return x, y, z


def generate_network(num_nodes: int, spacial_span: np.ndarray, max_ranks: int):

    network = Network(max_ranks, num_nodes, spacial_span)

    # generate nodes for each rank
    for rank in network.ranks:
        for i in range(network.nodes_per_rank()):
            spacial_span_rank = np.ndarray((2, 3))
            spacial_span_rank[0] = rank.position
            for j in range(3):
                spacial_span_rank[1][j] = rank.position[j] + network.dim_of_rank[j]

            network.add_node(random_3d_position(spacial_span_rank), rank.rank_id)

    # generate edges
    for rank in network.ranks:
        for node in rank.nodes:
            # pull number of edges based on normal distribution
            num_edges = int(np.round(np.random.default_rng().normal(6.35, 1.5)))
            id_list = [node[0]]  # init with own id to prevent edge to itself
            for i in range(num_edges):
                # pull target rank based on normal distribution
                # modulo number of ranks to prevent negative values and outliers
                target_rank = int(np.round(np.random.default_rng().normal(rank.rank_id, network.num_ranks/7))) \
                              % network.num_ranks
                # pull target id based on uniform distribution
                index = int(np.round(np.random.default_rng().uniform(0, network.nodes_per_rank() - 1)))
                target_id = network.ranks[target_rank].get_node(index)[0]
                # skip duplicate edges (ids are network global)
                if target_id in id_list:
                    continue
                else:
                    id_list.append(target_id)
                # add edge to out edges of this rank
                rank.add_out_edge(rank.rank_id, node[0], target_rank, target_id, 1)
                # add edge to in edges of target_rank
                network.ranks[target_rank].add_in_edge(rank.rank_id, node[0], target_rank, target_id, 1)

    return network


def merge_ranks(rank1: Rank, rank2: Rank, num_ranks: int):

    if rank1.rank_id < rank2.rank_id:
        rank = Rank(rank1.rank_id, rank1.position)
    else:
        rank = Rank(rank2.rank_id, rank2.position)

    rank.nodes = rank1.nodes + rank2.nodes

    # update ranks in edges
    edges = [rank1.in_edges, rank2.in_edges, rank1.out_edges, rank2.out_edges]
    switch = 0

    for e in edges:
        for edge in e:
            if edge[0] >= num_ranks:
                s_rank = edge[0] - num_ranks
            else:
                s_rank = edge[0]

            if edge[2] >= num_ranks:
                t_rank = edge[2] - num_ranks
            else:
                t_rank = edge[2]

            if switch < 2:
                rank.add_in_edge(s_rank, edge[1], t_rank, edge[3], edge[4])
            else:
                rank.add_out_edge(s_rank, edge[1], t_rank, edge[3], edge[4])
        switch += 1

    return rank


def half_number_ranks(net_old: Network):

    half_size = int(len(net_old.ranks)/2)
    network = Network(half_size, net_old.num_nodes, net_old.spacial_span)

    for i in range(half_size):
        network.ranks[i] = merge_ranks(net_old.ranks[i], net_old.ranks[i+half_size], half_size)

    return network
