from rank import Rank
from morton_code import *


class Network:

    def __init__(self, num_ranks: int, num_nodes: int, spacial_span: np.ndarray):

        self.spacial_span = spacial_span  # dimensions of network space
        self.node_id = 0  # is inconsistent when network is copied

        # check if number of ranks is 2^n
        if num_ranks == int(num_ranks):
            self.num_ranks = num_ranks  # number of ranks is maximum number of ranks
        else:
            self.num_ranks = int(pow(2, round(np.log2(num_ranks))))
            print("The maximum number of ranks was set to", self.num_ranks)

        # check if number of nodes is n * number of ranks
        if num_nodes % num_ranks == 0:
            self.num_nodes = num_nodes
        else:
            self.num_nodes = int(round(num_nodes / num_ranks) * num_ranks)
            print("The number of nodes was set to", self.num_nodes)

        # calculate dimensions of each rank
        x_parts, y_parts, z_parts = inverse_3d_morton_code(self.num_ranks - 1)
        x_parts += 1
        y_parts += 1
        z_parts += 1

        # x_size = (x_max - x_min) / x_parts
        x_size = (self.spacial_span[1][0] - self.spacial_span[0][0]) / x_parts
        # y_size = (y_max - y_min) / y_parts
        y_size = (self.spacial_span[1][1] - self.spacial_span[0][1]) / y_parts
        # z_size = (z_max - z_min) / z_parts
        z_size = (self.spacial_span[1][2] - self.spacial_span[0][2]) / z_parts

        self.dim_of_rank = x_size, y_size, z_size

        self.ranks = []  # list of ranks
        for i in range(num_ranks):
            # via Morton code assign position to rank
            x_rank_pos, y_rank_pos, z_rank_pos = inverse_3d_morton_code(i)
            position = x_rank_pos * self.dim_of_rank[0], \
                y_rank_pos * self.dim_of_rank[1], \
                z_rank_pos * self.dim_of_rank[2]
            new_rank = Rank(i, position)
            self.ranks.append(new_rank)

    def nodes_per_rank(self) -> int:
        return int(self.num_nodes / self.num_ranks)

    def add_node(self, node: (float, float, float), rank_id: int):
        self.ranks[rank_id].add_node(self.node_id, node)
        self.node_id += 1

    def add_edge(self, edge: (int, int, int), rank_id):
        self.ranks[rank_id].in_edges.append(edge)
