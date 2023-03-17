class Rank:

    def __init__(self, rank_id, position: (float, float, float)):
        self.rank_id = rank_id
        self.position = position  # min x, y, z that are part of this rank
        self.nodes = []  # node = id, (x, y, z)
        # edge = (source, target, weight)
        # source and target = (rank_id, node_id)
        self.in_edges = []
        self.out_edges = []

    def add_node(self, node_id: int, node_position: (float, float, float)):
        self.nodes.append((node_id, node_position))

    def get_node(self, index: int):
        return self.nodes[index]

    def add_in_edge(self, source_rank: int, source_id: int, target_rank: int, target_id: int, weight: int):
        self.in_edges.append((source_rank, source_id, target_rank, target_id, weight))

    def add_out_edge(self, source_rank: int, source_id: int, target_rank: int, target_id: int, weight: int):
        self.out_edges.append((source_rank, source_id, target_rank, target_id, weight))
