import os

from network import Network


def create_positions_files(directory: os.path, network: Network):
    """
    Creates the positions files based on the given network
    and writes it to the given directory.
    :param directory: the directory
    :param network: the network
    """

    for rank in network.ranks:
        # create the file
        zeroes = len(str(network.num_ranks-1)) - len(str(rank.rank_id))
        file = open(str(directory) + "\\" + "rank_" + "0" * zeroes + str(rank.rank_id) + "_positions.txt", "x")

        # write the file header
        header = ("# " + str(network.nodes_per_rank()) + " of " + str(network.num_nodes) + "\n" +
                  "# Minimum x: " + str(network.spacial_span[0][0]) + "\n" +
                  "# Minimum y: " + str(network.spacial_span[0][1]) + "\n" +
                  "# Minimum z: " + str(network.spacial_span[0][2]) + "\n" +
                  "# Maximum x: " + str(network.spacial_span[1][0]) + "\n" +
                  "# Maximum y: " + str(network.spacial_span[1][1]) + "\n" +
                  "# Maximum z: " + str(network.spacial_span[1][2]) + "\n" +
                  "# <local id> <pos x> <pos y> <pos z> <area> <type>\n")

        file.write(header)

        # insert each node as a line into the file
        for i in range(len(rank.nodes)):
            x, y, z = rank.nodes[i][1]  # node = id, (x, y, z)
            file.write(
                str(i + 1) + " " + str(x) + " " + str(y) + " " + str(z) + " " + "random" + " " + "ex" + "\n"
            )

        file.close()


def write_network_header(file, num_nodes_total, num_nodes_rank, num_ranks):
    header = "# Total number neurons: " + str(num_nodes_total) + "\n" + \
             "# Local number neurons: " + str(num_nodes_rank) + "\n" + \
             "# Number MPI ranks: " + str(num_ranks) + "\n" + \
             "# <target_rank> <target_id> <source_rank> <source_id> <weight> \n"

    file.write(header)


def create_network_files(directory: os.path, network: Network):
    """
    Creates the network files based on the given network
    and writes it to the given directory.
    :param directory: the directory
    :param network: the network
    """

    for rank in network.ranks:

        zeroes = len(str(network.num_ranks - 1)) - len(str(rank.rank_id))

        # create in_network and out_network file
        for i in range(2):
            if i == 0:
                suffix = "_in_network.txt"
                edges = rank.in_edges
            else:
                suffix = "_out_network.txt"
                edges = rank.out_edges

            file = open(str(directory) + "\\" + "rank_" + "0" * zeroes + str(rank.rank_id) + suffix, "x")

            # write header
            write_network_header(file, network.num_nodes, len(rank.nodes), network.num_ranks)

            # write line for each edge
            for edge in edges:
                target_rank, target_id, source_rank, source_id, weight = edge[2], edge[3], edge[0], edge[1], edge[4]
                file.write(
                    str(target_rank) + " " + str(target_id) + " " + str(source_rank) + " " + str(source_id)
                    + " " + str(weight) + "\n"
                )

            file.close()


def create_files(parent_dir: str, network: Network):
    """
    Creates the files based on the given network
    and writes it to the given directory.
    :param parent_dir: the directory
    :param network: the network
    """

    # create folder structure
    # maybe add parent dir with timestamp
    proc_path = os.path.join(parent_dir, str(network.num_ranks))
    network_path = os.path.join(proc_path, "network")
    positions_path = os.path.join(proc_path, "positions")

    os.mkdir(proc_path)
    os.mkdir(network_path)
    os.mkdir(positions_path)

    create_positions_files(positions_path, network)
    create_network_files(network_path, network)
