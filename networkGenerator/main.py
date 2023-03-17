import numpy as np

from file_writing import *
from network_generation import *
from morton_code import *


def main():
    # specify network parameters (distributions?)
    parent_dir = input("Enter the path to a directory to store the network\n"
                       "Example: C:\\Users\\user\\Downloads\\generated_nets\n")

    num_nodes = int(input("Enter total number of nodes\n"))
    different_ranks = int(input("Enter the number of different ranks that the network should be distributed on\n"))
    ranks = []
    print("Enter number of ranks one by one\n")
    for i in range(different_ranks):
        ranks.append(int(input()))

    ranks.sort(reverse=True)

    # take user input for minimum and maximum x, y, z values
    spacial_span = np.ndarray((2, 3), dtype=float)

    if input("Do you want to use the standard space (56 x 56 x 56)? Otherwise you must define your own.\n"
             "(Answer with y/n)\n") == "y":
        spacial_span = [[0, 0, 0], [56, 56, 56]]
    else:
        spacial_span[0][0] = input("Minimum x value (float): ")
        spacial_span[1][0] = input("Maximum x value (float): ")
        spacial_span[0][1] = input("Minimum y value (float): ")
        spacial_span[1][1] = input("Maximum y value (float): ")
        spacial_span[0][2] = input("Minimum z value (float): ")
        spacial_span[1][2] = input("Maximum z value (float): ")

    print("Generating...")

    network = generate_network(num_nodes, spacial_span, ranks[0])
    print("The network was created successfully!")

    # create files for different rank partitions
    for num_rank in ranks:
        for i in range(int(np.log2(network.num_ranks/num_rank))):
            network = half_number_ranks(network)

        create_files(parent_dir, network)
        print("The files for " + str(network.num_ranks) + " ranks were created successfully!")


if __name__ == '__main__':
    main()
