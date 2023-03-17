#pragma once

#include "CommunicationPatterns.h"
#include "DistributedGraph.h"
#include "MPIWrapper.h"
#include <cassert>
#include <functional>
#include <stdexcept>
#include <unordered_map>


/*
 * --------------- General ---------------
 * 
 * This implementation requires that the provided graph
 * - has no nodes with self-referencing edges
 * - has no edge duplicates
 * Otherwise, it can lead to incorrect results or error throwing.
 *
 *	distributed_8 (old format):
 *	- self-referencing edges exist (e.g. distributed_8: rank_0_in_edges.txt, Line 223)
 *
 *	large_graphs (neues format):
 *	- edge-duplicates exist (e.g. largeGraph: 32/network/rank_00_out_network.txt, Line 7-8)
 *
 *      => This properties can be analyzed with the AlgorithmTests::check_graph_property() method
*/

class Modularity {
public:
	/*|||-----------------------Modularity--------------------------------
	 *
	 * Function to compute the modularity of the graph
	 *
	 * Returns: The Modularity according to paper
	 *  "A tutorial in connectome analysis: Topological and spatial features of brain networks"
	 *  by Marcus Kaiser, 2011 in NeuroImage, (892-907), page 898
	 *
	 * Parameters
	 * graph:           A DistributedGraph (Function is MPI compliant)
	 * 
	 * MPI Constraints: Function must be called on all ranks simultaneously
	 * 					Function returns correct information to all ranks
	 */
	static double compute_modularity(DistributedGraph& graph);
	
	static double compute_modularity_sequential(const DistributedGraph& graph);
	/*-------------------------Modularity----------------------------------|||*/

private:
	typedef struct {
		std::uint64_t node_in_degree;
		std::uint64_t node_out_degree;
		std::uint64_t area_globalID;
	} NodeModularityInfo;
};
