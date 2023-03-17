#pragma once

#include "CommunicationPatterns.h"
#include "DistributedGraph.h"
#include "MPIWrapper.h"
#include "NodeCounter.h"
#include <algorithm>
#include <cassert>
#include <functional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

class NetworkMotifs {
public:
	/*|||-----------------------NetworkMotifs--------------------------------
	 *
	 * Function to compute the fraction of networkMotifs of the graph
	 *
	 * Returns: The fraction of NetworkMotifs in regard to their total count using
	 *          the network motif types and numeration according to paper
	 *  "A tutorial in connectome analysis: Topological and spatial features of brain networks"
	 *  by Marcus Kaiser, 2011 in NeuroImage, (892-907), page 899
     *          result[0] is the total count of motifs
     *          result[i] is the fraction of motif i [1...13] (total count motif i / result[0])
	 *
     * Parameters 
     * graph:           A DistributedGraph (Function is MPI compliant)
     * 
	 * MPI Constraints: Function must be called on all ranks simultaneously
	 * 					Function returns correct information to all ranks
     */
    static std::array<long double,14> compute_network_triple_motifs
    (
        DistributedGraph& graph,
        unsigned int resultToRank = 0
    );
    /*-------------------------NetworkMotifs----------------------------------|||*/

	static std::array<long double, 14> compute_network_triple_motifs_sequential(DistributedGraph& graph, unsigned int my_rank = 0);

private:
	struct StdPair_hash {
		template <class T1, class T2>
		std::size_t operator()(const std::pair<T1, T2>& p) const {
			return std::hash<T1>{}(p.first) ^ std::hash<T2>{}(p.second);
		}
	};

	typedef struct {
		std::uint64_t node_3_rank;
		std::uint64_t node_3_local;
		std::uint16_t motif_type_bit_array = 0;

		void self_test();
		void set_motif_types(std::vector<int> motif_types);
		void unset_motif_types(std::vector<int> motif_types);
		void unset_all_but_motif_types(std::vector<int> motif_types);
		bool is_motif_type_set(int motif_type);
		void print_out_complete();
		void print_out();
		bool check_validity();
	} ThreeMotifStructure;

	static std::uint16_t update_edge_bit_array(const DistributedGraph& graph, std::uint16_t exists_edge_bit_array,
						  unsigned int node_2_rank, std::uint64_t node_2_local,
						  unsigned int node_3_rank, std::uint64_t node_3_local);
};
